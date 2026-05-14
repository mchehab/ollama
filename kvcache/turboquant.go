package kvcache

import (
	"context"
	"log/slog"
	"math"
	"os"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/turboquant"
)

// forceFused opts the K+V CUDA path back into fused inline-decode (path 2).
// The default for K+V on CUDA is DequantKV + stock FA (path 1): faster on
// Pascal (4.4× decode, 8.95× prefill) and bit-identical in PPL across
// llama3.1/3.2 and qwen2.5 at long context. OLLAMA_TQ_FORCE_FUSED=1 is for
// benchmarking and HIP-portability work.
var forceFused = os.Getenv("OLLAMA_TQ_FORCE_FUSED") != ""

type TurboQuantCache struct {
	meta      *Causal
	preset    turboquant.Preset
	isReserve bool

	compressedK ml.TQCompressedKManager

	// phase2Checked ensures GPU encode is activated at most once.
	phase2Checked bool

	headDim    int
	numKVHeads int

	// CONTRACT for encodeResults / vEncodeResults: these maps are written by
	// Put and read by Get on a SINGLE goroutine per cache instance. This
	// matches the runner's per-step model-evaluation loop (Init → Put-per-layer
	// → Get-per-layer → Compute → Reset, all serial), and mirrors the same
	// goroutine-affinity contract that the inner *Causal cache already
	// assumes for its own per-layer tensor maps (c.keys, c.values, c.ctxs).
	// Violating this contract — e.g. dispatching layers across goroutines, or
	// overlapping a reserve pass with an inference pass — will trip Go's
	// runtime concurrent-map-access detector and panic.

	// encodeResults stores per-layer EncodeK result tensors for the current
	// forward pass. DequantK uses them as src[0] to establish the graph
	// dependency (encode before dequant in the ggml scheduler).
	encodeResults map[int]ml.Tensor

	// vEncodeResults stores per-layer EncodeV result tensors for the current
	// forward pass. DequantV uses them to establish the encode→dequant ordering.
	vEncodeResults map[int]ml.Tensor

	// logPathOnce ensures each active Get() path is logged at most once per
	// cache instance (avoids log spam: Get() is called every layer every step).
	logPathOnce [8]sync.Once

	// fusedFallbackEligible gates the inline-decode fused-FA fallback paths
	// (Get paths 2 and 4). CUDA/ROCm and Metal all support D=64, 128, 256, 512 —
	// covering llama3.2 (64), llama3.1/qwen2.5 (128), Gemma3/Gemma4 local-attn
	// (256), and Gemma4 global-attn (512). Other dims fall back to the DequantK
	// + stock FA path (Get paths 0/1/5).
	fusedFallbackEligible bool

	// preferFusedAttn is true on Metal. For K+V WHT presets the DequantKV →
	// stock FA path (Path 1) materialises a full f16 intermediate, doubling
	// KV bandwidth at long context. The fused inline-decode kernel (Path 2)
	// reads packed K+V once with no intermediate. On Metal the fused path is
	// dramatically faster; on CUDA/ROCm the intermediate stays in L2 and
	// stock FA is highly tuned, so DequantKV wins. When this flag is true
	// the Get() router skips Path 1 and uses Path 2 for K+V presets.
	preferFusedAttn bool

	// curQueryLen is the number of query tokens in the current forward pass,
	// captured at StartForward from len(batch.Positions). curQueryLen==1
	// means decode; curQueryLen>1 means prefill (or a batched prompt chunk).
	curQueryLen int

	// pendingKBiases buffers K projection biases set by SetLayerKBias before
	// activateGPUEncode has initialised compressedK. Applied to compressedK
	// once the manager is created.
	pendingKBiases map[int]ml.Tensor
}

// isSWACausal reports whether a *Causal has sliding-window attention
// active. Plain Causal caches have swaWindowSize either 0 (before Init
// normalizes the default) or math.MaxInt32 (after); SWA constructors set
// it to the actual window size.
func isSWACausal(c *Causal) bool {
	return c.swaWindowSize > 0 && c.swaWindowSize != math.MaxInt32
}

// AttentionKVWrapper is implemented by caches that embed *Recurrent and
// expose the attention half of a hybrid (SSM/recurrent + attention) cache.
// WrapWithTurboQuant uses it to inject TurboQuant compression into the
// attention KV path without disturbing conv/recurrent state buffers.
// *kvcache.Recurrent implements this interface, so any model HybridCache that
// embeds *Recurrent satisfies it automatically via Go method promotion.
type AttentionKVWrapper interface {
	AttentionKV() *Causal
	SetAttentionKV(Cache)
}

// WrapWithTurboQuant returns a cache that applies TurboQuant compression to
// global-attention Causal layers and a bool reporting whether any wrapping
// took effect. For a top-level *Causal (non-SWA), it returns a new
// *TurboQuantCache. For a *WrapperCache, it mutates the caches slice in
// place, replacing every non-SWA *Causal sub-cache with a *TurboQuantCache,
// and returns the same *WrapperCache pointer. This enables TQ on SWA models
// like gemma3/gemma4 where the global attention layers dominate KV memory
// at long context. Returns (cache, false) if no eligible sub-caches were
// found.
func WrapWithTurboQuant(cache Cache, preset turboquant.Preset) (Cache, bool) {
	switch c := cache.(type) {
	case *Causal:
		// Reject SWA caches. Plain NewCausalCache leaves swaWindowSize=0
		// until Init() normalizes it to math.MaxInt32; SWA constructors set
		// it to the actual window size. "Plain causal" means the field is
		// either 0 (uninitialized default) or math.MaxInt32 (post-Init).
		if isSWACausal(c) {
			slog.Warn("turboquant: top-level Causal is sliding-window, cannot wrap")
			return cache, false
		}
		return &TurboQuantCache{
			meta:           c,
			preset:         preset,
			encodeResults:  make(map[int]ml.Tensor),
			vEncodeResults: make(map[int]ml.Tensor),
		}, true

	case *WrapperCache:
		// Mutate sub-caches in place: replace every *Causal (including SWA)
		// with a *TurboQuantCache. The TQ Put path takes the indexed
		// addressing route (EncodeKAt/EncodeKVAt) whenever curLocs is
		// fragmented, so SWA eviction's scattered free cells are written
		// correctly. All ~60 gemma3/gemma4 layers benefit from compression.
		wrapped := 0
		for i, sub := range c.caches {
			inner, ok := sub.(*Causal)
			if !ok {
				continue
			}
			c.caches[i] = &TurboQuantCache{
				meta:           inner,
				preset:         preset,
				encodeResults:  make(map[int]ml.Tensor),
				vEncodeResults: make(map[int]ml.Tensor),
			}
			wrapped++
		}
		if wrapped == 0 {
			slog.Warn("turboquant: no eligible Causal sub-caches in WrapperCache, falling back to unwrapped cache")
			return cache, false
		}
		slog.Info("turboquant: wrapped Causal sub-caches inside WrapperCache",
			"count", wrapped, "preset", preset.Name)
		return cache, true

	case AttentionKVWrapper:
		inner := c.AttentionKV()
		if inner == nil {
			slog.Warn("turboquant: hybrid cache inner kv is not *Causal (already wrapped?), leaving as-is")
			return cache, false
		}
		if isSWACausal(inner) {
			slog.Warn("turboquant: hybrid cache inner *Causal is sliding-window, cannot wrap")
			return cache, false
		}
		c.SetAttentionKV(&TurboQuantCache{
			meta:           inner,
			preset:         preset,
			encodeResults:  make(map[int]ml.Tensor),
			vEncodeResults: make(map[int]ml.Tensor),
		})
		slog.Info("turboquant: wrapped attention KV in hybrid recurrent cache",
			"preset", preset.Name)
		return cache, true

	default:
		slog.Warn("turboquant: underlying cache is not *Causal or *WrapperCache, falling back to unwrapped cache")
		return cache, false
	}
}

func (c *TurboQuantCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	// K is always compressed; suppress inner Causal from allocating it.
	c.meta.SkipK = true
	// V is compressed when ValueBits > 0; skip the inner Causal V allocation.
	if c.preset.ValueBits > 0 {
		c.meta.SkipV = true
	}
	c.meta.Init(backend, ml.DTypeF16, maxSequences, capacity, maxBatch)
	slog.Info("turboquant cache initialized", "preset", c.preset.Name,
		"K_bits", c.preset.KeyPrimaryBits, "V_bits", c.preset.ValueBits)
}

func (c *TurboQuantCache) Close() {
	if c.compressedK != nil {
		c.compressedK.Close()
		c.compressedK = nil
	}
	c.meta.Close()
}

func (c *TurboQuantCache) SetLayer(layer int)              { c.meta.SetLayer(layer) }
func (c *TurboQuantCache) SetConfig(config ml.CacheConfig) { c.meta.SetConfig(config) }

// FusedEligible reports whether the inline-decode fused kernel path is active.
// Only valid after the first Put() call (which triggers activateGPUEncode).
// Returns false if headDim is not in the supported set — DequantK slow path is active.
func (c *TurboQuantCache) FusedEligible() bool { return c.fusedFallbackEligible }

// SetLayerKBias passes the K projection bias tensor for the given layer to the
// TQ compression manager. Called once per layer at model init for architectures
// that include a bias on the K projection (e.g. Qwen2). The bias is subtracted
// from K before rotation in the encoder; attention correctness is preserved
// because a constant shift in K cancels out in softmax.
func (c *TurboQuantCache) SetLayerKBias(layer int, bias ml.Tensor) {
	if bias == nil {
		return
	}
	if c.compressedK == nil {
		// activateGPUEncode hasn't run yet — buffer for later application.
		if c.pendingKBiases == nil {
			c.pendingKBiases = make(map[int]ml.Tensor)
		}
		c.pendingKBiases[layer] = bias
		return
	}
	type kBiasSetter interface {
		SetLayerKBias(layer int, bias ml.Tensor)
	}
	if kbs, ok := c.compressedK.(kBiasSetter); ok {
		kbs.SetLayerKBias(layer, bias)
	}
}

func (c *TurboQuantCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	c.isReserve = reserve
	c.curQueryLen = len(batch.Positions)
	clear(c.encodeResults)
	clear(c.vEncodeResults)
	return c.meta.StartForward(ctx, batch, reserve)
}

func (c *TurboQuantCache) Put(ctx ml.Context, key, value ml.Tensor) {
	// Capture headDim early — needed even during reserve to size placeholders.
	if c.headDim == 0 && key != nil {
		c.headDim = key.Dim(0)
		c.numKVHeads = key.Dim(1)
	}

	// Activate the GPU encode path on the first Put (reserve or not) once we
	// know headDim/numKVHeads. Doing this during reserve lets EnsureLayer run
	// on the reserve pass, which books TQ's per-layer persistent K/V buffers
	// into btDeviceMemory.Cache[layer] (via newTensor's ctx.layer accounting
	// in EnsureLayer). Without this, the scheduler's fit/alloc probe sees a
	// 0-byte Cache for tq2/tq3 and only the f16 V half for tq2k/tq3k, which
	// is wrong for both headline-footprint reporting and long-context fit math.
	if !c.phase2Checked && c.headDim > 0 {
		c.phase2Checked = true
		c.activateGPUEncode()
	}

	if c.isReserve {
		c.meta.Put(ctx, key, value)
		// Eagerly allocate TQ/i4 persistent buffers during reserve so the
		// scheduler's per-layer Cache totals reflect the real post-compression
		// footprint. Also create the encode graph node so the reserve graph has
		// the same K-branch structure as the inference graph. Without the encode
		// node, gallocr's live-range analysis sees a truncated graph and
		// over-allocates scratch (measured: +222 MiB for asym+outliers presets).
		if c.compressedK != nil {
			layer := c.meta.curLayer
			capacity := len(c.meta.cells)
			c.compressedK.EnsureLayer(layer, capacity)
			if c.preset.ValueBits > 0 {
				c.compressedK.EnsureVLayer(layer, capacity)
				kResult, vResult := c.compressedK.EncodeKV(ctx, layer, key, value, 0)
				if kResult != nil {
					ctx.Forward(kResult)
					c.encodeResults[layer] = kResult
					c.vEncodeResults[layer] = vResult
				}
			} else {
				encodeResult := c.compressedK.EncodeK(ctx, layer, key, 0)
				if encodeResult != nil {
					ctx.Forward(encodeResult)
					c.encodeResults[layer] = encodeResult
				}
			}
		}
		return
	}

	// Decide between contiguous (firstCell+i) and indexed (locs[i]) addressing.
	// Plain Causal caches hand back a contiguous run from findLocs; SWA caches
	// can return fragmented slots. The TQ kernels handle both via the *At
	// variants, which upload the locs tensor and branch block-uniformly.
	firstCell := 0
	contiguous := true
	if len(c.meta.curLocs) > 0 {
		firstCell = c.meta.curLocs[0]
		for i := 1; i < len(c.meta.curLocs); i++ {
			if c.meta.curLocs[i] != firstCell+i {
				contiguous = false
				break
			}
		}
	}

	if c.compressedK != nil {
		layer := c.meta.curLayer
		capacity := len(c.meta.cells)

		c.compressedK.EnsureLayer(layer, capacity)
		if c.preset.ValueBits > 0 {
			c.compressedK.EnsureVLayer(layer, capacity)
		}

		var locsTensor ml.Tensor
		if !contiguous {
			locs32 := make([]int32, len(c.meta.curLocs))
			for i, v := range c.meta.curLocs {
				locs32[i] = int32(v)
			}
			locsTensor = ctx.Input().FromInts(locs32, len(locs32))
		}

		if c.preset.ValueBits > 0 {
			// Combined K+V encode: single GGML op, two back-to-back kernels.
			var kResult, vResult ml.Tensor
			if contiguous {
				kResult, vResult = c.compressedK.EncodeKV(ctx, layer, key, value, firstCell)
			} else {
				kResult, vResult = c.compressedK.EncodeKVAt(ctx, layer, key, value, locsTensor)
			}
			if kResult != nil {
				ctx.Forward(kResult)
				c.encodeResults[layer] = kResult
				c.vEncodeResults[layer] = vResult
			}
		} else {
			// K-only presets (tq2k/tq3k): V stays as f16 in the Causal cache.
			var encodeResult ml.Tensor
			if contiguous {
				encodeResult = c.compressedK.EncodeK(ctx, layer, key, firstCell)
			} else {
				encodeResult = c.compressedK.EncodeKAt(ctx, layer, key, locsTensor)
			}
			if encodeResult != nil {
				ctx.Forward(encodeResult)
				c.encodeResults[layer] = encodeResult
			}
		}

		// Inner Causal.Put() tracks cell metadata (positions, masks).
		// SkipK and SkipV suppress the actual K/V tensor writes.
		c.meta.Put(ctx, key, value)
		return
	}

	c.meta.Put(ctx, key, value)
}

// tqGetPath enumerates the routing decisions that Get() can make.
// Both the inference and reserve branches must agree on the chosen path —
// any divergence causes gallocr to pre-reserve a different scratch shape
// than inference produces, leading to silent memory corruption on long
// contexts (we've been bitten by this before; see feedback memory
// `feedback_reserve_inference_graph_must_match`).
type tqGetPath int

const (
	tqPathNone                tqGetPath = iota // no encoded K — caller falls back to meta or placeholder
	tqPathKOnlyFusedNoAsymm                    // Path 0: K-only, no asymmetric, fused inline-decode
	tqPathKOnlyDequantNoAsymm                  // Path 0 fallback: K-only, no asymmetric, DequantK
	tqPathKVDequant                            // Path 1: K+V combined dequant → stock FA (CUDA/ROCm default)
	tqPathKVFused                              // Path 2: K+V fused inline-decode (Metal default; or OLLAMA_TQ_FORCE_FUSED)
	tqPathKVDequantWHT                         // Path 2b: K+V WHT fallback — DequantK+WHT + DequantV+WHT
	tqPathKOnlyDequantWHT                      // Path 2c: K-only WHT fused dequant+WHT undo
	tqPathKOnlyInlineDecode                    // Path 4: K-only fused inline-decode (Metal/ROCm without 2c)
	tqPathSeparateDequant                      // Path 5: separate K+V dequant — last-resort fallback
)

// tqGetResult captures the outcome of a single routing decision.
type tqGetResult struct {
	path  tqGetPath
	key   ml.Tensor // K (or fused K+V tqTensor when skipMetaValue=true)
	value ml.Tensor // V tensor when produced by the path; nil when V comes from meta cache or is inline
	// skipMetaValue is set when key carries V inline (K+V fused) — caller must
	// NOT pull V from meta or synthesize an f16 V placeholder, otherwise the
	// fused-path's bandwidth win is undone (path 2 inserts a phantom f16 V scratch).
	skipMetaValue bool
}

// routeGet is the single source of truth for how a Get() call dispatches against
// the compressed-K manager. The reserve graph and the inference graph BOTH call
// this with their respective firstCell so gallocr pre-reserves the same scratch
// shape inference produces. Adding/removing/reordering a path here automatically
// keeps both branches in sync.
//
// Returns (tqPathNone, nil, nil, false) when no encoded K exists for this layer
// — caller should fall back to meta.Get / placeholder synthesis.
func (c *TurboQuantCache) routeGet(ctx ml.Context, layer, firstCell, nCells int) tqGetResult {
	// IMPORTANT: this routing must stay identical for reserve and inference.
	// Both branches of Get() call this function with their respective firstCell,
	// and gallocr's pre-reservation depends on it producing the same set of
	// graph nodes inference will produce. New paths go HERE only — never inline
	// a path-decision into Get() directly, or the two branches will silently
	// diverge and gallocr will under- or over-reserve KV scratch.
	if c.compressedK == nil {
		return tqGetResult{path: tqPathNone}
	}
	enc := c.encodeResults[layer]
	if enc == nil {
		return tqGetResult{path: tqPathNone}
	}
	vEnc := c.vEncodeResults[layer]

	// ── K+V presets (preset.ValueBits > 0) ──────────────────────────────
	// Order matches the original inference branch: fused first (when the
	// backend prefers it or the user forced it), then combined DequantKV
	// (CUDA/ROCm default), then the WHT-no-outlier dequant fallback, then
	// separate dequant. Reordering this changes the path taken by ablation
	// configurations like OLLAMA_TQ_DISABLE_OUTLIERS=1 on Metal.
	if c.preset.ValueBits > 0 && vEnc != nil {
		// Path 2: K+V fused inline-decode.
		// Auto-selected on Metal (preferFusedAttn=true) because materialising
		// the f16 intermediate doubles KV bandwidth at long context. Opt-in
		// elsewhere via OLLAMA_TQ_FORCE_FUSED for benchmarking. Reserve must
		// honour forceFused too or scratch sizes diverge.
		if c.fusedFallbackEligible && (forceFused || c.preferFusedAttn) {
			if tqkv, ok := c.compressedK.GetAsTQTensorKV(ctx, layer, enc, vEnc, firstCell, nCells); ok {
				return tqGetResult{path: tqPathKVFused, key: tqkv, skipMetaValue: true}
			}
		}
		// Path 1: K+V combined DequantKV → stock FA (CUDA/ROCm default for
		// non-WHT and WHT+outlier; the dispatcher fuses WHT undo into the
		// kernels when v_rotation != NULL && k_has_outliers).
		if !forceFused && !c.preferFusedAttn &&
			(!c.compressedK.HasRotation() || c.preset.HasOutlierSplit()) {
			k, v := c.compressedK.DequantKV(ctx, layer, enc, vEnc, firstCell, nCells)
			if k != nil && v != nil {
				return tqGetResult{path: tqPathKVDequant, key: k, value: v}
			}
		}
		// Path 2b: WHT K+V no-outlier (test-only configuration).
		// DequantK + WHTUndo + DequantV + WHTUndo → stock FA. Specifically
		// handles HasRotation && !HasOutlier, which Path 1 skips.
		if c.compressedK.HasRotation() && !c.preset.HasOutlierSplit() {
			k := c.compressedK.DequantK(ctx, layer, enc, firstCell, nCells)
			k = c.compressedK.WHTUndo(ctx, layer, k)
			v := c.compressedK.DequantV(ctx, layer, vEnc, firstCell, nCells)
			v = c.compressedK.WHTUndo(ctx, layer, v)
			if k != nil && v != nil {
				return tqGetResult{path: tqPathKVDequantWHT, key: k, value: v}
			}
		}
		// Path 5: separate dequant — last-resort fallback. Return tqPathNone
		// when both planes failed so the caller falls back cleanly to meta.Get
		// rather than receiving a partial (key=nil, value=non-nil) result that
		// the inference guard would silently drop while still emitting the
		// graph node for the dropped tensor.
		k := c.compressedK.DequantK(ctx, layer, enc, firstCell, nCells)
		v := c.compressedK.DequantV(ctx, layer, vEnc, firstCell, nCells)
		if k == nil && v == nil {
			return tqGetResult{path: tqPathNone}
		}
		return tqGetResult{path: tqPathSeparateDequant, key: k, value: v}
	}

	// ── K-only presets (preset.ValueBits == 0) ──────────────────────────
	// Path 0: no asymmetric primary (ablation under OLLAMA_TQ_DISABLE_ASYMMETRIC).
	if !c.preset.AsymmetricPrimary {
		if c.fusedFallbackEligible {
			if tqk, ok := c.compressedK.GetAsTQTensor(ctx, layer, enc, firstCell, nCells); ok {
				return tqGetResult{path: tqPathKOnlyFusedNoAsymm, key: tqk}
			}
		}
		if k := c.compressedK.DequantK(ctx, layer, enc, firstCell, nCells); k != nil {
			return tqGetResult{path: tqPathKOnlyDequantNoAsymm, key: k}
		}
	}
	// Path 2c: WHT K-only with fused dequant+WHT support (CUDA, and Metal
	// since 2026-05-11). Single-op DequantK with WHT signs threaded through.
	if c.compressedK.HasRotation() && c.compressedK.SupportsFusedDequantKWHT() {
		if k := c.compressedK.DequantK(ctx, layer, enc, firstCell, nCells); k != nil {
			return tqGetResult{path: tqPathKOnlyDequantWHT, key: k}
		}
	}
	// Path 4: K-only fused inline-decode. Used when 2c isn't available
	// (some ROCm builds, or pre-2026-05-11 Metal).
	if c.fusedFallbackEligible {
		if tqk, ok := c.compressedK.GetAsTQTensor(ctx, layer, enc, firstCell, nCells); ok {
			return tqGetResult{path: tqPathKOnlyInlineDecode, key: tqk}
		}
	}
	// Path 5 (K-only): plain DequantK fallback.
	if k := c.compressedK.DequantK(ctx, layer, enc, firstCell, nCells); k != nil {
		return tqGetResult{path: tqPathSeparateDequant, key: k}
	}
	return tqGetResult{path: tqPathNone}
}

// pathLogger maps each tqGetPath to its (logPathOnce slot, message, level).
// Inference uses this; reserve doesn't log. Slot 3 is intentionally unused
// (legacy from path renumbering) — leaving it that way to avoid churn.
func (c *TurboQuantCache) pathLogger(p tqGetPath) (slot int, msg string, warn bool) {
	switch p {
	case tqPathKOnlyFusedNoAsymm:
		return 0, "turboquant: using K-only fused path", false
	case tqPathKOnlyDequantNoAsymm:
		return 0, "turboquant: using K-only DequantK + f16 V path (fused unavailable)", false
	case tqPathKVDequant:
		if c.compressedK.HasRotation() {
			return 1, "turboquant: WHT K+V — fused DequantKV with WHT undo → stock FA", false
		}
		return 1, "turboquant: using combined DequantKV + stock FA path", false
	case tqPathKVFused:
		if c.preferFusedAttn {
			return 2, "turboquant: K+V fused inline-decode (Metal preferred — avoids f16 intermediate)", false
		}
		return 2, "turboquant: using K+V fused inline-decode path (forced via OLLAMA_TQ_FORCE_FUSED)", true
	case tqPathKVDequantWHT:
		return 5, "turboquant: WHT K+V — DequantK + DequantV + WHT undo → stock FA", false
	case tqPathKOnlyDequantWHT:
		return 6, "turboquant: WHT K-only — fused DequantK+WHT undo → stock FA (f16 V)", false
	case tqPathKOnlyInlineDecode:
		return 4, "turboquant: using K-only inline-decode fused kernel", false
	case tqPathSeparateDequant:
		return 7, "turboquant: falling back to separate K + V dequant path", true
	}
	return -1, "", false
}

func (c *TurboQuantCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	if c.isReserve {
		key, value, mask := c.meta.Get(ctx)
		// SkipK: synthesize K for graph sizing. Both reserve and inference
		// route through routeGet so gallocr pre-reserves exactly the scratch
		// shape inference will produce. Without this lock-step routing the
		// graph-size estimate diverges from runtime allocation; gemma4:31b
		// burned 384 MiB on this exact bug before the routing was unified.
		if key == nil && c.headDim > 0 {
			nCells := 1
			if c.meta.curMask != nil {
				nCells = c.meta.curMask.Dim(0)
			}
			r := c.routeGet(ctx, c.meta.curLayer, 0, nCells)
			if r.key != nil {
				key = r.key
			}
			if r.value != nil {
				value = r.value
			}
			// SkipV: synthesize V placeholder unless the chosen path placed a
			// K+V fused tensor (V carried inline in `key`). Synthesising f16
			// Zeros for the fused path would re-introduce the scratch buffer
			// the fused path is specifically designed to avoid.
			if value == nil && !r.skipMetaValue {
				value = ctx.Input().Zeros(ml.DTypeF16, c.headDim, c.numKVHeads, nCells)
			}
			if key == nil {
				key = ctx.Input().Zeros(ml.DTypeF16, c.headDim, c.numKVHeads, nCells)
			}
		}
		return key, value, mask
	}

	if c.compressedK != nil {
		layer := c.meta.curLayer
		firstCell := c.meta.curCellRange.min
		nCells := c.meta.curMask.Dim(0)

		// Single source of truth for path selection — see routeGet.
		// The reserve graph above calls the same helper; that lock-step is
		// load-bearing for gallocr scratch sizing. Accept any non-nil tensor
		// so a partial result (e.g. K dequant succeeded but V didn't) isn't
		// silently dropped after its graph node was already emitted.
		r := c.routeGet(ctx, layer, firstCell, nCells)
		if r.path != tqPathNone && (r.key != nil || r.value != nil) {
			if slot, msg, warn := c.pathLogger(r.path); slot >= 0 {
				c.logPathOnce[slot].Do(func() {
					if warn {
						slog.Warn(msg)
					} else {
						slog.Info(msg)
					}
				})
			}
			_, metaValue, mask := c.meta.Get(ctx)
			value := r.value
			if value == nil && !r.skipMetaValue {
				value = metaValue
			}
			if r.skipMetaValue {
				return r.key, nil, mask
			}
			return r.key, value, mask
		}
	}

	return c.meta.Get(ctx)
}

// activateGPUEncode initialises the TQ compressed-K manager if the backend
// supports it and re-enables Q rotation (stored K is in rotated space).
func (c *TurboQuantCache) activateGPUEncode() {
	// fallbackToF16 un-skips K/V on the inner Causal so subsequent Put/Get
	// on this cache store and read f16 tensors like an ordinary non-TQ cache.
	// Init() sets SkipK/SkipV unconditionally, so any failure to activate GPU
	// encode must reverse those flags before the current Put() continues into
	// c.meta.Put — otherwise SDPA receives a nil K/V from Get and segfaults.
	// K allocation in Causal.Put is lazy (keyed on presence of c.keys[layer])
	// so flipping the flag on the first Put is sufficient.
	fallbackToF16 := func() {
		c.meta.SkipK = false
		c.meta.SkipV = false
	}

	// The user requested a measurement-grade preset if asymmetric primary
	// quantization is on. If we can't activate it on the GPU we produce
	// exactly f16 output and any PPL measurement is a lie. Log that failure
	// at ERROR with a distinctive marker so smoke tests and humans can both
	// notice instead of silently reading bit-identical-to-f16 numbers and
	// calling them success.
	measurementRequested := c.preset.AsymmetricPrimary
	logActivation := func(gpuActive bool, path string) {
		level := slog.LevelInfo
		if measurementRequested && !gpuActive {
			level = slog.LevelError
		}
		slog.Log(context.TODO(), level, "TQ_ACTIVATION",
			"preset", c.preset.Name,
			"asymmetric_requested", c.preset.AsymmetricPrimary,
			"outlier_count", c.preset.OutlierCount,
			"gpu_active", gpuActive,
			"measurement_requested", measurementRequested,
			"path", path,
		)
	}

	// Architectural compatibility gate: the TQ encode kernel and fused fattn
	// templates handle headDim ∈ {64, 128, 256, 512}. Models outside this set
	// (e.g. glm-4.7 / DeepSeek-MLA at headDim=576) must skip TQ entirely and
	// stay on f16 — the slow DequantK fallback also relies on encoder kernels
	// that assume those dims.
	if c.headDim != 64 && c.headDim != 128 && c.headDim != 256 && c.headDim != 512 {
		slog.Warn("turboquant: headDim not supported, falling back to f16 KV cache",
			"preset", c.preset.Name,
			"headDim", c.headDim,
			"supported", "{64, 128, 256, 512}")
		logActivation(false, "f16-fallback-headdim-unsupported")
		fallbackToF16()
		return
	}

	tqb, ok := c.meta.backend.(ml.TQCompressedKBackend)
	if !ok {
		logActivation(false, "f16-fallback-no-tq-backend")
		fallbackToF16()
		return
	}

	// Pass the preset's outlier config so the manager can enable post-rotation
	// outlier split on the GPU encode path. This is required for correct
	// output on models with learned K bias (e.g. qwen2 family) and matches
	// the TurboQuant paper's validated experimental setup.
	mgr := tqb.NewTQCompressedKManager(
		c.headDim, c.numKVHeads, c.preset.KeyPrimaryBits, c.preset.RotationSeed,
		c.preset.ValueBits, c.preset.OutlierBits, c.preset.OutlierCount,
		c.preset.AsymmetricPrimary,
	)
	if mgr == nil {
		if measurementRequested {
			slog.Error("turboquant: GPU manager unavailable — SILENT F16 FALLBACK ACTIVE. Run on CUDA, or set OLLAMA_TQ_DISABLE_ASYMMETRIC=1 / OLLAMA_TQ_DISABLE_OUTLIERS=1 to drop to a backend-supported configuration.",
				"preset", c.preset.Name,
				"asymmetric", c.preset.AsymmetricPrimary)
		} else {
			slog.Info("turboquant: GPU encode not available, using f16 K fallback")
		}
		logActivation(false, "f16-fallback-mgr-nil")
		fallbackToF16()
		return
	}
	c.compressedK = mgr
	// Apply any K projection biases buffered before the manager was ready.
	if len(c.pendingKBiases) > 0 {
		type kBiasSetter interface {
			SetLayerKBias(layer int, bias ml.Tensor)
		}
		if kbs, ok := mgr.(kBiasSetter); ok {
			for layer, bias := range c.pendingKBiases {
				kbs.SetLayerKBias(layer, bias)
			}
		}
		c.pendingKBiases = nil
	}
	logActivation(true, "gpu-native")

	// Read the backend's K+V routing preference (Metal=true, CUDA/ROCm=false).
	// Wired into the Path 1 vs Path 2 selection in Get() — see the
	// preferFusedAttn field for rationale.
	type fusedAttnPreferrer interface {
		PreferFusedAttention() bool
	}
	if ff, ok := mgr.(fusedAttnPreferrer); ok {
		c.preferFusedAttn = ff.PreferFusedAttention()
		if c.preferFusedAttn {
			slog.Info("turboquant: K+V routing → fused inline-decode (Metal: avoids f16 intermediate)")
		}
	}

	// Large-D override: at headDim >= 512 the DequantKV f16 intermediate is
	// substantial (~16 MiB per layer at ctx=1024, growing with context) and
	// the fused inline-decode path (path 4) avoids materializing it. Route
	// K+V D>=512 through path 4 by promoting preferFusedAttn even on
	// CUDA/ROCm. The fused kernel pays a per-step decode penalty (~3× prefill
	// slowdown on Pascal P40 at D=512), partially recovered by codebook
	// pre-scaling in tq-fattn-vec.cuh.
	//
	// OLLAMA_TQ_FORCE_DEQUANT_KV=1 reverts to path 1 for benchmarking.
	if c.headDim >= 512 && !c.preferFusedAttn {
		if env := os.Getenv("OLLAMA_TQ_FORCE_DEQUANT_KV"); env == "1" {
			slog.Info("turboquant: K+V routing → DequantKV forced via OLLAMA_TQ_FORCE_DEQUANT_KV (D>=512 default would be path 4)")
		} else {
			c.preferFusedAttn = true
			slog.Info("turboquant: K+V routing → fused inline-decode (D>=512: skip f16 K+V materialization)",
				"headDim", c.headDim)
		}
	}

	// The headDim ∈ {64, 128, 256, 512} gate at the top of this function
	// ensures we only get here on supported dims, so the fused-FA fallback
	// paths (Get paths 2 and 4) are always eligible.
	c.fusedFallbackEligible = true

	slog.Info("turboquant: GPU-native encode active",
		"headDim", c.headDim, "numKVHeads", c.numKVHeads,
		"K_bits", c.preset.KeyPrimaryBits, "V_bits", c.preset.ValueBits)
}

func (c *TurboQuantCache) CopyPrefix(srcSeq, dstSeq int, prefixLen int32) {
	c.meta.CopyPrefix(srcSeq, dstSeq, prefixLen)
}

func (c *TurboQuantCache) CanResume(seq int, pos int32) bool {
	return c.meta.CanResume(seq, pos)
}

// Remove returns ErrNotSupported for any partial eviction when GPU-compressed
// K is active: the compressed buffer cannot be RoPE-shifted in-place, and the
// inner Causal.Remove path silently skips its shiftFn loop because SkipK
// leaves c.keys empty — survivors would keep their old RoPE embeddings while
// their positions are decremented, producing wrong attention scores. Only a
// full-sequence eviction (Remove(seq, 0, MaxInt32)) avoids the shift path;
// other callers must fall back to full reprocessing via ErrReprocessInputs.
func (c *TurboQuantCache) Remove(seq int, beginIndex, endIndex int32) error {
	if c.compressedK != nil && !(beginIndex == 0 && endIndex == math.MaxInt32) {
		return ErrNotSupported
	}
	return c.meta.Remove(seq, beginIndex, endIndex)
}

func (c *TurboQuantCache) SetCausal(ctx ml.Context, opts CausalOptions) {
	c.meta.SetCausal(ctx, opts)
}

// PresetFromDType resolves a runtime KV-cache DType to a turboquant.Preset.
// Returned presets pass through ApplyEnvOverrides so OLLAMA_TQ_DISABLE_*
// takes effect in the cache encode/decode path.
func PresetFromDType(dtype ml.DType) (turboquant.Preset, bool) {
	switch dtype {
	case ml.DTypeTQ2:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ2), true
	case ml.DTypeTQ3:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ3), true
	case ml.DTypeTQ3K:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ3K), true
	case ml.DTypeTQ2K:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ2K), true
	case ml.DTypeTQ4:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ4), true
	case ml.DTypeTQ4K:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ4K), true
	default:
		return turboquant.Preset{}, false
	}
}

var _ Cache = (*TurboQuantCache)(nil)

// Note: TurboQuantCache intentionally does NOT implement CheckpointCache.
// CheckpointCache is for recurrent caches that need special per-sequence
// state restoration. The inner *Causal cache supports plain CanResume, which
// is the right semantics for TQ-wrapped caches too — the runner will fall
// through to the CanResume branch when TurboQuantCache is in use (see
// runner/ollamarunner/cache.go:163-170). An earlier implementation had a
// stub PrepareRestore that always returned (0, false), which forced a full
// prompt reprocess on every resume for long-context gemma/llama runs.
