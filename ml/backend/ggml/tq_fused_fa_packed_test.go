package ggml

// Tests for the kernel_tq_fattn_vec_packed_outlier_* family:
// K+V fused inline-decode flash attention (outlier split, various head dims).
// This is the Metal-default path for tq2/tq3/tq4 K+V presets on Apple Silicon.
//
// Coverage:
//   packed_outlier_d64  — llama3.2:3b head_dim=64
//   packed_outlier      — D=128 (qwen2.5, llama3.1)
//   packed_outlier_d256 — D=256 (gemma3)
//   packed_outlier_d512 — D=512 (gemma4 global attention layers)

import (
	"math"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

type tqPackedFAConfig struct {
	headDim, nKVHeads, nCells int
	bits, vBits               int
	outlierBits, outlierCount int
	tol                       float32
	label                     string
}

func tqRunPackedFA(t *testing.T, cfg tqPackedFAConfig) {
	t.Helper()
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	rotationSeed := turboquant.PresetTQ3.RotationSeed ^ uint64(cfg.headDim)

	mgrAny := b.NewTQCompressedKManager(
		cfg.headDim, cfg.nKVHeads, cfg.bits,
		rotationSeed,
		cfg.vBits, // K+V mode
		cfg.outlierBits, cfg.outlierCount,
		false, // symmetric
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil (GPU unavailable or unsupported)")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, cfg.nCells)
	mgr.EnsureVLayer(0, cfg.nCells)

	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d", cfg.headDim, cfg.bits)
	}

	var rngState uint64 = 0xfab1_c400_0000_0000 | uint64(cfg.headDim)

	kData := make([]float32, cfg.headDim*cfg.nKVHeads*cfg.nCells)
	for i := range kData {
		kData[i] = float32(tqGaussian(&rngState))
	}
	for c := range cfg.nCells {
		for h := range cfg.nKVHeads {
			nSpikes := cfg.outlierCount / 4
			for range nSpikes {
				d := int(tqSplitmix64(&rngState) % uint64(cfg.headDim))
				kData[(c*cfg.nKVHeads+h)*cfg.headDim+d] *= 5.0
			}
		}
	}

	layerSeed := rotationSeed ^ uint64(1)
	rotation := turboquant.BuildRotation(cfg.headDim, layerSeed)

	qRaw := make([]float32, cfg.headDim*cfg.nKVHeads)
	for i := range qRaw {
		qRaw[i] = float32(tqGaussian(&rngState)) * 0.5
	}
	qRotated := make([]float32, cfg.headDim*cfg.nKVHeads)
	for h := range cfg.nKVHeads {
		rotQ := turboquant.ApplyRotation(qRaw[h*cfg.headDim:(h+1)*cfg.headDim], rotation)
		copy(qRotated[h*cfg.headDim:], rotQ)
	}

	vF32 := make([]float32, cfg.headDim*cfg.nCells*cfg.nKVHeads)
	for i := range vF32 {
		vF32[i] = float32(tqGaussian(&rngState)) * 0.25
	}

	kTensor := ctx.FromFloats(kData, cfg.headDim, cfg.nKVHeads, cfg.nCells)
	qTensor := ctx.FromFloats(qRotated, cfg.headDim, 1, cfg.nKVHeads)
	// V as f32 for encode; same shape as K (headDim, nCells, nKVHeads in ggml col-major).
	vTensorF32 := ctx.FromFloats(vF32, cfg.headDim, cfg.nCells, cfg.nKVHeads)

	encK := mgr.EncodeK(ctx, 0, kTensor, 0)
	if encK == nil {
		t.Fatal("EncodeK returned nil")
	}
	encV := mgr.EncodeV(ctx, 0, vTensorF32, 0)
	if encV == nil {
		t.Fatal("EncodeV returned nil — vBits may be 0 or EnsureVLayer not called")
	}

	// GetAsTQTensorKV wraps K+V packed tensors; tqFlashAttention dispatches to
	// packed_outlier_* kernel (v_packed=true path in ggml-metal-ops.cpp).
	tqkvRaw, ok := mgr.GetAsTQTensorKV(ctx, 0, encK, encV, 0, cfg.nCells)
	if !ok || tqkvRaw == nil {
		t.Skip("GetAsTQTensorKV returned (nil, false) — K+V fused FA not supported on this device")
	}
	tqkv := tqkvRaw.(*tqTensor)
	if tqkv.vPacked == nil {
		t.Fatal("tqTensor.vPacked is nil after GetAsTQTensorKV")
	}

	// The K+V packed dispatch asserts mask != nullptr to derive nCells from
	// mask->ne[0]. Provide an all-zero (no masking) f16 mask [nCells, 1].
	maskF32 := make([]float32, cfg.nCells)
	maskF16Bytes := tqF32SliceToF16Bytes(maskF32)
	maskTensor := ctx.FromBytes(ml.DTypeF16, maskF16Bytes, cfg.nCells, 1)

	attnScale := 1.0 / math.Sqrt(float64(cfg.headDim))
	// Pass tqkv.vPacked as the value tensor — matches the production K+V fused path.
	attnOut := b.tqFlashAttention(ctx, qTensor.(*Tensor), tqkv, tqkv.vPacked, maskTensor, attnScale, 0)
	if attnOut == nil {
		t.Fatal("tqFlashAttention returned nil")
	}

	ctx.Forward(encK, encV, attnOut).Compute(encK, encV, attnOut)
	_ = maskTensor // used as input to attnOut graph node

	gpuOut := attnOut.Floats()
	for i, v := range gpuOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s: gpuOut[%d]=%v (NaN/Inf)", cfg.label, i, v)
		}
	}

	// CPU reference: use quantized K and raw V. V-quantization noise is absorbed
	// into the tolerance (typically adds ~0.1–0.2 to max diff for 3-bit V).
	cpuPreset := turboquant.Preset{
		RotationSeed:   layerSeed,
		KeyPrimaryBits: cfg.bits,
		OutlierBits:    cfg.outlierBits,
		OutlierCount:   cfg.outlierCount,
	}
	kDecoded := tqCPUKRef(t, kData, cfg.nCells, cfg.nKVHeads, cfg.headDim, cpuPreset)

	maxDiff, nMismatches := tqCheckAttn(t, gpuOut, qRotated, kDecoded, vF32,
		1, cfg.nKVHeads, cfg.nKVHeads, cfg.nCells, cfg.headDim, attnScale, cfg.tol,
		cfg.label)
	t.Logf("%s: max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		cfg.label, maxDiff, nMismatches, cfg.headDim*cfg.nKVHeads, cfg.tol)
	if nMismatches > 0 {
		t.Fatalf("%s: %d mismatches beyond tol=%.2f", cfg.label, nMismatches, cfg.tol)
	}
}

// TestTQFusedFlashAttentionPackedOutlierD64 — kernel_tq_fattn_vec_packed_outlier_d64
// (llama3.2:3b head_dim=64, K+V preset on Metal).
func TestTQFusedFlashAttentionPackedOutlierD64(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 64, nKVHeads: 4, nCells: 8,
		bits: 3, vBits: 3, outlierBits: 4, outlierCount: 16,
		tol: 0.5, label: "packed_outlier_d64",
	})
}

// TestTQFusedFlashAttentionPackedOutlierD128 — kernel_tq_fattn_vec_packed_outlier
// (qwen2.5/llama3.1 head_dim=128, K+V preset on Metal).
func TestTQFusedFlashAttentionPackedOutlierD128(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 128, nKVHeads: 8, nCells: 8,
		bits: 3, vBits: 3, outlierBits: 4, outlierCount: 32,
		tol: 0.65, label: "packed_outlier_d128",
	})
}

// TestTQFusedFlashAttentionPackedOutlierD256 — kernel_tq_fattn_vec_packed_outlier_d256
// (gemma3 head_dim=256, K+V preset on Metal).
func TestTQFusedFlashAttentionPackedOutlierD256(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 256, nKVHeads: 4, nCells: 8,
		bits: 3, vBits: 3, outlierBits: 4, outlierCount: 32,
		tol: 0.65, label: "packed_outlier_d256",
	})
}

// TestTQFusedFlashAttentionPackedOutlierD512 — kernel_tq_fattn_vec_packed_outlier_d512
// (gemma4 global attention head_dim=512, K+V preset on Metal).
func TestTQFusedFlashAttentionPackedOutlierD512(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 512, nKVHeads: 4, nCells: 8,
		bits: 3, vBits: 3, outlierBits: 4, outlierCount: 32,
		tol: 0.65, label: "packed_outlier_d512",
	})
}
