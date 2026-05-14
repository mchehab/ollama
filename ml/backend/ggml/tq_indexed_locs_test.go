package ggml

// Round-trip correctness for the indexed-addressing path
// (EncodeKAt → DequantKAt with scattered physical-slot locs).
//
// The contiguous fast path is exercised by every other TQ test; this file
// specifically pins down that:
//   - Encode writes to physical slot locs[i] (not firstCell+i)
//   - Dequant reads from physical slot locs[i] back into dense row i
//   - Both pieces produce values that match the contiguous baseline applied
//     to the same dense token data (the data is independent of slot layout).
//
// The kernels are validated against themselves rather than a CPU reference
// because indexed-mode addressing only changes WHERE data lives in the cache
// buffer, not HOW it is encoded — so the encoded values must equal the
// contiguous-mode encoded values placed at the same physical slots.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ollama/ollama/turboquant"
)

func TestTQEncodeKAtFragmentedRoundTrip(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 128
		nKVHeads     = 4
		batchSize    = 6
		capacity     = 32 // larger than batchSize so we can scatter
		bits         = 3
		outlierBits  = 4
		outlierCount = 32
	)

	rotationSeed := turboquant.PresetTQ3K.RotationSeed

	mgrAny := b.NewTQCompressedKManager(
		headDim, nKVHeads, bits,
		rotationSeed,
		0, // vBits = 0 (K-only)
		outlierBits, outlierCount,
		false, // symmetric for a clean CPU-side comparison
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil (GPU not available)")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, capacity)

	rng := rand.New(rand.NewSource(0xfeed))
	kData := make([]float32, headDim*nKVHeads*batchSize)
	for i := range kData {
		kData[i] = float32(rng.NormFloat64()) * 0.7
	}
	kTensor := ctx.FromFloats(kData, headDim, nKVHeads, batchSize)

	// A scattered loc set: tokens i=0..5 → slots [5, 17, 1, 23, 8, 30].
	locs := []int32{5, 17, 1, 23, 8, 30}
	locsTensor := ctx.FromInts(locs, batchSize)

	encAt := mgr.EncodeKAt(ctx, 0, kTensor, locsTensor)
	if encAt == nil {
		t.Fatalf("EncodeKAt returned nil")
	}
	// nCells in the dequant view: read back at exactly the same slots.
	deqAt := mgr.DequantKAt(ctx, 0, encAt, locsTensor)
	if deqAt == nil {
		t.Fatalf("DequantKAt returned nil")
	}
	ctx.Forward(encAt, deqAt).Compute(encAt, deqAt)

	gpuOut := deqAt.Floats()

	// Build a CPU reference: encode each token individually with the CPU
	// reference encoder, dequant it, and require the GPU dequant at the
	// same dense row to match within scalar-quantization slop.
	cpuPreset := turboquant.Preset{
		RotationSeed:   rotationSeed,
		KeyPrimaryBits: bits,
		OutlierBits:    outlierBits,
		OutlierCount:   outlierCount,
	}
	maxDiff := float32(0)
	for tok := range batchSize {
		for h := range nKVHeads {
			perHead := kData[(tok*nKVHeads+h)*headDim : (tok*nKVHeads+h+1)*headDim]
			enc, err := turboquant.EncodeKeyPerHeadOutlier(perHead, cpuPreset)
			if err != nil {
				t.Fatalf("CPU encode tok=%d h=%d: %v", tok, h, err)
			}
			cpuRow := turboquant.DequantKeyPerHeadOutlier(enc, cpuPreset, headDim)
			gpuBase := (tok*nKVHeads + h) * headDim
			for d := range headDim {
				diff := float32(math.Abs(float64(gpuOut[gpuBase+d] - cpuRow[d])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
		}
	}
	// Scalar quantization slop is small but non-zero. f16 round-trip alone
	// contributes ~1e-3; outlier-split adds a bit more. 0.05 is conservative.
	if maxDiff > 0.05 {
		t.Fatalf("max abs diff between GPU indexed-mode dequant and CPU reference = %f (want < 0.05)", maxDiff)
	}
	t.Logf("scattered locs %v: max abs diff = %f", locs, maxDiff)
}

func TestTQEncodeKAtMatchesContiguousAtSameSlots(t *testing.T) {
	// Pins down the equivalence: encoding tokens [0..N-1] with
	// firstCell=F should leave the cache buffer in the same state as
	// encoding tokens [0..N-1] with locs = [F, F+1, ..., F+N-1].
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 128
		nKVHeads     = 2
		batchSize    = 4
		capacity     = 16
		bits         = 3
		outlierBits  = 4
		outlierCount = 32
		firstCell    = 3
	)

	rotationSeed := turboquant.PresetTQ3K.RotationSeed
	mgrAny := b.NewTQCompressedKManager(headDim, nKVHeads, bits, rotationSeed, 0, outlierBits, outlierCount, false)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, capacity)

	rng := rand.New(rand.NewSource(0xcafe))
	kData := make([]float32, headDim*nKVHeads*batchSize)
	for i := range kData {
		kData[i] = float32(rng.NormFloat64()) * 0.5
	}
	kTensor := ctx.FromFloats(kData, headDim, nKVHeads, batchSize)

	// Path 1: contiguous EncodeK starting at firstCell, then DequantK from the
	// same range.
	enc1 := mgr.EncodeK(ctx, 0, kTensor, firstCell)
	if enc1 == nil {
		t.Fatalf("EncodeK returned nil")
	}
	deq1 := mgr.DequantK(ctx, 0, enc1, firstCell, batchSize)
	if deq1 == nil {
		t.Fatalf("DequantK returned nil")
	}
	ctx.Forward(enc1, deq1).Compute(enc1, deq1)
	contiguous := append([]float32(nil), deq1.Floats()...)

	// Reset the cache state for a clean comparison.
	mgr.EnsureLayer(0, capacity)

	// Path 2: indexed EncodeKAt with locs = [F, F+1, F+2, F+3], then
	// DequantKAt with the same locs.
	locs := []int32{firstCell, firstCell + 1, firstCell + 2, firstCell + 3}
	locsTensor := ctx.FromInts(locs, batchSize)
	enc2 := mgr.EncodeKAt(ctx, 0, kTensor, locsTensor)
	if enc2 == nil {
		t.Fatalf("EncodeKAt returned nil")
	}
	deq2 := mgr.DequantKAt(ctx, 0, enc2, locsTensor)
	if deq2 == nil {
		t.Fatalf("DequantKAt returned nil")
	}
	ctx.Forward(enc2, deq2).Compute(enc2, deq2)
	indexed := deq2.Floats()

	if len(contiguous) != len(indexed) {
		t.Fatalf("dequant output length: contiguous=%d indexed=%d", len(contiguous), len(indexed))
	}
	maxDiff := float32(0)
	for i := range contiguous {
		diff := float32(math.Abs(float64(contiguous[i] - indexed[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	// Both paths land in the same physical slots and apply the same encode
	// kernel — outputs should be bit-identical.
	if maxDiff != 0 {
		t.Fatalf("contiguous (firstCell=%d) and indexed (locs=%v) diverge: maxDiff=%f", firstCell, locs, maxDiff)
	}
	t.Logf("contiguous-equivalent locs %v: maxDiff=%f", locs, maxDiff)
}
