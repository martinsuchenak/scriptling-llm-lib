package scriptlingllmlib

import (
	"math"
	"testing"
)

func TestF16LUTMatchesDecode(t *testing.T) {
	for i := 0; i < 65536; i++ {
		want := f16ToF32(uint16(i))
		got := f16LUT[i]
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("bits %d: want NaN, got %v", i, got)
			}
			continue
		}
		if got != want {
			t.Fatalf("bits %d: lut=%v decode=%v", i, got, want)
		}
	}
}

func TestInt8DotMatchesReference(t *testing.T) {
	for _, n := range []int{32, 64, 32 * 18, 32 * 48} {
		a := make([]int8, n)
		b := make([]int8, n)
		for i := range a {
			a[i] = int8(i%37 - 18)
			b[i] = int8((i*7)%51 - 25)
		}
		var ref int32
		for i := 0; i < n; i++ {
			ref += int32(a[i]) * int32(b[i])
		}
		if got := int8Dot(&a[0], &b[0], n); got != ref {
			t.Errorf("n=%d: int8Dot=%d ref=%d", n, got, ref)
		}
		if got := int8DotScalar(&a[0], &b[0], n); got != ref {
			t.Errorf("n=%d: int8DotScalar=%d ref=%d", n, got, ref)
		}
	}
}

// TestQ8Q8MatmulMatchesFloat checks the int8 activation path stays close to the
// full-precision float matmul — the whole point is that quantizing activations
// costs only ~1% accuracy.
func TestQ8Q8MatmulMatchesFloat(t *testing.T) {
	const outFeatures, cols = 64, 576 // 18 groups
	wflat := make([]float32, outFeatures*cols)
	for i := range wflat {
		wflat[i] = float32((i%29)-14) * 0.021
	}
	w := quantizeQ8RowsF32(wflat, outFeatures, cols)

	for _, xRows := range []int{1, 3} {
		xData := make([]float32, xRows*cols)
		for i := range xData {
			xData[i] = float32((i%23)-11) * 0.017
		}

		// Float reference via the existing kernel.
		ref := make([]float32, xRows*outFeatures)
		for xi := 0; xi < xRows; xi++ {
			for j := 0; j < outFeatures; j++ {
				ref[xi*outFeatures+j] = q8DotRowsF32(w.Raw, j*w.Groups*34, xData, xi*cols, w.Groups)
			}
		}

		got := make([]float32, xRows*outFeatures)
		q8q8MatmulInto(w, xData, xRows, cols, got)

		// Accuracy is measured against the magnitude of the output vector (the
		// meaningful scale), not per-element — a near-zero element from
		// cancellation can have large relative error while being numerically
		// irrelevant.
		var maxAbsErr, maxRef float64
		for i := range ref {
			maxAbsErr = math.Max(maxAbsErr, math.Abs(float64(got[i]-ref[i])))
			maxRef = math.Max(maxRef, math.Abs(float64(ref[i])))
		}
		if rel := maxAbsErr / maxRef; rel > 0.02 {
			t.Errorf("xRows=%d: error %.4f relative to vector scale exceeds 2%%", xRows, rel)
		}
	}
}

func TestQuantizeActivationsRoundTrip(t *testing.T) {
	const groups = 4
	x := make([]float32, groups*32)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.3))
	}
	xq := make([]int8, groups*32)
	xs := make([]float32, groups)
	quantizeActivationsQ8(x, groups, xq, xs)
	for g := 0; g < groups; g++ {
		for i := 0; i < 32; i++ {
			recon := float32(xq[g*32+i]) * xs[g]
			if d := math.Abs(float64(recon - x[g*32+i])); d > float64(xs[g]) {
				t.Errorf("g=%d i=%d: |%v-%v|=%v > scale %v", g, i, recon, x[g*32+i], d, xs[g])
			}
		}
	}
}
