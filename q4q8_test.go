package scriptlingllmlib

import (
	"encoding/binary"
	"math"
	"testing"
)

func makeQ4Row(groups int) []byte {
	raw := make([]byte, groups*18)
	sb := float64ToFloat16(0.04)
	for g := 0; g < groups; g++ {
		off := g * 18
		binary.LittleEndian.PutUint16(raw[off:], sb)
		for j := 0; j < 16; j++ {
			raw[off+2+j] = byte((g*13 + j*5) % 256)
		}
	}
	return raw
}

// TestQ4q8ScalarMatchesFloat guards the portable Q4 int8 path against the
// full-precision float kernel (≤5% on the vector scale; the int8 activation
// quantization is the only difference).
func TestQ4q8ScalarMatchesFloat(t *testing.T) {
	const outF, cols = 48, 576
	groups := cols / 32
	raw := make([]byte, outF*groups*18)
	sb := float64ToFloat16(0.04)
	for r := 0; r < outF; r++ {
		for g := 0; g < groups; g++ {
			off := (r*groups + g) * 18
			binary.LittleEndian.PutUint16(raw[off:], sb)
			for j := 0; j < 16; j++ {
				raw[off+2+j] = byte((r*13 + g*7 + j*5) % 256)
			}
		}
	}
	w := &QuantWeight{QType: "q4", Raw: raw, Groups: groups, Rows: outF, Cols: cols}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%23)-11) * 0.017
	}

	ref := make([]float32, outF)
	for j := 0; j < outF; j++ {
		ref[j] = q4DotRowsF32(raw, j*groups*18, x, 0, groups)
	}
	got := make([]float32, outF)
	q4q8MatmulInto(w, x, 1, cols, got)

	var maxErr, maxRef float64
	for i := range ref {
		maxErr = math.Max(maxErr, math.Abs(float64(got[i]-ref[i])))
		maxRef = math.Max(maxRef, math.Abs(float64(ref[i])))
	}
	if rel := maxErr / maxRef; rel > 0.05 {
		t.Errorf("q4 int8 vs float: %.4f relative error exceeds 5%%", rel)
	}
}

func TestQ4FusedMatchesScalarDecode(t *testing.T) {
	if !q4q8FusedAvail {
		t.Skip("no fused Q4 kernel on this arch")
	}
	for _, groups := range []int{1, 2, 7, 18, 48} {
		raw := makeQ4Row(groups)
		x := make([]float32, groups*32)
		for i := range x {
			x[i] = float32((i%23)-11) * 0.017
		}
		xq := make([]int8, groups*32)
		xs := make([]float32, groups)
		quantizeActivationsQ8(x, groups, xq, xs)
		// Kernel decodes the f16 weight scale itself and needs the 8·Σxq
		// correction; pass activation scales + correction.
		corr := make([]int32, groups)
		for g := 0; g < groups; g++ {
			var s int32
			for k := 0; k < 32; k++ {
				s += int32(xq[g*32+k])
			}
			corr[g] = 8 * s
		}
		fused := q4q8RowFused(&raw[0], &xq[0], &xs[0], &corr[0], groups)
		scalar := q4q8RowDotScalarDecode(raw, 0, xq, xs, groups)
		if math.Abs(float64(fused-scalar)) > 1e-3*(math.Abs(float64(scalar))+1) {
			t.Errorf("groups=%d: fused=%v scalarDecode=%v", groups, fused, scalar)
		}
	}
}
