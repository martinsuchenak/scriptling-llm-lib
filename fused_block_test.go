package scriptlingllmlib

import (
	"math"
	"testing"
)

func TestRmsNormFlat(t *testing.T) {
	xData := []float64{0.5, -0.3, 0.8}
	w := []float64{1.0, 1.0, 1.0}
	out := make([]float64, 3)

	rmsNormFlat(xData, w, 1e-5, 1, 3, out)

	ss := (0.25 + 0.09 + 0.64) / 3.0
	inv := 1.0 / math.Sqrt(ss+1e-5)
	for j := 0; j < 3; j++ {
		expected := xData[j] * inv * w[j]
		if math.Abs(out[j]-expected) > 1e-10 {
			t.Errorf("rmsNormFlat[%d] = %f, want %f", j, out[j], expected)
		}
	}
}

func TestRmsNormFlatMultiRow(t *testing.T) {
	xData := []float64{3.0, 4.0, 6.0, 8.0}
	w := []float64{1.0, 1.0}
	out := make([]float64, 4)

	rmsNormFlat(xData, w, 1e-5, 2, 2, out)

	for r := 0; r < 2; r++ {
		off := r * 2
		ss := (xData[off]*xData[off] + xData[off+1]*xData[off+1]) / 2.0
		inv := 1.0 / math.Sqrt(ss+1e-5)
		for j := 0; j < 2; j++ {
			expected := xData[off+j] * inv * w[j]
			if math.Abs(out[off+j]-expected) > 1e-10 {
				t.Errorf("rmsNormFlat row%d[%d] = %f, want %f", r, j, out[off+j], expected)
			}
		}
	}
}

func TestApplyRopeInPlace(t *testing.T) {
	data := []float64{1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5}
	dK := 4
	halfDim := 2
	freqs := make([]float64, halfDim)
	freqs[0] = 1.0 / math.Pow(10000.0, 0.0/float64(dK))
	freqs[1] = 1.0 / math.Pow(10000.0, 2.0/float64(dK))

	applyRopeInPlace(data, 2, dK, 5, freqs, halfDim)

	for r := 0; r < 2; r++ {
		pos := 5 + r
		off := r * dK
		for i := 0; i < halfDim; i++ {
			angle := freqs[i] * float64(pos)
			cosA := math.Cos(angle)
			sinA := math.Sin(angle)
			x0 := data[off+2*i]
			x1 := data[off+2*i+1]

			origX0 := 1.0
			origX1 := 0.0
			if r == 0 && i == 1 {
				origX0 = 0.0
				origX1 = 1.0
			} else if r == 1 && i == 0 {
				origX0 = 0.5
				origX1 = 0.5
			} else if r == 1 && i == 1 {
				origX0 = -0.5
				origX1 = 0.5
			}

			expected0 := origX0*cosA - origX1*sinA
			expected1 := origX0*sinA + origX1*cosA
			if math.Abs(x0-expected0) > 1e-10 {
				t.Errorf("data[%d] = %f, want %f", off+2*i, x0, expected0)
			}
			if math.Abs(x1-expected1) > 1e-10 {
				t.Errorf("data[%d] = %f, want %f", off+2*i+1, x1, expected1)
			}
		}
	}
}

func TestFusedAttentionHead(t *testing.T) {
	qData := []float64{1.0, 0.0}
	kData := []float64{1.0, 0.0, 0.0, 1.0}
	vData := []float64{1.0, 0.0, 0.0, 1.0}

	out := make([]float64, 2)
	fusedAttentionHead(qData, kData, vData, 1, 2, 2, false, 0, out, 0)

	total := out[0] + out[1]
	if total < 0.99 || total > 1.01 {
		t.Errorf("attention output sum = %f, want ~1.0", total)
	}
	if out[0] < 0.5 {
		t.Errorf("attention[0] = %f, expected dominant", out[0])
	}
}

func TestFusedAttentionHeadCausal(t *testing.T) {
	qData := []float64{1.0, 0.0, 0.0, 1.0}
	kData := []float64{1.0, 0.0, 0.0, 1.0}
	vData := []float64{1.0, 0.0, 0.0, 1.0}

	out := make([]float64, 4)
	fusedAttentionHead(qData, kData, vData, 2, 2, 2, true, 0, out, 0)

	if out[0] < 0.99 {
		t.Errorf("causal row0 should attend only to pos0: got %f", out[0])
	}
}

func TestSplitHeadsData(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	heads := splitHeadsData(data, 2, 3, 3)

	if len(heads) != 3 {
		t.Fatalf("splitHeadsData count = %d, want 3", len(heads))
	}

	dK := 1
	for h := 0; h < 3; h++ {
		if len(heads[h]) != 2*dK {
			t.Fatalf("head[%d] len = %d, want %d", h, len(heads[h]), 2*dK)
		}
		for r := 0; r < 2; r++ {
			expected := data[r*3+h*dK]
			if heads[h][r*dK] != expected {
				t.Errorf("head[%d][%d] = %f, want %f", h, r*dK, heads[h][r*dK], expected)
			}
		}
	}
}

func TestMergeHeadsData(t *testing.T) {
	dK := 2
	seqLen := 2
	nHeads := 2
	data := make([]float64, nHeads*seqLen*dK)
	for i := range data {
		data[i] = float64(i)
	}

	result := mergeHeadsData(data, seqLen, nHeads, dK)
	if len(result) != seqLen*nHeads*dK {
		t.Fatalf("mergeHeadsData len = %d, want %d", len(result), seqLen*nHeads*dK)
	}

	for s := 0; s < seqLen; s++ {
		for h := 0; h < nHeads; h++ {
			for d := 0; d < dK; d++ {
				srcIdx := h*seqLen*dK + s*dK + d
				dstIdx := s*nHeads*dK + h*dK + d
				if result[dstIdx] != data[srcIdx] {
					t.Errorf("mergeHeadsData[%d] = %f, want %f (from data[%d])", dstIdx, result[dstIdx], data[srcIdx], srcIdx)
				}
			}
		}
	}
}

func TestRepeatKVData(t *testing.T) {
	heads := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	expanded := repeatKVData(heads, 3)
	if len(expanded) != 6 {
		t.Fatalf("repeatKVData count = %d, want 6", len(expanded))
	}

	for i := 0; i < 3; i++ {
		if expanded[i][0] != 1.0 || expanded[i][1] != 2.0 {
			t.Errorf("expanded[%d] = %v, want [1, 2]", i, expanded[i])
		}
	}
	for i := 3; i < 6; i++ {
		if expanded[i][0] != 3.0 || expanded[i][1] != 4.0 {
			t.Errorf("expanded[%d] = %v, want [3, 4]", i, expanded[i])
		}
	}
}
