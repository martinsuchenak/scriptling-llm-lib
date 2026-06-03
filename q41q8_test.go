package scriptlingllmlib

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestQ41q8MatchesFloat checks the fused Q4_1 int8 path stays close to the
// scalar float Q4_1 kernel (only the ~1% activation quantization differs).
func TestQ41q8MatchesFloat(t *testing.T) {
	if !q41q8FusedAvail {
		t.Skip("no fused Q4_1 kernel on this arch")
	}
	const outF, cols = 48, 576
	groups := cols / 32
	raw := make([]byte, outF*groups*20)
	d := float64ToFloat16(0.05)
	m := float64ToFloat16(-0.3)
	for r := 0; r < outF; r++ {
		for g := 0; g < groups; g++ {
			off := (r*groups + g) * 20
			binary.LittleEndian.PutUint16(raw[off:], d)
			binary.LittleEndian.PutUint16(raw[off+2:], m)
			for j := 0; j < 16; j++ {
				raw[off+4+j] = byte((r*13 + g*7 + j*5) % 256)
			}
		}
	}
	w := &QuantWeight{QType: "q4_1", Raw: raw, Groups: groups, Rows: outF, Cols: cols}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%23)-11) * 0.017
	}

	ref := make([]float32, outF)
	for j := 0; j < outF; j++ {
		ref[j] = q41DotRowsF32(raw, j*groups*20, x, 0, groups)
	}
	got := make([]float32, outF)
	q41q8MatmulInto(w, x, 1, cols, got)

	var maxErr, maxRef float64
	for i := range ref {
		maxErr = math.Max(maxErr, math.Abs(float64(got[i]-ref[i])))
		maxRef = math.Max(maxRef, math.Abs(float64(ref[i])))
	}
	if rel := maxErr / maxRef; rel > 0.05 {
		t.Errorf("q4_1 int8 vs float: %.4f relative error exceeds 5%%", rel)
	}
}
