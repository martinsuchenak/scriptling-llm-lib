//go:build amd64

package scriptlingllmlib

import (
	"math"
	"testing"
)

// TestQ41ScalarRefMatchesAVX2 guarantees q41q8RowScalarRef (the reference the
// AVX-VNNI kernel's runtime self-check compares against) is itself correct, by
// checking it against the validated AVX2 kernel. The VNNI kernel is developed
// without VNNI hardware, so this is what makes the on-device self-check sound.
func TestQ41ScalarRefMatchesAVX2(t *testing.T) {
	if !q41q8FusedAvail {
		t.Skip("AVX2 kernel unavailable")
	}
	for _, groups := range []int{1, 2, 3, 8, 18, 32, 48, 64} {
		raw, xq, xs, sumXq := q41q8vnniInputs(groups)
		ref := q41q8RowScalarRef(raw, xq, xs, sumXq, groups)
		asm := q41q8RowDotAVX2(&raw[0], &xq[0], &xs[0], &sumXq[0], groups)
		if d := math.Abs(float64(ref - asm)); d > 1e-3*(math.Abs(float64(ref))+1) {
			t.Errorf("groups=%d: scalarRef=%.6f AVX2=%.6f diff=%.2e", groups, ref, asm, d)
		}
	}
}
