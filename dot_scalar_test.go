package scriptlingllmlib

import (
	"math"
	"testing"
)

// TestQ8DotRowsScalarMatchesSIMD guards the kernel selector: the scalar
// fallback (which the model uses on machines where SIMD is penalized) must
// produce the same result as the assembly kernel. If these diverge, decode
// output is silently wrong on whichever path is selected.
func TestQ8DotRowsScalarMatchesSIMD(t *testing.T) {
	for _, groups := range []int{1, 2, 7, 18, 64, 128} {
		raw := make([]byte, groups*34)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		x := make([]float32, groups*32)
		for i := range x {
			x[i] = float32((i%23)-11) * 0.017
		}

		got := q8DotRowsScalar(&raw[0], &x[0], groups)
		want := q8DotRowsAsm(&raw[0], &x[0], groups)

		// Both reduce the same products; allow a small tolerance for the
		// different summation orders (scalar groups vs SIMD lanes).
		tol := float32(1e-3) * (absf(want) + 1)
		if absf(got-want) > tol {
			t.Errorf("groups=%d: scalar=%v simd=%v (diff %v > tol %v)",
				groups, got, want, absf(got-want), tol)
		}
	}
}

func TestQ8DotRowsScalarMatchesGroupwise(t *testing.T) {
	// q8DotRowsScalar must agree with the per-group reference used elsewhere.
	const groups = 18
	raw := make([]byte, groups*34)
	for i := range raw {
		raw[i] = byte(i*13 + 5)
	}
	x := make([]float32, groups*32)
	for i := range x {
		x[i] = float32(i%19) * 0.03
	}

	var ref float32
	for g := 0; g < groups; g++ {
		ref += q8DotGroupXF32(raw, g*34, x, g*32)
	}
	got := q8DotRowsScalar(&raw[0], &x[0], groups)
	if absf(got-ref) > 1e-3*(absf(ref)+1) {
		t.Errorf("scalar=%v ref=%v", got, ref)
	}
}

func absf(f float32) float32 { return float32(math.Abs(float64(f))) }
