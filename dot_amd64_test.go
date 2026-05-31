//go:build amd64

package scriptlingllmlib

import "testing"

// TestSSEKernelMatchesScalar guards the SSE fallback kernel (used on x86 CPUs
// without AVX, e.g. Westmere Xeons). It must (a) assemble with only legacy SSE
// encodings — VEX/AVX instructions here fault with SIGILL on those CPUs, which
// is exactly the regression this catches at the source level — and (b) produce
// the same result as the scalar reference. This test runs on any amd64 host
// because legacy SSE executes everywhere; the no-VEX property is enforced by
// the instruction choices in dot_amd64.s.
func TestSSEKernelMatchesScalar(t *testing.T) {
	for _, groups := range []int{1, 2, 7, 18, 48, 64} {
		wflat := make([]float32, groups*32)
		for i := range wflat {
			wflat[i] = float32((i%29)-14) * 0.021
		}
		w := quantizeQ8RowsF32(wflat, 1, groups*32)
		x := make([]float32, groups*32)
		for i := range x {
			x[i] = float32((i%23)-11) * 0.017
		}
		sse := q8DotRowsAsmSSE(&w.Raw[0], &x[0], groups)
		ref := q8DotRowsScalar(&w.Raw[0], &x[0], groups)
		if absf(sse-ref) > 1e-3*(absf(ref)+1) {
			t.Errorf("groups=%d: sse=%v scalar=%v", groups, sse, ref)
		}
	}
}
