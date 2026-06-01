//go:build amd64

package scriptlingllmlib

import "time"

func cpuHasAVXVNNI() bool

//go:noescape
func q8q8RowDotVNNI(wPtr *byte, xqPtr *int8, scalePtr *float32, corrPtr *int32, groups int) float32

func init() {
	// Gate on CPU support, then self-check the result against the scalar
	// reference (this kernel is developed on hardware without VNNI, so the
	// runtime check is the correctness guarantee — a wrong kernel is disabled
	// rather than allowed to corrupt output).
	if cpuHasAVXVNNI() && vnniSelfCheckOK() && vnniFasterThanFused() {
		q8q8VNNIDot = q8q8RowDotVNNI
		q8q8VNNIAvail = true
	}
}

// vnniFasterThanFused benchmarks the VNNI kernel against the fused AVX2 kernel
// (q8q8RowDotAVX2) so VNNI is only used where it actually wins.
func vnniFasterThanFused() bool {
	const groups = 32
	wflat := make([]float32, groups*32)
	for i := range wflat {
		wflat[i] = float32((i%29)-14) * 0.021
	}
	w := quantizeQ8RowsF32(wflat, 1, groups*32)
	x := make([]float32, groups*32)
	for i := range x {
		x[i] = float32((i%23)-11) * 0.017
	}
	xq := make([]int8, groups*32)
	xs := make([]float32, groups)
	quantizeActivationsQ8(x, groups, xq, xs)
	cmb := make([]float32, groups)
	corr := make([]int32, groups)
	for g := 0; g < groups; g++ {
		cmb[g] = f16LUT[uint16(w.Raw[g*34])|uint16(w.Raw[g*34+1])<<8] * xs[g]
		var s int32
		for k := 0; k < 32; k++ {
			s += int32(xq[g*32+k])
		}
		corr[g] = 128 * s
	}

	bench := func(fn func()) time.Duration {
		for i := 0; i < 200; i++ {
			fn()
		}
		best := time.Duration(1<<62 - 1)
		for t := 0; t < 7; t++ {
			start := time.Now()
			for i := 0; i < 2000; i++ {
				fn()
			}
			if d := time.Since(start); d < best {
				best = d
			}
		}
		return best
	}

	vnni := bench(func() { kernelBenchSink += q8q8RowDotVNNI(&w.Raw[0], &xq[0], &cmb[0], &corr[0], groups) })
	fused := bench(func() { kernelBenchSink += q8q8RowDotAVX2(&w.Raw[0], &xq[0], &cmb[0], groups) })
	return vnni < fused
}

// vnniSelfCheckOK runs the VNNI kernel against the scalar int8 reference on a
// few group counts and returns true only if they agree. Guards against any
// error in the (untestable-without-VNNI-hardware) assembly.
func vnniSelfCheckOK() bool {
	for _, groups := range []int{1, 3, 8, 18} {
		wflat := make([]float32, groups*32)
		for i := range wflat {
			wflat[i] = float32((i%29)-14) * 0.021
		}
		w := quantizeQ8RowsF32(wflat, 1, groups*32)
		x := make([]float32, groups*32)
		for i := range x {
			x[i] = float32((i%23)-11) * 0.017
		}
		xq := make([]int8, groups*32)
		xs := make([]float32, groups)
		quantizeActivationsQ8(x, groups, xq, xs)

		cmb := make([]float32, groups)
		corr := make([]int32, groups)
		for g := 0; g < groups; g++ {
			cmb[g] = f16LUT[uint16(w.Raw[g*34])|uint16(w.Raw[g*34+1])<<8] * xs[g]
			var s int32
			for k := 0; k < 32; k++ {
				s += int32(xq[g*32+k])
			}
			corr[g] = 128 * s
		}

		got := q8q8RowDotVNNI(&w.Raw[0], &xq[0], &cmb[0], &corr[0], groups)
		want := q8q8RowDot(w.Raw, 0, xq, xs, groups)
		d := got - want
		if d < 0 {
			d = -d
		}
		ref := want
		if ref < 0 {
			ref = -ref
		}
		if d > 1e-3*(ref+1) {
			return false
		}
	}
	return true
}
