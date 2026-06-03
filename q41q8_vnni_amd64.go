//go:build amd64

package scriptlingllmlib

import "time"

//go:noescape
func q41q8RowDotVNNI(wPtr *byte, xqPtr *int8, xScalePtr *float32, sumXqPtr *int32, groups int) float32

// This init runs after q41q8_row_amd64.go's (filename order), so it can replace
// the AVX2 kernel with the AVX-VNNI one where that is available and faster. Like
// the q8q8 VNNI kernel, this is developed on hardware without VNNI, so a runtime
// self-check against the scalar reference is the correctness guarantee: a wrong
// kernel is disabled rather than allowed to corrupt output.
func init() {
	if cpuHasAVXVNNI() && q41q8VNNISelfCheckOK() && q41q8VNNIFasterThanAVX2() {
		q41q8RowFused = q41q8RowDotVNNI
		q41q8FusedAvail = true
	}
}

// q41q8RowScalarRef computes the exact q41q8 row dot in scalar Go:
// result = Σ_g xScale_g · ( d_g·Σ(nibble·xq) + m_g·Σxq_g ). Used to validate the
// assembly kernels. Verified against q41q8RowDotAVX2 in tests on AVX2 hardware.
func q41q8RowScalarRef(raw []byte, xq []int8, xs []float32, sumXq []int32, groups int) float32 {
	var acc float32
	for g := 0; g < groups; g++ {
		off := g * 20
		d := f16LUT[uint16(raw[off])|uint16(raw[off+1])<<8]
		m := f16LUT[uint16(raw[off+2])|uint16(raw[off+3])<<8]
		var qSum int32
		base := g * 32
		for i := 0; i < 16; i++ {
			b := raw[off+4+i]
			qSum += int32(b&0x0F)*int32(xq[base+i]) + int32(b>>4)*int32(xq[base+16+i])
		}
		acc += xs[g] * (d*float32(qSum) + m*float32(sumXq[g]))
	}
	return acc
}

// q41q8vnniInputs builds a representative Q4_1 weight row + int8 activations for
// a given group count (shared by the self-check and benchmark).
func q41q8vnniInputs(groups int) (raw []byte, xq []int8, xs []float32, sumXq []int32) {
	wflat := make([]float32, groups*32)
	for i := range wflat {
		wflat[i] = float32((i%29)-14) * 0.021
	}
	// Build a Q4_1 weight by quantizing to Q8 then re-expressing... simpler: pack
	// arbitrary valid nibbles with plausible f16 d/m.
	raw = make([]byte, groups*20)
	for g := 0; g < groups; g++ {
		off := g * 20
		// d, m as f16 of small values.
		dm := float64ToFloat16(0.05 + 0.001*float64(g%7))
		mm := float64ToFloat16(-0.03 - 0.001*float64(g%5))
		raw[off] = byte(dm)
		raw[off+1] = byte(dm >> 8)
		raw[off+2] = byte(mm)
		raw[off+3] = byte(mm >> 8)
		for i := 0; i < 16; i++ {
			lo := byte((g*16 + i) % 16)
			hi := byte((g*7 + i*3) % 16)
			raw[off+4+i] = lo | (hi << 4)
		}
	}
	x := make([]float32, groups*32)
	for i := range x {
		x[i] = float32((i%23)-11) * 0.017
	}
	xq = make([]int8, groups*32)
	xs = make([]float32, groups)
	quantizeActivationsQ8(x, groups, xq, xs)
	sumXq = make([]int32, groups)
	for g := 0; g < groups; g++ {
		var s int32
		for k := 0; k < 32; k++ {
			s += int32(xq[g*32+k])
		}
		sumXq[g] = s
	}
	return raw, xq, xs, sumXq
}

func q41q8VNNISelfCheckOK() bool {
	for _, groups := range []int{1, 3, 8, 18} {
		raw, xq, xs, sumXq := q41q8vnniInputs(groups)
		got := q41q8RowDotVNNI(&raw[0], &xq[0], &xs[0], &sumXq[0], groups)
		want := q41q8RowScalarRef(raw, xq, xs, sumXq, groups)
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

func q41q8VNNIFasterThanAVX2() bool {
	const groups = 32
	raw, xq, xs, sumXq := q41q8vnniInputs(groups)
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
	vnni := bench(func() { kernelBenchSink += q41q8RowDotVNNI(&raw[0], &xq[0], &xs[0], &sumXq[0], groups) })
	avx2 := bench(func() { kernelBenchSink += q41q8RowDotAVX2(&raw[0], &xq[0], &xs[0], &sumXq[0], groups) })
	return vnni < avx2
}
