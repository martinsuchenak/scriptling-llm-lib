package scriptlingllmlib

import (
	"encoding/binary"
	"time"
)

// Q4_1 × Q8 (int8 activations) path.
//
// Q4_1 (group = 2-byte d + 2-byte m + 16 nibble bytes; weight = d*nibble + m)
// had only the scalar float kernel. The fused AVX2 kernel below does the int8
// MACs with VPMADDUBSW and computes  xScale·(d·Σ(nibble·xq) + m·Σxq)  per group.
// Only used where it beats the scalar kernel (and only on AVX2); the existing
// float path remains the fallback elsewhere.

var useInt8Q41 bool
var q41q8RowFused func(wPtr *byte, xqPtr *int8, xScalePtr *float32, sumXqPtr *int32, groups int) float32
var q41q8FusedAvail bool

func q41q8MatmulInto(w *QuantWeight, xData []float32, xRows, xCols int, dst []float32) {
	groups := w.Groups
	rowBytes := groups * 20
	outFeatures := w.Rows

	xqP := int8Pool.Get().(*[]int8)
	xsP := scalePool.Get().(*[]float32)
	sumP := int32Pool.Get().(*[]int32)
	xq := growInt8(*xqP, xRows*xCols)
	xs := growSlice(*xsP, xRows*groups)
	sumXq := growInt32(*sumP, xRows*groups)
	for r := 0; r < xRows; r++ {
		quantizeActivationsQ8(xData[r*xCols:(r+1)*xCols], groups, xq[r*xCols:], xs[r*groups:])
	}
	for xi := 0; xi < xRows; xi++ {
		for g := 0; g < groups; g++ {
			base := xi*xCols + g*32
			var s int32
			for k := 0; k < 32; k++ {
				s += int32(xq[base+k])
			}
			sumXq[xi*groups+g] = s
		}
	}

	raw := w.Raw
	parallelFor(xRows*outFeatures, func(start, end int) {
		for idx := start; idx < end; idx++ {
			xi := idx / outFeatures
			j := idx % outFeatures
			dst[idx] = q41q8RowFused(&raw[j*rowBytes], &xq[xi*xCols], &xs[xi*groups], &sumXq[xi*groups], groups)
		}
	})

	*xqP = xq
	int8Pool.Put(xqP)
	*xsP = xs
	scalePool.Put(xsP)
	*sumP = sumXq
	int32Pool.Put(sumP)
}

// shouldUseInt8Q41 benchmarks the fused Q4_1 kernel against the scalar float one.
func shouldUseInt8Q41() bool {
	if !q41q8FusedAvail {
		return false
	}
	const groups = 32
	raw := make([]byte, groups*20)
	d := float64ToFloat16(0.05)
	m := float64ToFloat16(-0.3)
	for g := 0; g < groups; g++ {
		off := g * 20
		binary.LittleEndian.PutUint16(raw[off:], d)
		binary.LittleEndian.PutUint16(raw[off+2:], m)
		for j := 0; j < 16; j++ {
			raw[off+4+j] = byte((g*7 + j*3) % 256)
		}
	}
	x := make([]float32, groups*32)
	for i := range x {
		x[i] = float32((i%23)-11) * 0.017
	}
	xq := make([]int8, groups*32)
	xs := make([]float32, groups)
	quantizeActivationsQ8(x, groups, xq, xs)
	sumXq := make([]int32, groups)
	for g := 0; g < groups; g++ {
		var s int32
		for k := 0; k < 32; k++ {
			s += int32(xq[g*32+k])
		}
		sumXq[g] = s
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
			if dt := time.Since(start); dt < best {
				best = dt
			}
		}
		return best
	}

	int8Time := bench(func() { kernelBenchSink += q41q8RowFused(&raw[0], &xq[0], &xs[0], &sumXq[0], groups) })
	floatTime := bench(func() { kernelBenchSink += q41DotRowsF32(raw, 0, x, 0, groups) })
	return int8Time*5 < floatTime*4
}
