package scriptlingllmlib

import (
	"encoding/binary"
	"time"
)

// Q4_0 × Q8 (int8 activations) path.
//
// Q4_0 has no float SIMD kernel, so it was the slowest format. Quantizing
// activations to int8 and decoding the 4-bit weights with a fused SIMD kernel
// (q4q8RowDotAVX2) turns it into a real integer-SIMD matmul. The simple per-row
// fallback (q4q8RowDotScalarDecode) decodes nibbles in Go and is only used where
// the fused kernel is unavailable.
//
// Q4_0 layout: 18-byte groups = 2-byte f16 scale + 16 nibble bytes. Byte j holds
// weight j (low nibble) and weight j+16 (high nibble), each a signed nibble-8.

var useInt8Q4 bool

// q4q8RowFused, when available, decodes + dots a whole Q4_0 row in one call,
// decoding the f16 weight scale in-kernel and taking the per-group activation
// scales directly.
var q4q8RowFused func(wPtr *byte, xqPtr *int8, xScalePtr *float32, groups int) float32
var q4q8FusedAvail bool

// q4q8RowDotScalarDecode is the portable fallback: decode nibbles in Go, then
// use the int8Dot primitive for the integer MACs.
func q4q8RowDotScalarDecode(raw []byte, rOff int, xq []int8, xScales []float32, groups int) float32 {
	var wq [32]int8
	_ = wq[31]
	var total float32
	for g := 0; g < groups; g++ {
		ro := rOff + g*18
		ws := f16LUT[uint16(raw[ro])|uint16(raw[ro+1])<<8]
		nib := raw[ro+2 : ro+18 : ro+18]
		for j := 0; j < 16; j++ {
			b := nib[j]
			wq[j] = int8(b&0x0F) - 8
			wq[j+16] = int8(b>>4) - 8
		}
		total += ws * xScales[g] * float32(int8Dot(&wq[0], &xq[g*32], 32))
	}
	return total
}

// q4q8MatmulInto computes (xRows×xCols) · (outFeatures×xCols)^T into dst for a
// Q4_0 weight, quantizing each activation row to int8 once.
func q4q8MatmulInto(w *QuantWeight, xData []float32, xRows, xCols int, dst []float32) {
	groups := w.Groups
	rowBytes := groups * 18
	outFeatures := w.Rows

	xqP := int8Pool.Get().(*[]int8)
	xsP := scalePool.Get().(*[]float32)
	xq := growInt8(*xqP, xRows*xCols)
	xs := growSlice(*xsP, xRows*groups)
	for r := 0; r < xRows; r++ {
		quantizeActivationsQ8(xData[r*xCols:(r+1)*xCols], groups, xq[r*xCols:], xs[r*groups:])
	}

	raw := w.Raw
	if q4q8FusedAvail {
		// The kernel decodes the f16 weight scale itself, so we pass the
		// activation scales straight through — no per-row scale pass.
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				dst[idx] = q4q8RowFused(&raw[j*rowBytes], &xq[xi*xCols], &xs[xi*groups], groups)
			}
		})
	} else {
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				dst[idx] = q4q8RowDotScalarDecode(raw, j*rowBytes, xq[xi*xCols:], xs[xi*groups:], groups)
			}
		})
	}

	*xqP = xq
	int8Pool.Put(xqP)
	*xsP = xs
	scalePool.Put(xsP)
}

// shouldUseInt8Q4 benchmarks the int8 Q4 path against the scalar float Q4 kernel.
func shouldUseInt8Q4() bool {
	const groups = 32
	raw := make([]byte, groups*18)
	scaleBits := float64ToFloat16(0.05)
	for g := 0; g < groups; g++ {
		off := g * 18
		binary.LittleEndian.PutUint16(raw[off:], scaleBits)
		for j := 0; j < 16; j++ {
			raw[off+2+j] = byte((g*7 + j*3) % 256)
		}
	}
	x := make([]float32, groups*32)
	for i := range x {
		x[i] = float32((i%23)-11) * 0.017
	}
	xq := make([]int8, groups*32)
	xs := make([]float32, groups)
	quantizeActivationsQ8(x, groups, xq, xs)
	cmb := make([]float32, groups)
	for g := 0; g < groups; g++ {
		cmb[g] = f16LUT[uint16(raw[g*18])|uint16(raw[g*18+1])<<8] * xs[g]
	}

	int8Fn := func() {
		if q4q8FusedAvail {
			kernelBenchSink += q4q8RowFused(&raw[0], &xq[0], &cmb[0], groups)
		} else {
			kernelBenchSink += q4q8RowDotScalarDecode(raw, 0, xq, xs, groups)
		}
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

	int8Time := bench(int8Fn)
	floatTime := bench(func() {
		kernelBenchSink += q4DotRowsF32(raw, 0, x, 0, groups)
	})

	return int8Time*5 < floatTime*4
}
