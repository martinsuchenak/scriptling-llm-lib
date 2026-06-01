package scriptlingllmlib

import (
	"os"
	"sync"
	"time"
	"unsafe"
)

// Scratch pools for quantized activations, to keep the per-matmul quantization
// allocation-free on the hot path.
var int8Pool = sync.Pool{New: func() interface{} { b := make([]int8, 0, 8192); return &b }}
var scalePool = sync.Pool{New: func() interface{} { b := make([]float32, 0, 512); return &b }}

// Q8×Q8 matmul path.
//
// The default float kernel multiplies int8 weights by float32 activations,
// which on some virtualized hosts is forced down a heavily-penalized float SIMD
// path (F16C + 256-bit float conversions). Quantizing the activations to int8
// lets the hot loop be pure-integer SIMD (see int8DotAVX2), which is ~3.6x
// faster there. The tradeoff is a ~1% activation quantization error, which is
// negligible for inference.
//
// useInt8Q8 is decided at init by benchmarking the int8 row dot against the
// float kernel on this machine, so the path is only taken where it actually
// wins (e.g. it stays on the float kernel on Apple NEON, where that path is
// already fast).

// int8Dot returns sum(a[i]*b[i]) over n int8 values (n a multiple of 32).
// Defaults to scalar; replaced with an AVX2 implementation on capable amd64.
var int8Dot = int8DotScalar

// q8q8RowFused, when available, computes a whole weight row × int8 activations
// in one call (int MACs + per-group scalar-float scaling), avoiding the
// per-group call and scale overhead of the generic q8q8RowDot path. The scales
// it takes are pre-combined (weight scale × activation scale) per group.
var q8q8RowFused func(wPtr *byte, xqPtr *int8, scalePtr *float32, groups int) float32
var q8q8FusedAvail bool

// q8q8VNNIAvail enables the AVX-VNNI kernel (q8q8VNNIDot), which needs the
// per-group correction array (128·Σxq). Set in vnni_amd64.go after a self-check.
var q8q8VNNIAvail bool
var q8q8VNNIDot func(wPtr *byte, xqPtr *int8, scalePtr *float32, corrPtr *int32, groups int) float32

var int32Pool = sync.Pool{New: func() interface{} { b := make([]int32, 0, 512); return &b }}

// useInt8Q8 enables the quantized-activation matmul path.
var useInt8Q8 bool

// f16LUT maps every float16 bit pattern to its float32 value, so decoding a
// weight group's scale is a single load instead of branchy bit-twiddling.
var f16LUT [65536]float32

// kernelBenchSink keeps the init-time kernel micro-benchmarks (here and in
// cpu_amd64.go) from being optimized away as dead code. Declared here so it is
// available on every architecture.
var kernelBenchSink float32

func init() {
	for i := 0; i < 65536; i++ {
		f16LUT[i] = f16ToF32(uint16(i))
	}
	// SLLM_Q8_KERNEL overrides the auto-selection: "int8" forces the quantized-
	// activation path, "float" forces the full-precision kernel (use this if the
	// ~1% activation quantization error matters). Anything else (incl. unset)
	// auto-selects by benchmarking.
	switch os.Getenv("SLLM_Q8_KERNEL") {
	case "int8":
		useInt8Q8 = true
		useInt8Q4 = true
	case "float":
		useInt8Q8 = false
		useInt8Q4 = false
	default:
		useInt8Q8 = shouldUseInt8Q8()
		useInt8Q4 = shouldUseInt8Q4()
	}
}

func int8DotScalar(aPtr, bPtr *int8, n int) int32 {
	a := unsafe.Slice(aPtr, n)
	b := unsafe.Slice(bPtr, n)
	var s int32
	for i := 0; i < n; i++ {
		s += int32(a[i]) * int32(b[i])
	}
	return s
}

// quantizeActivationsQ8 quantizes a float32 vector into per-group int8 values
// and per-group float32 scales (groups of 32). qDst must hold groups*32 int8;
// sDst must hold groups float32.
func quantizeActivationsQ8(x []float32, groups int, qDst []int8, sDst []float32) {
	for g := 0; g < groups; g++ {
		base := g * 32
		var maxAbs float32
		for i := 0; i < 32; i++ {
			v := x[base+i]
			if v < 0 {
				v = -v
			}
			if v > maxAbs {
				maxAbs = v
			}
		}
		var scale, inv float32
		if maxAbs > 0 {
			scale = maxAbs / 127
			inv = 1 / scale
		}
		sDst[g] = scale
		for i := 0; i < 32; i++ {
			q := int32(x[base+i] * inv)
			if q > 127 {
				q = 127
			} else if q < -128 {
				q = -128
			}
			qDst[base+i] = int8(q)
		}
	}
}

// q8q8RowDot computes one Q8_0 weight row (34-byte groups: f16 scale + 32 int8)
// dotted with a pre-quantized int8 activation vector. The integer MACs run in
// SIMD; the per-group scales are applied in scalar float (cheap, and avoids the
// penalized float SIMD path).
func q8q8RowDot(wRaw []byte, wOff int, xq []int8, xScales []float32, groups int) float32 {
	var total float32
	for g := 0; g < groups; g++ {
		ro := wOff + g*34
		ws := f16LUT[uint16(wRaw[ro])|uint16(wRaw[ro+1])<<8]
		wp := (*int8)(unsafe.Pointer(&wRaw[ro+2]))
		isum := int8Dot(wp, &xq[g*32], 32)
		total += ws * xScales[g] * float32(isum)
	}
	return total
}

// q8q8MatmulInto computes (xRows×xCols) · (outFeatures×xCols)^T into dst using
// the int8 activation path: each activation row is quantized to int8 once, then
// every weight row is dotted against it with integer SIMD. dst must hold
// xRows*outFeatures float32.
func q8q8MatmulInto(w *QuantWeight, xData []float32, xRows, xCols int, dst []float32) {
	groups := w.Groups
	rowBytes := groups * 34
	outFeatures := w.Rows

	xqP := int8Pool.Get().(*[]int8)
	xsP := scalePool.Get().(*[]float32)
	xq := growInt8(*xqP, xRows*xCols)
	xs := growSlice(*xsP, xRows*groups)
	for r := 0; r < xRows; r++ {
		quantizeActivationsQ8(xData[r*xCols:(r+1)*xCols], groups, xq[r*xCols:], xs[r*groups:])
	}

	raw := w.Raw
	if q8q8VNNIAvail {
		// Per-group activation correction (128·Σxq), shared across weight rows.
		corrP := int32Pool.Get().(*[]int32)
		corr := growInt32(*corrP, xRows*groups)
		for xi := 0; xi < xRows; xi++ {
			for g := 0; g < groups; g++ {
				base := xi*xCols + g*32
				var s int32
				for k := 0; k < 32; k++ {
					s += int32(xq[base+k])
				}
				corr[xi*groups+g] = 128 * s
			}
		}
		parallelFor(xRows*outFeatures, func(start, end int) {
			cmbP := scalePool.Get().(*[]float32)
			cmb := growSlice(*cmbP, groups)
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				wOff := j * rowBytes
				xsRow := xs[xi*groups:]
				for g := 0; g < groups; g++ {
					ro := wOff + g*34
					cmb[g] = f16LUT[uint16(raw[ro])|uint16(raw[ro+1])<<8] * xsRow[g]
				}
				dst[idx] = q8q8VNNIDot(&raw[wOff], &xq[xi*xCols], &cmb[0], &corr[xi*groups], groups)
			}
			*cmbP = cmb
			scalePool.Put(cmbP)
		})
		*corrP = corr
		int32Pool.Put(corrP)
	} else if q8q8FusedAvail {
		parallelFor(xRows*outFeatures, func(start, end int) {
			// One reusable combined-scale buffer per worker chunk.
			cmbP := scalePool.Get().(*[]float32)
			cmb := growSlice(*cmbP, groups)
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				wOff := j * rowBytes
				xsRow := xs[xi*groups:]
				for g := 0; g < groups; g++ {
					ro := wOff + g*34
					cmb[g] = f16LUT[uint16(raw[ro])|uint16(raw[ro+1])<<8] * xsRow[g]
				}
				dst[idx] = q8q8RowFused(&raw[wOff], &xq[xi*xCols], &cmb[0], groups)
			}
			*cmbP = cmb
			scalePool.Put(cmbP)
		})
	} else {
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				dst[idx] = q8q8RowDot(raw, j*rowBytes, xq[xi*xCols:], xs[xi*groups:], groups)
			}
		})
	}

	*xqP = xq
	int8Pool.Put(xqP)
	*xsP = xs
	scalePool.Put(xsP)
}

func growInt8(b []int8, n int) []int8 {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]int8, n)
}

func growInt32(b []int32, n int) []int32 {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]int32, n)
}

// shouldUseInt8Q8 benchmarks the int8 row dot against the float kernel on a
// representative row and returns true when int8 is clearly faster. Mirrors the
// other init-time micro-benchmarks; uses min-of-trials to resist noise.
func shouldUseInt8Q8() bool {
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

	int8Time := bench(func() {
		kernelBenchSink += q8q8RowDot(w.Raw, 0, xq, xs, groups)
	})
	floatTime := bench(func() {
		kernelBenchSink += q8DotRowsAsm(&w.Raw[0], &x[0], groups)
	})

	// Require a clear margin so we only switch (and accept the quantization
	// error) where the win is real.
	return int8Time*5 < floatTime*4
}
