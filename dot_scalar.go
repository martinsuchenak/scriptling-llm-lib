package scriptlingllmlib

import (
	"math"
	"unsafe"
)

// q8DotRowsScalar is a pure-Go implementation of the Q8_0 row dot product with
// the same pointer signature as the assembly kernels (q8DotRowsAsm*). It exists
// so the kernel selector in cpu_*.go can fall back to it when the SIMD kernels
// are slower than scalar — which happens on some virtualized CPUs where the
// AVX2/F16C path is heavily penalized.
//
// Layout: `groups` consecutive 34-byte blocks. Each block is a 2-byte float16
// scale followed by 32 int8 quants. x is `groups*32` consecutive float32 values.
func q8DotRowsScalar(rawPtr *byte, xPtr *float32, groups int) float32 {
	raw := unsafe.Slice(rawPtr, groups*34)
	x := unsafe.Slice(xPtr, groups*32)

	var total float32
	for g := 0; g < groups; g++ {
		rOff := g * 34
		s := f16ToF32(uint16(raw[rOff]) | uint16(raw[rOff+1])<<8)
		q := unsafe.Slice((*uint64)(unsafe.Pointer(&raw[rOff+2])), 4)
		xBase := g * 32

		var sum float32
		for w := 0; w < 4; w++ {
			chunk := q[w]
			i := xBase + w*8
			sum += float32(int8(chunk))*x[i] +
				float32(int8(chunk>>8))*x[i+1] +
				float32(int8(chunk>>16))*x[i+2] +
				float32(int8(chunk>>24))*x[i+3] +
				float32(int8(chunk>>32))*x[i+4] +
				float32(int8(chunk>>40))*x[i+5] +
				float32(int8(chunk>>48))*x[i+6] +
				float32(int8(chunk>>56))*x[i+7]
		}
		total += s * sum
	}
	return total
}

// f16ToF32 converts an IEEE-754 half-precision value to float32.
func f16ToF32(u16 uint16) float32 {
	sign := uint32(u16&0x8000) << 16
	exp := uint32((u16 >> 10) & 0x1f)
	frac := uint32(u16 & 0x3ff)
	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		f32 := math.Float32frombits(sign | (frac << 13))
		return f32 / float32(1<<24)
	}
	if exp == 31 {
		if frac == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}
		return math.Float32frombits(sign | 0x7fc00000)
	}
	return math.Float32frombits(sign | ((exp + 112) << 23) | (frac << 13))
}
