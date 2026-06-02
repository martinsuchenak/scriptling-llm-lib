package scriptlingllmlib

import "encoding/binary"

// Native packed k-quant matmul kernels. Instead of dequantizing a k-quant weight
// to dense float32 at load time (4 bytes/weight), we keep the super-blocks packed
// (~0.5-0.9 bytes/weight) and dequantize-and-dot on the fly here. Decode reads
// ~4-6x fewer bytes per token, which is the dominant cost of token generation.
//
// These are the scalar reference kernels; they mirror the dequantizers in
// kquants.go exactly (validated by parity tests) and dot each super-block's
// weights against the matching slice of the float activation row. SIMD
// (AVX2/NEON) versions can replace the inner loops later without changing the
// packed format or the dispatch.
//
// Super-block sizes: Q4_K 144 B, Q5_K 176 B, Q6_K 210 B — all 256 elements.

const (
	q4kBlockBytes = 144
	q5kBlockBytes = 176
	q6kBlockBytes = 210
)

// q4kDotRowF32 returns the dot product of one Q4_K weight row (nSB super-blocks
// at raw[rOff:]) with x[xOff:xOff+nSB*256].
func q4kDotRowF32(raw []byte, rOff int, x []float32, xOff, nSB int) float32 {
	var sum float32
	for b := 0; b < nSB; b++ {
		sum += q4kDotBlockF32(raw, rOff+b*q4kBlockBytes, x, xOff+b*256)
	}
	return sum
}

func q4kDotBlockF32(raw []byte, off int, x []float32, xb int) float32 {
	d := float32(float16ToFloat64(binary.LittleEndian.Uint16(raw[off:])))
	dmin := float32(float16ToFloat64(binary.LittleEndian.Uint16(raw[off+2:])))
	scales := raw[off+4 : off+16]
	qs := raw[off+16 : off+144]
	var sum float32
	is, qo, xi := 0, 0, xb
	for j := 0; j < 256; j += 64 {
		sc, m := getScaleMinK4(is, scales)
		d1, m1 := d*float32(sc), dmin*float32(m)
		sc, m = getScaleMinK4(is+1, scales)
		d2, m2 := d*float32(sc), dmin*float32(m)
		for l := 0; l < 32; l++ {
			sum += (d1*float32(qs[qo+l]&0x0F) - m1) * x[xi+l]
		}
		for l := 0; l < 32; l++ {
			sum += (d2*float32(qs[qo+l]>>4) - m2) * x[xi+32+l]
		}
		qo += 32
		is += 2
		xi += 64
	}
	return sum
}

// q5kDotRowF32 / q5kDotBlockF32 — Q5_K (low 4 bits in qs, 5th bit in qh).
func q5kDotRowF32(raw []byte, rOff int, x []float32, xOff, nSB int) float32 {
	var sum float32
	for b := 0; b < nSB; b++ {
		sum += q5kDotBlockF32(raw, rOff+b*q5kBlockBytes, x, xOff+b*256)
	}
	return sum
}

func q5kDotBlockF32(raw []byte, off int, x []float32, xb int) float32 {
	d := float32(float16ToFloat64(binary.LittleEndian.Uint16(raw[off:])))
	dmin := float32(float16ToFloat64(binary.LittleEndian.Uint16(raw[off+2:])))
	scales := raw[off+4 : off+16]
	qh := raw[off+16 : off+48]
	qs := raw[off+48 : off+176]
	var sum float32
	is, qo, xi := 0, 0, xb
	u1, u2 := byte(1), byte(2)
	for j := 0; j < 256; j += 64 {
		sc, m := getScaleMinK4(is, scales)
		d1, m1 := d*float32(sc), dmin*float32(m)
		sc, m = getScaleMinK4(is+1, scales)
		d2, m2 := d*float32(sc), dmin*float32(m)
		for l := 0; l < 32; l++ {
			var hi float32
			if qh[l]&u1 != 0 {
				hi = 16
			}
			sum += (d1*(float32(qs[qo+l]&0x0F)+hi) - m1) * x[xi+l]
		}
		for l := 0; l < 32; l++ {
			var hi float32
			if qh[l]&u2 != 0 {
				hi = 16
			}
			sum += (d2*(float32(qs[qo+l]>>4)+hi) - m2) * x[xi+32+l]
		}
		qo += 32
		is += 2
		xi += 64
		u1 <<= 2
		u2 <<= 2
	}
	return sum
}

// q6kDotRowF32 / q6kDotBlockF32 — Q6_K (6-bit: low 4 in ql, high 2 in qh, signed).
func q6kDotRowF32(raw []byte, rOff int, x []float32, xOff, nSB int) float32 {
	var sum float32
	for b := 0; b < nSB; b++ {
		sum += q6kDotBlockF32(raw, rOff+b*q6kBlockBytes, x, xOff+b*256)
	}
	return sum
}

func q6kDotBlockF32(raw []byte, off int, x []float32, xb int) float32 {
	d := float32(float16ToFloat64(binary.LittleEndian.Uint16(raw[off+208:])))
	var sum float32
	qlOff, qhOff, scOff, xi := off, off+128, off+192, xb
	for n := 0; n < 256; n += 128 {
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := float32(int8((raw[qlOff+l]&0xF)|((raw[qhOff+l]>>0)&3)<<4) - 32)
			q2 := float32(int8((raw[qlOff+l+32]&0xF)|((raw[qhOff+l]>>2)&3)<<4) - 32)
			q3 := float32(int8((raw[qlOff+l]>>4)|((raw[qhOff+l]>>4)&3)<<4) - 32)
			q4 := float32(int8((raw[qlOff+l+32]>>4)|((raw[qhOff+l]>>6)&3)<<4) - 32)
			sc0 := float32(int8(raw[scOff+is+0]))
			sc2 := float32(int8(raw[scOff+is+2]))
			sc4 := float32(int8(raw[scOff+is+4]))
			sc6 := float32(int8(raw[scOff+is+6]))
			sum += d * sc0 * q1 * x[xi+l]
			sum += d * sc2 * q2 * x[xi+l+32]
			sum += d * sc4 * q3 * x[xi+l+64]
			sum += d * sc6 * q4 * x[xi+l+96]
		}
		xi += 128
		qlOff += 64
		qhOff += 32
		scOff += 8
	}
	return sum
}
