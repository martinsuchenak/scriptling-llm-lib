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

// convertQ4KToQ41 re-expresses Q4_K weights as Q4_1, which is bit-identical at
// the sub-block level (Q4_K sub-block j: w = d*sc_j*q4 - dmin*m_j; Q4_1 block:
// w = q4*scale + min, so scale=d*sc_j, min=-dmin*m_j, same 4-bit q4). The payoff
// is reusing the existing fused q41q8 int8 SIMD kernel — fast and ~lossless,
// versus a bespoke Q4_K kernel. Output is laid out exactly like a native Q4_1
// tensor: rows of (cols/32) 20-byte blocks in output order.
func convertQ4KToQ41(src []byte, srcOff, rows, cols int) []byte {
	nSB := cols / 256 // Q4_K super-blocks per row
	out := make([]byte, rows*(cols/32)*20)
	oi := 0
	for r := 0; r < rows; r++ {
		rowOff := srcOff + r*nSB*q4kBlockBytes
		for sb := 0; sb < nSB; sb++ {
			off := rowOff + sb*q4kBlockBytes
			d := float16ToFloat64(binary.LittleEndian.Uint16(src[off:]))
			dmin := float16ToFloat64(binary.LittleEndian.Uint16(src[off+2:]))
			scales := src[off+4 : off+16]
			qs := src[off+16:] // 128 bytes for this super-block
			for jj := 0; jj < 8; jj++ {
				sc, m := getScaleMinK4(jj, scales)
				binary.LittleEndian.PutUint16(out[oi:], float64ToFloat16(d*float64(sc)))
				binary.LittleEndian.PutUint16(out[oi+2:], float64ToFloat16(-dmin*float64(m)))
				k := (jj / 2) * 32
				// Pack the sub-block's 32 nibbles into Q4_1 order: byte i holds
				// element i (low) and element i+16 (high).
				if jj&1 == 0 {
					for i := 0; i < 16; i++ {
						out[oi+4+i] = (qs[k+i] & 0x0F) | ((qs[k+i+16] & 0x0F) << 4)
					}
				} else {
					for i := 0; i < 16; i++ {
						out[oi+4+i] = (qs[k+i] >> 4) | ((qs[k+i+16] >> 4) << 4)
					}
				}
				oi += 20
			}
		}
	}
	return out
}

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

// convertQ5KToTwoQ41 expresses Q5_K as two Q4_1 weights that sum to it, so the
// fast q41q8 kernel can be run twice (no bespoke Q5_K kernel). A Q5_K value is
// q5 = q4 + 16*high_bit, so
//   w = scale*q5 + min = (scale*q4 + min) + (16*scale)*high_bit
// The first term is a Q4_1 block (low nibbles, scale, min); the second is a Q4_1
// block (the high bit as a 0/1 "q4", scale 16*scale, min 0). Output is
// [low blocks for all rows][high blocks for all rows], each half a native Q4_1
// tensor of (cols/32) 20-byte blocks per row.
func convertQ5KToTwoQ41(src []byte, srcOff, rows, cols int) []byte {
	nSB := cols / 256
	half := rows * (cols / 32) * 20
	out := make([]byte, 2*half)
	lo, hi := 0, half
	for r := 0; r < rows; r++ {
		rowOff := srcOff + r*nSB*q5kBlockBytes
		for sb := 0; sb < nSB; sb++ {
			off := rowOff + sb*q5kBlockBytes
			d := float16ToFloat64(binary.LittleEndian.Uint16(src[off:]))
			dmin := float16ToFloat64(binary.LittleEndian.Uint16(src[off+2:]))
			scales := src[off+4 : off+16]
			qh := src[off+16 : off+48] // 32 bytes (256 high bits)
			qs := src[off+48:]         // 128 bytes (256 low nibbles)
			for jj := 0; jj < 8; jj++ {
				sc, m := getScaleMinK4(jj, scales)
				dsc := d * float64(sc)

				// Low Q4_1 block: scale=dsc, min=-dmin*m, q4 = low nibble.
				binary.LittleEndian.PutUint16(out[lo:], float64ToFloat16(dsc))
				binary.LittleEndian.PutUint16(out[lo+2:], float64ToFloat16(-dmin*float64(m)))
				k := (jj / 2) * 32
				if jj&1 == 0 {
					for i := 0; i < 16; i++ {
						out[lo+4+i] = (qs[k+i] & 0x0F) | ((qs[k+i+16] & 0x0F) << 4)
					}
				} else {
					for i := 0; i < 16; i++ {
						out[lo+4+i] = (qs[k+i] >> 4) | ((qs[k+i+16] >> 4) << 4)
					}
				}
				lo += 20

				// High Q4_1 block: scale=16*dsc, min=0, q4 = (qh>>jj)&1.
				binary.LittleEndian.PutUint16(out[hi:], float64ToFloat16(16*dsc))
				binary.LittleEndian.PutUint16(out[hi+2:], 0)
				for i := 0; i < 16; i++ {
					out[hi+4+i] = ((qh[i] >> jj) & 1) | (((qh[i+16] >> jj) & 1) << 4)
				}
				hi += 20
			}
		}
	}
	return out
}

// q5k1MatmulInto runs the two Q4_1 halves of a converted Q5_K weight through the
// fast q41q8 kernel and sums them (dst = low + high). w.Raw is [low half][high
// half], each a native Q4_1 tensor.
func q5k1MatmulInto(w *QuantWeight, xData []float32, xRows, xCols int, dst []float32) {
	half := w.Rows * w.Groups * 20
	lowW := &QuantWeight{QType: "q4_1", Raw: w.Raw[:half], Groups: w.Groups, Rows: w.Rows, Cols: w.Cols}
	highW := &QuantWeight{QType: "q4_1", Raw: w.Raw[half:], Groups: w.Groups, Rows: w.Rows, Cols: w.Cols}
	q41q8MatmulInto(lowW, xData, xRows, xCols, dst)

	n := xRows * w.Rows
	tmpP := scalePool.Get().(*[]float32)
	tmp := growSlice(*tmpP, n)
	q41q8MatmulInto(highW, xData, xRows, xCols, tmp)
	for i := 0; i < n; i++ {
		dst[i] += tmp[i]
	}
	*tmpP = tmp
	scalePool.Put(tmpP)
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
