package scriptlingllmlib

import "encoding/binary"

// iq4nlValues is the fixed 16-entry non-linear codebook for IQ4_NL / IQ4_XS
// (ggml kvalues_iq4nl). The 4-bit index selects one of these int8 levels.
var iq4nlValues = [16]int8{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

// dequantizeIQ4NLNative decodes IQ4_NL (GGUF type 20): 32-element blocks of
// f16 scale + 16 nibble bytes, weight = d * codebook[nibble]. Nibbles use the
// Q4_0 split layout — qs[j]'s low nibble is element j, its high nibble element
// j+16. IQ4_NL is an i-quant (non-linear codebook) that some k-quant repacks use
// for rows whose width isn't a multiple of 256.
func dequantizeIQ4NLNative(data []byte, offset, nElements int) []float64 {
	const blockBytes = 18
	result := make([]float64, nElements)
	for b := 0; b < nElements/32; b++ {
		base := offset + b*blockBytes
		d := float16ToFloat64(binary.LittleEndian.Uint16(data[base:]))
		qs := data[base+2:]
		for j := 0; j < 16; j++ {
			result[b*32+j] = d * float64(iq4nlValues[qs[j]&0x0F])
			result[b*32+16+j] = d * float64(iq4nlValues[qs[j]>>4])
		}
	}
	return result
}

// dequantizeQ5_1Native decodes Q5_1 (GGUF type 7), a legacy 32-element block
// quant (d, min, 5-bit quants). It is not a k-quant, but k-quant model repacks
// use it as the fallback for rows whose width isn't a multiple of 256, so the
// loader needs it to open those files. Block layout: d(2) min(2) qh(4) qs(16).
func dequantizeQ5_1Native(data []byte, offset, nElements int) []float64 {
	result := make([]float64, nElements)
	for g := 0; g < nElements/32; g++ {
		base := offset + g*24
		d := float16ToFloat64(binary.LittleEndian.Uint16(data[base:]))
		m := float16ToFloat64(binary.LittleEndian.Uint16(data[base+2:]))
		qh := binary.LittleEndian.Uint32(data[base+4:])
		for j := 0; j < 16; j++ {
			q := data[base+8+j]
			xh0 := int32((qh>>uint(j))<<4) & 0x10
			xh1 := int32(qh>>uint(j+12)) & 0x10
			x0 := int32(q&0x0F) | xh0
			x1 := int32(q>>4) | xh1
			result[g*32+j] = float64(x0)*d + m
			result[g*32+j+16] = float64(x1)*d + m
		}
	}
	return result
}

// K-quant (super-block) dequantizers, ported from the canonical ggml reference.
// Every k-quant packs 256 weights (QK_K) per super-block with hierarchical
// scales. These decode a tensor to []float64; the loader then either uses the
// floats directly (1D tensors) or requantizes 2D weights to Q8_0 for the fast
// integer kernels. Block byte sizes:
//   Q2_K = 84, Q3_K = 110, Q4_K = 144, Q5_K = 176, Q6_K = 210 (in gguf.go).

// getScaleMinK4 unpacks the 6-bit scale/min pair j (0..7) from the 12 packed
// scale bytes of a Q4_K / Q5_K super-block (ggml get_scale_min_k4).
func getScaleMinK4(j int, q []byte) (sc, m byte) {
	if j < 4 {
		sc = q[j] & 63
		m = q[j+4] & 63
	} else {
		sc = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
		m = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
	}
	return sc, m
}

func dequantizeQ4KNative(data []byte, offset, nElements int) []float64 {
	const blockBytes = 144
	out := make([]float64, nElements)
	oi := 0
	for b := 0; b < nElements/256; b++ {
		base := offset + b*blockBytes
		d := float16ToFloat64(binary.LittleEndian.Uint16(data[base:]))
		dmin := float16ToFloat64(binary.LittleEndian.Uint16(data[base+2:]))
		scales := data[base+4 : base+16]
		qs := data[base+16 : base+144]
		is, qo := 0, 0
		for j := 0; j < 256; j += 64 {
			sc, m := getScaleMinK4(is, scales)
			d1, m1 := d*float64(sc), dmin*float64(m)
			sc, m = getScaleMinK4(is+1, scales)
			d2, m2 := d*float64(sc), dmin*float64(m)
			for l := 0; l < 32; l++ {
				out[oi] = d1*float64(qs[qo+l]&0x0F) - m1
				oi++
			}
			for l := 0; l < 32; l++ {
				out[oi] = d2*float64(qs[qo+l]>>4) - m2
				oi++
			}
			qo += 32
			is += 2
		}
	}
	return out
}

func dequantizeQ5KNative(data []byte, offset, nElements int) []float64 {
	const blockBytes = 176
	out := make([]float64, nElements)
	oi := 0
	for b := 0; b < nElements/256; b++ {
		base := offset + b*blockBytes
		d := float16ToFloat64(binary.LittleEndian.Uint16(data[base:]))
		dmin := float16ToFloat64(binary.LittleEndian.Uint16(data[base+2:]))
		scales := data[base+4 : base+16]
		qh := data[base+16 : base+48]
		qs := data[base+48 : base+176]
		is, qo := 0, 0
		u1, u2 := byte(1), byte(2)
		for j := 0; j < 256; j += 64 {
			sc, m := getScaleMinK4(is, scales)
			d1, m1 := d*float64(sc), dmin*float64(m)
			sc, m = getScaleMinK4(is+1, scales)
			d2, m2 := d*float64(sc), dmin*float64(m)
			for l := 0; l < 32; l++ {
				hi := 0.0
				if qh[l]&u1 != 0 {
					hi = 16
				}
				out[oi] = d1*(float64(qs[qo+l]&0x0F)+hi) - m1
				oi++
			}
			for l := 0; l < 32; l++ {
				hi := 0.0
				if qh[l]&u2 != 0 {
					hi = 16
				}
				out[oi] = d2*(float64(qs[qo+l]>>4)+hi) - m2
				oi++
			}
			qo += 32
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
	return out
}

func dequantizeQ2KNative(data []byte, offset, nElements int) []float64 {
	const blockBytes = 84
	out := make([]float64, nElements)
	oi := 0
	for b := 0; b < nElements/256; b++ {
		base := offset + b*blockBytes
		scales := data[base : base+16]
		qs := data[base+16 : base+80]
		d := float16ToFloat64(binary.LittleEndian.Uint16(data[base+80:]))
		dmin := float16ToFloat64(binary.LittleEndian.Uint16(data[base+82:]))
		is, qo := 0, 0
		for n := 0; n < 256; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				sc := scales[is]
				is++
				dl, ml := d*float64(sc&0x0F), dmin*float64(sc>>4)
				for l := 0; l < 16; l++ {
					out[oi] = dl*float64((qs[qo+l]>>shift)&3) - ml
					oi++
				}
				sc = scales[is]
				is++
				dl, ml = d*float64(sc&0x0F), dmin*float64(sc>>4)
				for l := 0; l < 16; l++ {
					out[oi] = dl*float64((qs[qo+16+l]>>shift)&3) - ml
					oi++
				}
				shift += 2
			}
			qo += 32
		}
	}
	return out
}

func dequantizeQ3KNative(data []byte, offset, nElements int) []float64 {
	const blockBytes = 110
	const kmask1 = 0x03030303
	const kmask2 = 0x0f0f0f0f
	out := make([]float64, nElements)
	oi := 0
	for b := 0; b < nElements/256; b++ {
		base := offset + b*blockBytes
		hmask := data[base : base+32]
		qs := data[base+32 : base+96]
		scb := data[base+96 : base+108]
		dAll := float16ToFloat64(binary.LittleEndian.Uint16(data[base+108:]))

		// Unpack the 16 six-bit scales (ggml stores them across 12 bytes).
		var aux [4]uint32
		aux[0] = binary.LittleEndian.Uint32(scb[0:])
		aux[1] = binary.LittleEndian.Uint32(scb[4:])
		aux[2] = binary.LittleEndian.Uint32(scb[8:])
		tmp := aux[2]
		aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
		aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
		aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
		aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
		var scales [16]int8
		for k := 0; k < 4; k++ {
			scales[k*4+0] = int8(byte(aux[k]))
			scales[k*4+1] = int8(byte(aux[k] >> 8))
			scales[k*4+2] = int8(byte(aux[k] >> 16))
			scales[k*4+3] = int8(byte(aux[k] >> 24))
		}

		m := byte(1)
		is, qo := 0, 0
		for n := 0; n < 256; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				dl := dAll * float64(int(scales[is])-32)
				is++
				for l := 0; l < 16; l++ {
					sub := 4.0
					if hmask[l]&m != 0 {
						sub = 0
					}
					out[oi] = dl * (float64((qs[qo+l]>>shift)&3) - sub)
					oi++
				}
				dl = dAll * float64(int(scales[is])-32)
				is++
				for l := 0; l < 16; l++ {
					sub := 4.0
					if hmask[l+16]&m != 0 {
						sub = 0
					}
					out[oi] = dl * (float64((qs[qo+16+l]>>shift)&3) - sub)
					oi++
				}
				shift += 2
				m <<= 1
			}
			qo += 32
		}
	}
	return out
}
