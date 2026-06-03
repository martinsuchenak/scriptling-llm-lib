package scriptlingllmlib

import (
	"context"
	"encoding/binary"
	"math"
	"sync"
	"unsafe"
)

type KVCacheF32 struct {
	K [][]float32
	V [][]float32
}

type TransformerBlockF32 struct {
	AttnNormW []float32
	WQ        interface{}
	WK        interface{}
	WV        interface{}
	WO        interface{}
	QBias     []float32
	KBias     []float32
	VBias     []float32
	QNormW    []float32
	KNormW    []float32
	FFNNormW  []float32
	WGate     interface{}
	WUp       interface{}
	WDown     interface{}
}

type blockBuffers struct {
	normed     []float32
	qData      []float32
	kData      []float32
	vData      []float32
	attnOut    []float32
	merged     []float32
	oData      []float32
	ffnNormed  []float32
	gateData   []float32
	upData     []float32
	siluData   []float32
	downData   []float32
	outputNorm []float32
	logits     []float32
}

type InferenceModelF32 struct {
	Config      ModelConfig
	Arch        string
	TokenEmb    []float32
	EmbRows     int
	EmbCols     int
	Blocks      []TransformerBlockF32
	FinalNormW  []float32
	OutputW     interface{}
	KVCaches    []KVCacheF32
	Tokenizer   *Tokenizer
	ChatTpl     string
	nHeads      int
	nKVHeads    int
	dK          int
	nRep        int
	ropeFreqs   []float32
	ropeHalfDim int
	bufs        blockBuffers
	xDataBuf    []float32
	PrefillMs   float64
	DecodeMs    float64
	// ctx, if set, cancels generation between decode steps. It lives on the
	// per-request clone, so concurrent requests each carry their own.
	ctx context.Context
	// onToken, if set, receives each decoded text delta as it is generated.
	// Like ctx, it lives on the per-request clone.
	onToken func(string)
}

func growSlice(b []float32, n int) []float32 {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]float32, n)
}

func readF16F32(b []byte, off int) float32 {
	u16 := binary.LittleEndian.Uint16(b[off:])
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

var matmulPool = sync.Pool{
	New: func() interface{} {
		b := make([]float32, 0, 8192)
		return &b
	},
}

func matmulAlloc(n int) []float32 {
	iface := matmulPool.Get()
	var b []float32
	if iface != nil {
		bp := iface.(*[]float32)
		b = *bp
	}
	if cap(b) < n {
		b = make([]float32, n)
	} else {
		b = b[:n]
	}
	return b
}

type f32BufPool struct {
	pool sync.Pool
}

func (p *f32BufPool) get(n int) []float32 {
	iface := p.pool.Get()
	var b []float32
	if iface != nil {
		bp := iface.(*[]float32)
		b = *bp
	}
	if cap(b) < n {
		b = make([]float32, n)
	} else {
		b = b[:n]
		for i := range b {
			b[i] = 0
		}
	}
	return b
}

func (p *f32BufPool) put(b []float32) {
	bp := &b
	p.pool.Put(bp)
}

var bp f32BufPool

func rmsNormFlatF32(xData, w []float32, eps float32, rows, cols int, out []float32) {
	for r := 0; r < rows; r++ {
		off := r * cols
		var ss float32
		for j := 0; j < cols; j++ {
			v := xData[off+j]
			ss += v * v
		}
		inv := 1.0 / float32(math.Sqrt(float64(ss)/float64(cols)+float64(eps)))
		for j := 0; j < cols; j++ {
			out[off+j] = xData[off+j] * inv * w[j]
		}
	}
}

func applyRopeInPlaceF32(data []float32, rows, dK, startPos int, freqs []float32, halfDim int) {
	for r := 0; r < rows; r++ {
		pos := startPos + r
		off := r * dK
		for i := 0; i < halfDim; i++ {
			angle := freqs[i] * float32(pos)
			cosA := float32(math.Cos(float64(angle)))
			sinA := float32(math.Sin(float64(angle)))
			x0 := data[off+2*i]
			x1 := data[off+2*i+1]
			data[off+2*i] = x0*cosA - x1*sinA
			data[off+2*i+1] = x0*sinA + x1*cosA
		}
	}
}

func fusedAttentionHeadF32(qData, kData, vData []float32, qRows, dK, kRows int, causal bool, cacheLen int, out []float32, outOff int) {
	scale := 1.0 / float32(math.Sqrt(float64(dK)))
	scores := bp.get(kRows)
	exps := bp.get(kRows)
	defer bp.put(scores)
	defer bp.put(exps)

	for qi := 0; qi < qRows; qi++ {
		qOff := qi * dK
		for ki := 0; ki < kRows; ki++ {
			kOff := ki * dK
			var dot float32
			for d := 0; d < dK; d++ {
				dot += qData[qOff+d] * kData[kOff+d]
			}
			scores[ki] = dot * scale
		}
		if causal {
			limit := cacheLen + qi + 1
			for ki := limit; ki < kRows; ki++ {
				scores[ki] = float32(math.Inf(-1))
			}
		}
		maxScore := scores[0]
		for ki := 1; ki < kRows; ki++ {
			if scores[ki] > maxScore {
				maxScore = scores[ki]
			}
		}
		var sumExp float32
		for ki := 0; ki < kRows; ki++ {
			e := float32(math.Exp(float64(scores[ki] - maxScore)))
			exps[ki] = e
			sumExp += e
		}
		invSum := 1.0 / sumExp
		rowOff := outOff + qi*dK
		for d := 0; d < dK; d++ {
			var val float32
			for ki := 0; ki < kRows; ki++ {
				val += exps[ki] * invSum * vData[ki*dK+d]
			}
			out[rowOff+d] = val
		}
	}
}

func splitHeadsDataF32(data []float32, rows, cols, nHeads int) [][]float32 {
	dK := cols / nHeads
	heads := make([][]float32, nHeads)
	for h := 0; h < nHeads; h++ {
		headData := make([]float32, rows*dK)
		for r := 0; r < rows; r++ {
			srcOff := r*cols + h*dK
			dstOff := r * dK
			copy(headData[dstOff:dstOff+dK], data[srcOff:srcOff+dK])
		}
		heads[h] = headData
	}
	return heads
}

func repeatKVDataF32(heads [][]float32, nRep int) [][]float32 {
	expanded := make([][]float32, 0, len(heads)*nRep)
	for _, h := range heads {
		for i := 0; i < nRep; i++ {
			expanded = append(expanded, h)
		}
	}
	return expanded
}

func mergeHeadsDataF32(data []float32, seqLen, nHeads, dK int) []float32 {
	totalCols := nHeads * dK
	result := make([]float32, seqLen*totalCols)
	for s := 0; s < seqLen; s++ {
		for h := 0; h < nHeads; h++ {
			srcOff := h*seqLen*dK + s*dK
			dstOff := s*totalCols + h*dK
			copy(result[dstOff:dstOff+dK], data[srcOff:srcOff+dK])
		}
	}
	return result
}

func q8DotGroupXF32(raw []byte, rawOff int, xData []float32, xBase int) float32 {
	s := readF16F32(raw, rawOff)
	q := unsafe.Slice((*byte)(unsafe.Pointer(&raw[rawOff+2])), 32)
	qw := unsafe.Slice((*uint64)(unsafe.Pointer(&q[0])), 4)

	var sum float32
	for w := 0; w < 4; w++ {
		chunk := qw[w]
		b0 := float32(int8(chunk))
		b1 := float32(int8(chunk >> 8))
		b2 := float32(int8(chunk >> 16))
		b3 := float32(int8(chunk >> 24))
		b4 := float32(int8(chunk >> 32))
		b5 := float32(int8(chunk >> 40))
		b6 := float32(int8(chunk >> 48))
		b7 := float32(int8(chunk >> 56))
		i := xBase + w*8
		sum += s * (b0*xData[i] + b1*xData[i+1] + b2*xData[i+2] + b3*xData[i+3] +
			b4*xData[i+4] + b5*xData[i+5] + b6*xData[i+6] + b7*xData[i+7])
	}
	return sum
}

func q4DotGroupXF32(raw []byte, rawOff int, xData []float32, xBase int) float32 {
	s := readF16F32(raw, rawOff)
	var sum float32
	for j := 0; j < 16; j += 4 {
		b0 := raw[rawOff+2+j+0]
		b1 := raw[rawOff+2+j+1]
		b2 := raw[rawOff+2+j+2]
		b3 := raw[rawOff+2+j+3]
		sum += float32(int8(b0&0xF)-8)*xData[xBase+j+0] + float32(int8(b0>>4)-8)*xData[xBase+j+16]
		sum += float32(int8(b1&0xF)-8)*xData[xBase+j+1] + float32(int8(b1>>4)-8)*xData[xBase+j+17]
		sum += float32(int8(b2&0xF)-8)*xData[xBase+j+2] + float32(int8(b2>>4)-8)*xData[xBase+j+18]
		sum += float32(int8(b3&0xF)-8)*xData[xBase+j+3] + float32(int8(b3>>4)-8)*xData[xBase+j+19]
	}
	return s * sum
}

func q4_1DotGroupXF32(raw []byte, rawOff int, xData []float32, xBase int) float32 {
	d := readF16F32(raw, rawOff)
	m := readF16F32(raw, rawOff+2)
	var qSum, xSum float32
	for j := 0; j < 16; j += 4 {
		b0 := raw[rawOff+4+j+0]
		b1 := raw[rawOff+4+j+1]
		b2 := raw[rawOff+4+j+2]
		b3 := raw[rawOff+4+j+3]
		qSum += float32(b0&0xF)*xData[xBase+j+0] + float32(b0>>4)*xData[xBase+j+16]
		qSum += float32(b1&0xF)*xData[xBase+j+1] + float32(b1>>4)*xData[xBase+j+17]
		qSum += float32(b2&0xF)*xData[xBase+j+2] + float32(b2>>4)*xData[xBase+j+18]
		qSum += float32(b3&0xF)*xData[xBase+j+3] + float32(b3>>4)*xData[xBase+j+19]
		xSum += xData[xBase+j+0] + xData[xBase+j+16]
		xSum += xData[xBase+j+1] + xData[xBase+j+17]
		xSum += xData[xBase+j+2] + xData[xBase+j+18]
		xSum += xData[xBase+j+3] + xData[xBase+j+19]
	}
	return d*qSum + m*xSum
}

func q5DotGroupXF32(raw []byte, rawOff int, xData []float32, xBase int) float32 {
	s := readF16F32(raw, rawOff)
	qh := binary.LittleEndian.Uint32(raw[rawOff+2:])
	qs := unsafe.Slice((*byte)(unsafe.Pointer(&raw[rawOff+6])), 16)

	var sum float32
	for j := 0; j < 16; j += 4 {
		xh0_0 := byte((qh>>(j+0))<<4) & 0x10
		xh1_0 := byte(qh>>(j+12)) & 0x10
		v0_0 := float32(int32(qs[j+0]&0x0F|xh0_0)) - 16
		v1_0 := float32(int32(qs[j+0]>>4|xh1_0)) - 16
		sum += v0_0*xData[xBase+j] + v1_0*xData[xBase+j+16]

		xh0_1 := byte((qh>>(j+1))<<4) & 0x10
		xh1_1 := byte(qh>>(j+13)) & 0x10
		v0_1 := float32(int32(qs[j+1]&0x0F|xh0_1)) - 16
		v1_1 := float32(int32(qs[j+1]>>4|xh1_1)) - 16
		sum += v0_1*xData[xBase+j+1] + v1_1*xData[xBase+j+17]

		xh0_2 := byte((qh>>(j+2))<<4) & 0x10
		xh1_2 := byte(qh>>(j+14)) & 0x10
		v0_2 := float32(int32(qs[j+2]&0x0F|xh0_2)) - 16
		v1_2 := float32(int32(qs[j+2]>>4|xh1_2)) - 16
		sum += v0_2*xData[xBase+j+2] + v1_2*xData[xBase+j+18]

		xh0_3 := byte((qh>>(j+3))<<4) & 0x10
		xh1_3 := byte(qh>>(j+15)) & 0x10
		v0_3 := float32(int32(qs[j+3]&0x0F|xh0_3)) - 16
		v1_3 := float32(int32(qs[j+3]>>4|xh1_3)) - 16
		sum += v0_3*xData[xBase+j+3] + v1_3*xData[xBase+j+19]
	}
	return s * sum
}

func modelMatmulIntoF32(w interface{}, xData []float32, xRows, xCols int, dst []float32) ([]float32, int, int) {
	switch wt := w.(type) {
	case *QuantWeight:
		return modelMatmulQuantIntoF32(wt, xData, xRows, xCols, dst)
	case []float32:
		return modelMatmulFloatIntoF32(wt, xData, xRows, xCols, dst)
	}
	return nil, 0, 0
}

func modelMatmulQuantIntoF32(w *QuantWeight, xData []float32, xRows, xCols int, dst []float32) ([]float32, int, int) {
	rawBytes := w.Raw
	outFeatures := w.Rows
	result := growSlice(dst, xRows*outFeatures)

	switch w.QType {
	case "q8":
		if useInt8Q8 {
			q8q8MatmulInto(w, xData, xRows, xCols, result)
			return result, xRows, outFeatures
		}
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 34
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				result[xi*outFeatures+j] = q8DotRowsF32(rawBytes, j*rowBytes, xData, xi*xCols, groupsPerRow)
			}
		})
	case "q4":
		if useInt8Q4 {
			q4q8MatmulInto(w, xData, xRows, xCols, result)
			return result, xRows, outFeatures
		}
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 18
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				result[xi*outFeatures+j] = q4DotRowsF32(rawBytes, j*rowBytes, xData, xi*xCols, groupsPerRow)
			}
		})
	case "q4_1":
		if useInt8Q41 {
			q41q8MatmulInto(w, xData, xRows, xCols, result)
			return result, xRows, outFeatures
		}
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 20
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				result[xi*outFeatures+j] = q41DotRowsF32(rawBytes, j*rowBytes, xData, xi*xCols, groupsPerRow)
			}
		})
	case "q5":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 22
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				result[xi*outFeatures+j] = q5DotRowsF32(rawBytes, j*rowBytes, xData, xi*xCols, groupsPerRow)
			}
		})
	case "q5k1":
		q5k1MatmulInto(w, xData, xRows, xCols, result)
	case "q4k", "q5k", "q6k":
		nSB := w.Groups
		var rowBytes int
		var dot func([]byte, int, []float32, int, int) float32
		switch w.QType {
		case "q4k":
			rowBytes, dot = nSB*q4kBlockBytes, q4kDotRowF32
		case "q5k":
			rowBytes, dot = nSB*q5kBlockBytes, q5kDotRowF32
		case "q6k":
			rowBytes, dot = nSB*q6kBlockBytes, q6kDotRowF32
		}
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				result[xi*outFeatures+j] = dot(rawBytes, j*rowBytes, xData, xi*xCols, nSB)
			}
		})
	}
	return result, xRows, outFeatures
}

func modelMatmulFloatIntoF32(wFlat []float32, xData []float32, xRows, xCols int, dst []float32) ([]float32, int, int) {
	if len(wFlat) == 0 || xCols == 0 {
		return nil, 0, 0
	}
	wRows := len(wFlat) / xCols
	result := growSlice(dst, xRows*wRows)
	parallelFor(xRows*wRows, func(start, end int) {
		for idx := start; idx < end; idx++ {
			xi := idx / wRows
			j := idx % wRows
			var sum float32
			xOff := xi * xCols
			wOff := j * xCols
			for l := 0; l < xCols; l++ {
				sum += xData[xOff+l] * wFlat[wOff+l]
			}
			result[xi*wRows+j] = sum
		}
	})
	return result, xRows, wRows
}

func q8DotRowsF32(raw []byte, rOff int, xData []float32, xOff, groups int) float32 {
	if groups > 0 {
		return q8DotRowsAsm(&raw[rOff], &xData[xOff], groups)
	}
	return 0
}

func q4DotRowsF32(raw []byte, rOff int, xData []float32, xOff, groups int) float32 {
	var sum float32
	for g := 0; g < groups; g++ {
		sum += q4DotGroupXF32(raw, rOff+g*18, xData, xOff+g*32)
	}
	return sum
}

func q41DotRowsF32(raw []byte, rOff int, xData []float32, xOff, groups int) float32 {
	var sum float32
	for g := 0; g < groups; g++ {
		sum += q4_1DotGroupXF32(raw, rOff+g*20, xData, xOff+g*32)
	}
	return sum
}

func q5DotRowsF32(raw []byte, rOff int, xData []float32, xOff, groups int) float32 {
	var sum float32
	for g := 0; g < groups; g++ {
		sum += q5DotGroupXF32(raw, rOff+g*22, xData, xOff+g*32)
	}
	return sum
}

func modelMatmulQuantF32(w *QuantWeight, xData []float32, xRows, xCols int) ([]float32, int, int) {
	rawBytes := w.Raw
	outFeatures := w.Rows

	switch w.QType {
	case "q8":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 34
		result := matmulAlloc(xRows * outFeatures)
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				xOff := xi * xCols
				rOff := j * rowBytes
				var sum float32
				for g := 0; g < groupsPerRow; g++ {
					sum += q8DotGroupXF32(rawBytes, rOff+g*34, xData, xOff+g*32)
				}
				result[xi*outFeatures+j] = sum
			}
		})
		return result, xRows, outFeatures
	case "q4":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 18
		result := matmulAlloc(xRows * outFeatures)
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				xOff := xi * xCols
				rOff := j * rowBytes
				var sum float32
				for g := 0; g < groupsPerRow; g++ {
					sum += q4DotGroupXF32(rawBytes, rOff+g*18, xData, xOff+g*32)
				}
				result[xi*outFeatures+j] = sum
			}
		})
		return result, xRows, outFeatures
	case "q4_1":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 20
		result := matmulAlloc(xRows * outFeatures)
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				xOff := xi * xCols
				rOff := j * rowBytes
				var sum float32
				for g := 0; g < groupsPerRow; g++ {
					sum += q4_1DotGroupXF32(rawBytes, rOff+g*20, xData, xOff+g*32)
				}
				result[xi*outFeatures+j] = sum
			}
		})
		return result, xRows, outFeatures
	case "q5":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 22
		result := matmulAlloc(xRows * outFeatures)
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				xOff := xi * xCols
				rOff := j * rowBytes
				var sum float32
				for g := 0; g < groupsPerRow; g++ {
					sum += q5DotGroupXF32(rawBytes, rOff+g*22, xData, xOff+g*32)
				}
				result[xi*outFeatures+j] = sum
			}
		})
		return result, xRows, outFeatures
	}

	return nil, 0, 0
}

func modelMatmulFloatF32(wFlat []float32, xData []float32, xRows, xCols int) ([]float32, int, int) {
	if len(wFlat) == 0 || xCols == 0 {
		return nil, 0, 0
	}
	wCols := xCols
	wRows := len(wFlat) / wCols
	result := matmulAlloc(xRows * wRows)
	parallelFor(xRows*wRows, func(start, end int) {
		for idx := start; idx < end; idx++ {
			xi := idx / wRows
			j := idx % wRows
			xOff := xi * xCols
			wOff := j * wCols
			var sum float32
			for l := 0; l < xCols; l++ {
				sum += xData[xOff+l] * wFlat[wOff+l]
			}
			result[xi*wRows+j] = sum
		}
	})
	return result, xRows, wRows
}

func modelMatmulRowQuantIntoF32(w *QuantWeight, normed []float32, dst []float32) []float32 {
	rawBytes := w.Raw
	outFeatures := w.Rows
	result := growSlice(dst, outFeatures)

	switch w.QType {
	case "q8":
		if useInt8Q8 {
			q8q8MatmulInto(w, normed, 1, w.Cols, result)
			return result
		}
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 34
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				result[j] = q8DotRowsF32(rawBytes, j*rowBytes, normed, 0, groupsPerRow)
			}
		})
		return result
	case "q4":
		if useInt8Q4 {
			q4q8MatmulInto(w, normed, 1, w.Cols, result)
			return result
		}
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 18
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				result[j] = q4DotRowsF32(rawBytes, j*rowBytes, normed, 0, groupsPerRow)
			}
		})
		return result
	case "q4_1":
		if useInt8Q41 {
			q41q8MatmulInto(w, normed, 1, w.Cols, result)
			return result
		}
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 20
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				result[j] = q41DotRowsF32(rawBytes, j*rowBytes, normed, 0, groupsPerRow)
			}
		})
		return result
	case "q5":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 22
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				result[j] = q5DotRowsF32(rawBytes, j*rowBytes, normed, 0, groupsPerRow)
			}
		})
		return result
	case "q5k1":
		q5k1MatmulInto(w, normed, 1, w.Cols, result)
		return result
	case "q4k":
		nSB := w.Groups
		rowBytes := nSB * q4kBlockBytes
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				result[j] = q4kDotRowF32(rawBytes, j*rowBytes, normed, 0, nSB)
			}
		})
		return result
	case "q5k":
		nSB := w.Groups
		rowBytes := nSB * q5kBlockBytes
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				result[j] = q5kDotRowF32(rawBytes, j*rowBytes, normed, 0, nSB)
			}
		})
		return result
	case "q6k":
		nSB := w.Groups
		rowBytes := nSB * q6kBlockBytes
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				result[j] = q6kDotRowF32(rawBytes, j*rowBytes, normed, 0, nSB)
			}
		})
		return result
	}

	return nil
}

func modelMatmulRowQuantF32(w *QuantWeight, normed []float32) []float32 {
	rawBytes := w.Raw
	outFeatures := w.Rows

	switch w.QType {
	case "q8":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 34
		result := matmulAlloc(outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rOff := j * rowBytes
				var sum float32
				for g := 0; g < groupsPerRow; g++ {
					sum += q8DotGroupXF32(rawBytes, rOff+g*34, normed, g*32)
				}
				result[j] = sum
			}
		})
		return result
	case "q4":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 18
		result := matmulAlloc(outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rOff := j * rowBytes
				var sum float32
				for g := 0; g < groupsPerRow; g++ {
					sum += q4DotGroupXF32(rawBytes, rOff+g*18, normed, g*32)
				}
				result[j] = sum
			}
		})
		return result
	case "q4_1":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 20
		result := matmulAlloc(outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rOff := j * rowBytes
				var sum float32
				for g := 0; g < groupsPerRow; g++ {
					sum += q4_1DotGroupXF32(rawBytes, rOff+g*20, normed, g*32)
				}
				result[j] = sum
			}
		})
		return result
	case "q5":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 22
		result := matmulAlloc(outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rOff := j * rowBytes
				var sum float32
				for g := 0; g < groupsPerRow; g++ {
					sum += q5DotGroupXF32(rawBytes, rOff+g*22, normed, g*32)
				}
				result[j] = sum
			}
		})
		return result
	}

	return nil
}

func modelMatmulRowFloatF32(wFlat []float32, normed []float32) []float32 {
	wRows := len(wFlat) / len(normed)
	wCols := len(normed)
	logits := matmulAlloc(wRows)
	parallelFor(wRows, func(start, end int) {
		for j := start; j < end; j++ {
			wOff := j * wCols
			var sum float32
			for l := 0; l < wCols; l++ {
				sum += normed[l] * wFlat[wOff+l]
			}
			logits[j] = sum
		}
	})
	return logits
}

func f64ToF32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func f32ToF64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

func argmaxF32(v []float32) int {
	best := 0
	bv := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bv {
			bv = v[i]
			best = i
		}
	}
	return best
}
