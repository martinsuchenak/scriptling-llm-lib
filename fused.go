package scriptlingllmlib

import (
	"context"
	"encoding/binary"
	"math"
	"unsafe"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func q4kDotBlockFast(raw []byte, blkOff int, x []float64, xOff int) float64 {
	d := float16ToFloat64(binary.LittleEndian.Uint16(raw[blkOff:]))
	dmin := float16ToFloat64(binary.LittleEndian.Uint16(raw[blkOff+2:]))
	scalesOff := blkOff + 4
	qsOff := blkOff + 16

	var sum float64
	is := 0
	qPos := 0

	for group := 0; group < 4; group++ {
		sc0, m0 := getScaleMinK4(is, raw[scalesOff:])
		sc1, m1 := getScaleMinK4(is+1, raw[scalesOff:])
		d1 := d * float64(sc0)
		m1v := dmin * float64(m0)
		d2 := d * float64(sc1)
		m2v := dmin * float64(m1)

		qBase := qsOff + qPos
		xBase := xOff + group*64

		qw := unsafe.Slice((*uint64)(unsafe.Pointer(&raw[qBase])), 4)
		for w := 0; w < 4; w++ {
			chunk := qw[w]
			b0 := byte(chunk)
			b1 := byte(chunk >> 8)
			b2 := byte(chunk >> 16)
			b3 := byte(chunk >> 24)
			b4 := byte(chunk >> 32)
			b5 := byte(chunk >> 40)
			b6 := byte(chunk >> 48)
			b7 := byte(chunk >> 56)
			i := xBase + w*8
			sum += (d1*float64(b0&0xF)-m1v)*x[i] + (d1*float64(b1&0xF)-m1v)*x[i+1] +
				(d1*float64(b2&0xF)-m1v)*x[i+2] + (d1*float64(b3&0xF)-m1v)*x[i+3] +
				(d1*float64(b4&0xF)-m1v)*x[i+4] + (d1*float64(b5&0xF)-m1v)*x[i+5] +
				(d1*float64(b6&0xF)-m1v)*x[i+6] + (d1*float64(b7&0xF)-m1v)*x[i+7]
		}
		for w := 0; w < 4; w++ {
			chunk := qw[w]
			b0 := byte(chunk)
			b1 := byte(chunk >> 8)
			b2 := byte(chunk >> 16)
			b3 := byte(chunk >> 24)
			b4 := byte(chunk >> 32)
			b5 := byte(chunk >> 40)
			b6 := byte(chunk >> 48)
			b7 := byte(chunk >> 56)
			i := xBase + 32 + w*8
			sum += (d2*float64(b0>>4)-m2v)*x[i] + (d2*float64(b1>>4)-m2v)*x[i+1] +
				(d2*float64(b2>>4)-m2v)*x[i+2] + (d2*float64(b3>>4)-m2v)*x[i+3] +
				(d2*float64(b4>>4)-m2v)*x[i+4] + (d2*float64(b5>>4)-m2v)*x[i+5] +
				(d2*float64(b6>>4)-m2v)*x[i+6] + (d2*float64(b7>>4)-m2v)*x[i+7]
		}

		qPos += 32
		is += 2
	}

	return sum
}

func math_sqrt(x float64) float64 {
	return math.Sqrt(x)
}

func fnOutputLogits(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 3); err != nil {
		return err
	}

	xData, xRows, xCols, ok := object.GetFloatMatrix(args[0])
	if !ok {
		return errors.NewTypeError("FLOAT_MATRIX", args[0].Type().String())
	}
	if xRows == 0 {
		return errors.NewError("output_logits: x cannot be empty")
	}

	normW, err := toFloatList(args[1], "output_logits", "norm_weight")
	if err != nil {
		return err
	}
	eps := 1e-5
	if kwargs.Has("eps") {
		eps = kwargs.MustGetFloat("eps", 1e-5)
	}

	lastOff := (xRows - 1) * xCols
	var ss float64
	for j := 0; j < xCols; j++ {
		ss += xData[lastOff+j] * xData[lastOff+j]
	}
	inv := 1.0 / math_sqrt(ss/float64(xCols)+eps)

	normed := make([]float64, xCols)
	for j := 0; j < xCols; j++ {
		normed[j] = xData[lastOff+j] * inv * normW[j]
	}

	outW := args[2]

	switch w := outW.(type) {
	case *object.String:
		return outputLogitsQ8(normed, w)
	default:
		if wData, wRows, wCols, wok := object.GetFloatMatrix(outW); wok {
			return outputLogitsFloat(normed, wData, wRows, wCols)
		}
	}

	return errors.NewError("output_logits: unsupported weight type")
}

func outputLogitsQ8(normed []float64, raw *object.String) object.Object {
	rawBytes := []byte(raw.Value)
	groupsPerRow := len(normed) / 32
	rowBytes := groupsPerRow * 34
	outFeatures := len(rawBytes) / rowBytes

	logits := make([]float64, outFeatures)
	parallelFor(outFeatures, func(start, end int) {
		for j := start; j < end; j++ {
			rOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				sum += q8DotGroupX(rawBytes, rOff+g*34, normed, g*32)
			}
			logits[j] = sum
		}
	})
	return object.NewFloatArray1D(logits)
}

func outputLogitsFloat(normed []float64, wData []float64, wRows, wCols int) object.Object {
	logits := make([]float64, wRows)
	parallelFor(wRows, func(start, end int) {
		for j := start; j < end; j++ {
			wOff := j * wCols
			var sum float64
			for l := 0; l < wCols; l++ {
				sum += normed[l] * wData[wOff+l]
			}
			logits[j] = sum
		}
	})
	return object.NewFloatArray1D(logits)
}
