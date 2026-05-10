package scriptlingllmlib

import (
	"context"
	"math"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

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
