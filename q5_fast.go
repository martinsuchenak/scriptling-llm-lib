package scriptlingllmlib

import (
	"context"
	"encoding/binary"
	"unsafe"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func q5DotGroupX(raw []byte, rawOff int, xData []float64, xBase int) float64 {
	s := readF16(raw, rawOff)
	qh := binary.LittleEndian.Uint32(raw[rawOff+2:])
	qs := unsafe.Slice((*byte)(unsafe.Pointer(&raw[rawOff+6])), 16)

	var sum float64
	for j := 0; j < 16; j++ {
		xh0 := byte((qh>>(j+0))<<4) & 0x10
		xh1 := byte(qh>>(j+12)) & 0x10

		v0 := float64(int32((qs[j]&0x0F)|xh0)) - 16
		v1 := float64(int32((qs[j]>>4)|xh1)) - 16

		sum += v0*xData[xBase+j] + v1*xData[xBase+j+16]
	}
	return s * sum
}

func q5DotGroup(raw []byte, rawOff int, x []float64, xOff int) float64 {
	s := readF16(raw, rawOff)
	qh := binary.LittleEndian.Uint32(raw[rawOff+2:])
	qs := unsafe.Slice((*byte)(unsafe.Pointer(&raw[rawOff+6])), 16)

	var sum float64
	for j := 0; j < 16; j++ {
		xh0 := byte((qh>>(j+0))<<4) & 0x10
		xh1 := byte(qh>>(j+12)) & 0x10

		v0 := float64(int32((qs[j]&0x0F)|xh0)) - 16
		v1 := float64(int32((qs[j]>>4)|xh1)) - 16

		sum += v0*x[xOff+j] + v1*x[xOff+j+16]
	}
	return s * sum
}

func fnLinearQ5Fast(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 3); err != nil {
		return err
	}
	raw, ok := args[1].(*object.String)
	if !ok {
		return errors.NewTypeError("STRING", args[1].Type().String())
	}
	gpr, err := args[2].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[2].Type().String())
	}
	if gpr <= 0 {
		return errors.NewError("linear_q5: groups_per_row must be positive")
	}
	groupsPerRow := int(gpr)

	rawBytes := []byte(raw.Value)
	rowBytes := groupsPerRow * 22
	outFeatures := len(rawBytes) / rowBytes
	inFeatures := groupsPerRow * 32

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_q5: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_q5: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		data := make([]float64, 0, xRows*outFeatures)
		for xi := 0; xi < xRows; xi++ {
			xRowOff := xi * xCols
			for j := 0; j < outFeatures; j++ {
				wRawOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q5DotGroupX(rawBytes, wRawOff+g*22, xData, xRowOff+g*32)
				}
				data = append(data, sum)
			}
		}
		return object.NewFloatArray2D(data, xRows, outFeatures)
	}

	xMat, errObj := toFloatMatrix(args[0], "linear_q5", "x")
	if errObj != nil {
		return errObj
	}
	seqLen := len(xMat)
	if seqLen == 0 || outFeatures == 0 {
		return errors.NewError("linear_q5: inputs cannot be empty")
	}

	data := make([]float64, 0, seqLen*outFeatures)
	for xi := 0; xi < seqLen; xi++ {
		xRow := xMat[xi]
		for j := 0; j < outFeatures; j++ {
			wRawOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				sum += q5DotGroup(rawBytes, wRawOff+g*22, xRow, g*32)
			}
			data = append(data, sum)
		}
	}
	return object.NewFloatArray2D(data, seqLen, outFeatures)
}

func fnLinearRowQ5Fast(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 3); err != nil {
		return err
	}
	raw, ok := args[1].(*object.String)
	if !ok {
		return errors.NewTypeError("STRING", args[1].Type().String())
	}
	gpr, err := args[2].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[2].Type().String())
	}
	if gpr <= 0 {
		return errors.NewError("linear_row_q5: groups_per_row must be positive")
	}
	groupsPerRow := int(gpr)

	rawBytes := []byte(raw.Value)
	rowBytes := groupsPerRow * 22
	outFeatures := len(rawBytes) / rowBytes
	inFeatures := groupsPerRow * 32

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_row_q5: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_row_q5: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		lastOff := (xRows - 1) * xCols
		result := make([]float64, outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				wRawOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q5DotGroupX(rawBytes, wRawOff+g*22, xData, lastOff+g*32)
				}
				result[j] = sum
			}
		})
		return object.NewFloatArray1D(result)
	}

	xMat, errObj := toFloatMatrix(args[0], "linear_row_q5", "x")
	if errObj != nil {
		return errObj
	}
	if len(xMat) == 0 || outFeatures == 0 {
		return errors.NewError("linear_row_q5: inputs cannot be empty")
	}
	lastRow := xMat[len(xMat)-1]
	if len(lastRow) != inFeatures {
		return errors.NewError("linear_row_q5: x columns (%d) must match in_features (%d)", len(lastRow), inFeatures)
	}

	result := make([]float64, outFeatures)
	parallelFor(outFeatures, func(start, end int) {
		for j := start; j < end; j++ {
			wRawOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				sum += q5DotGroup(rawBytes, wRawOff+g*22, lastRow, g*32)
			}
			result[j] = sum
		}
	})
	return object.NewFloatArray1D(result)
}

func fnDequantizeQ5_0(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	raw, ok := args[0].(*object.String)
	if !ok {
		return errors.NewTypeError("STRING", args[0].Type().String())
	}
	nGroups64, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	nGroups := int(nGroups64)

	rawBytes := []byte(raw.Value)
	result := make([]float64, nGroups*32)

	for g := 0; g < nGroups; g++ {
		off := g * 22
		d := readF16(rawBytes, off)
		qh := binary.LittleEndian.Uint32(rawBytes[off+2:])
		qsOff := off + 6
		outOff := g * 32

		for j := 0; j < 16; j++ {
			xh0 := byte((qh>>(j+0))<<4) & 0x10
			xh1 := byte(qh>>(j+12)) & 0x10
			x0 := float64(int32((rawBytes[qsOff+j]&0x0F)|xh0)) - 16
			x1 := float64(int32((rawBytes[qsOff+j]>>4)|xh1)) - 16
			result[outOff+j] = x0 * d
			result[outOff+j+16] = x1 * d
		}
	}

	return object.NewFloatArray1D(result)
}
