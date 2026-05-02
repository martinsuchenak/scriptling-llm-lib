package scriptlingllmlib

import (
	"context"
	"encoding/binary"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

// Q6_K block: 210 bytes per 256 elements
// Layout: ql(128) + qh(64) + scales(16) + d(2) = 210
// 16 sub-blocks of 16 values each with int8 scale
// 6-bit signed values: lower 4 bits in ql, upper 2 bits in qh, offset by -32

const q6kBlockSize = 210

func dequantizeQ6KBlock(raw []byte, off int, out []float64, outOff int) {
	d := float16ToFloat64(binary.LittleEndian.Uint16(raw[off+208:]))
	qlOff := off
	qhOff := off + 128
	scalesOff := off + 192

	for n := 0; n < 256; n += 128 {
		for l := 0; l < 32; l++ {
			is := l / 16

			q1 := int(int8((raw[qlOff+l]&0xF)|((raw[qhOff+l]>>0)&3)<<4)) - 32
			q2 := int(int8((raw[qlOff+l+32]&0xF)|((raw[qhOff+l]>>2)&3)<<4)) - 32
			q3 := int(int8((raw[qlOff+l]>>4)|((raw[qhOff+l]>>4)&3)<<4)) - 32
			q4 := int(int8((raw[qlOff+l+32]>>4)|((raw[qhOff+l]>>6)&3)<<4)) - 32

			sc0 := float64(int8(raw[scalesOff+is+0]))
			sc2 := float64(int8(raw[scalesOff+is+2]))
			sc4 := float64(int8(raw[scalesOff+is+4]))
			sc6 := float64(int8(raw[scalesOff+is+6]))

			out[outOff+l+0] = d * sc0 * float64(q1)
			out[outOff+l+32] = d * sc2 * float64(q2)
			out[outOff+l+64] = d * sc4 * float64(q3)
			out[outOff+l+96] = d * sc6 * float64(q4)
		}
		outOff += 128
		qlOff += 64
		qhOff += 32
		scalesOff += 8
	}
}

func q6kDotBlock(raw []byte, blkOff int, x []float64, xOff int) float64 {
	d := float16ToFloat64(binary.LittleEndian.Uint16(raw[blkOff+208:]))
	qlOff := blkOff
	qhOff := blkOff + 128
	scalesOff := blkOff + 192

	var sum float64

	for n := 0; n < 256; n += 128 {
		for l := 0; l < 32; l++ {
			is := l / 16

			q1 := float64(int(int8((raw[qlOff+l]&0xF)|((raw[qhOff+l]>>0)&3)<<4)) - 32)
			q2 := float64(int(int8((raw[qlOff+l+32]&0xF)|((raw[qhOff+l]>>2)&3)<<4)) - 32)
			q3 := float64(int(int8((raw[qlOff+l]>>4)|((raw[qhOff+l]>>4)&3)<<4)) - 32)
			q4 := float64(int(int8((raw[qlOff+l+32]>>4)|((raw[qhOff+l]>>6)&3)<<4)) - 32)

			sc0 := float64(int8(raw[scalesOff+is+0]))
			sc2 := float64(int8(raw[scalesOff+is+2]))
			sc4 := float64(int8(raw[scalesOff+is+4]))
			sc6 := float64(int8(raw[scalesOff+is+6]))

			sum += d * sc0 * q1 * x[xOff+l+0]
			sum += d * sc2 * q2 * x[xOff+l+32]
			sum += d * sc4 * q3 * x[xOff+l+64]
			sum += d * sc6 * q4 * x[xOff+l+96]
		}
		xOff += 128
		qlOff += 64
		qhOff += 32
		scalesOff += 8
	}

	return sum
}

func fnLinearQ6K(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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

	rawBytes := []byte(raw.Value)
	elementsPerBlock := 256
	groupsPerRow := int(gpr)
	rowBytes := groupsPerRow * q6kBlockSize
	outFeatures := len(rawBytes) / rowBytes
	inFeatures := groupsPerRow * elementsPerBlock

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_q6_k: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_q6_k: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		data := make([]float64, 0, xRows*outFeatures)
		for xi := 0; xi < xRows; xi++ {
			xRowOff := xi * xCols
			for j := 0; j < outFeatures; j++ {
				rowRawOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					blkOff := rowRawOff + g*q6kBlockSize
					xOff := xRowOff + g*elementsPerBlock
					sum += q6kDotBlock(rawBytes, blkOff, xData, xOff)
				}
				data = append(data, sum)
			}
		}
		return object.NewFloatArray2D(data, xRows, outFeatures)
	}

	xMat, errObj := toFloatMatrix(args[0], "linear_q6_k", "x")
	if errObj != nil {
		return errObj
	}
	seqLen := len(xMat)
	if seqLen == 0 || outFeatures == 0 {
		return errors.NewError("linear_q6_k: inputs cannot be empty")
	}

	data := make([]float64, 0, seqLen*outFeatures)
	for xi := 0; xi < seqLen; xi++ {
		xRow := xMat[xi]
		for j := 0; j < outFeatures; j++ {
			rowRawOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				blkOff := rowRawOff + g*q6kBlockSize
				xOff := g * elementsPerBlock
				sum += q6kDotBlock(rawBytes, blkOff, xRow, xOff)
			}
			data = append(data, sum)
		}
	}
	return object.NewFloatArray2D(data, seqLen, outFeatures)
}

func fnLinearRowQ6K(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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

	rawBytes := []byte(raw.Value)
	elementsPerBlock := 256
	groupsPerRow := int(gpr)
	rowBytes := groupsPerRow * q6kBlockSize
	outFeatures := len(rawBytes) / rowBytes
	inFeatures := groupsPerRow * elementsPerBlock

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_row_q6_k: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_row_q6_k: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		lastOff := (xRows - 1) * xCols
		result := make([]float64, outFeatures)
		for j := 0; j < outFeatures; j++ {
			rowRawOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				blkOff := rowRawOff + g*q6kBlockSize
				xOff := lastOff + g*elementsPerBlock
				sum += q6kDotBlock(rawBytes, blkOff, xData, xOff)
			}
			result[j] = sum
		}
		return object.NewFloatArray1D(result)
	}

	xMat, errObj := toFloatMatrix(args[0], "linear_row_q6_k", "x")
	if errObj != nil {
		return errObj
	}
	if len(xMat) == 0 || outFeatures == 0 {
		return errors.NewError("linear_row_q6_k: inputs cannot be empty")
	}
	lastRow := xMat[len(xMat)-1]
	if len(lastRow) != inFeatures {
		return errors.NewError("linear_row_q6_k: x columns (%d) must match in_features (%d)", len(lastRow), inFeatures)
	}

	result := make([]float64, outFeatures)
	for j := 0; j < outFeatures; j++ {
		rowRawOff := j * rowBytes
		var sum float64
		for g := 0; g < groupsPerRow; g++ {
			blkOff := rowRawOff + g*q6kBlockSize
			xOff := g * elementsPerBlock
			sum += q6kDotBlock(rawBytes, blkOff, lastRow, xOff)
		}
		result[j] = sum
	}
	return object.NewFloatArray1D(result)
}

func fnDequantizeQ6K(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}

	raw, ok := args[0].(*object.String)
	if !ok {
		return errors.NewTypeError("STRING", args[0].Type().String())
	}
	nBlocks64, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	nBlocks := int(nBlocks64)

	rawBytes := []byte(raw.Value)
	result := make([]float64, nBlocks*256)

	for b := 0; b < nBlocks; b++ {
		off := b * q6kBlockSize
		dequantizeQ6KBlock(rawBytes, off, result, b*256)
	}

	return object.NewFloatArray1D(result)
}
