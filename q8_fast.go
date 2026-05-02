package scriptlingllmlib

import (
	"context"
	"encoding/binary"
	"unsafe"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func init() {
	if unsafe.Sizeof(int(0)) != 8 {
		panic("q8_fast requires 64-bit platform")
	}
}

func readF16(b []byte, off int) float64 {
	return float16ToFloat64(binary.LittleEndian.Uint16(b[off : off+2]))
}

func q8DotGroupX(raw []byte, rawOff int, xData []float64, xBase int) float64 {
	s := readF16(raw, rawOff)
	q := unsafe.Slice((*byte)(unsafe.Pointer(&raw[rawOff+2])), 32)
	qw := unsafe.Slice((*uint64)(unsafe.Pointer(&q[0])), 4)

	sum := 0.0
	for w := 0; w < 4; w++ {
		chunk := qw[w]
		b0 := float64(int8(chunk))
		b1 := float64(int8(chunk >> 8))
		b2 := float64(int8(chunk >> 16))
		b3 := float64(int8(chunk >> 24))
		b4 := float64(int8(chunk >> 32))
		b5 := float64(int8(chunk >> 40))
		b6 := float64(int8(chunk >> 48))
		b7 := float64(int8(chunk >> 56))
		i := xBase + w*8
		sum += s * (b0*xData[i] + b1*xData[i+1] + b2*xData[i+2] + b3*xData[i+3] +
			b4*xData[i+4] + b5*xData[i+5] + b6*xData[i+6] + b7*xData[i+7])
	}
	return sum
}

func q8DotGroup(raw []byte, rawOff int, x []float64, xOff int) float64 {
	s := readF16(raw, rawOff)
	q := unsafe.Slice((*byte)(unsafe.Pointer(&raw[rawOff+2])), 32)
	qw := unsafe.Slice((*uint64)(unsafe.Pointer(&q[0])), 4)

	sum := 0.0
	for w := 0; w < 4; w++ {
		chunk := qw[w]
		b0 := float64(int8(chunk))
		b1 := float64(int8(chunk >> 8))
		b2 := float64(int8(chunk >> 16))
		b3 := float64(int8(chunk >> 24))
		b4 := float64(int8(chunk >> 32))
		b5 := float64(int8(chunk >> 40))
		b6 := float64(int8(chunk >> 48))
		b7 := float64(int8(chunk >> 56))
		i := w * 8
		sum += s * (b0*x[xOff+i] + b1*x[xOff+i+1] + b2*x[xOff+i+2] + b3*x[xOff+i+3] +
			b4*x[xOff+i+4] + b5*x[xOff+i+5] + b6*x[xOff+i+6] + b7*x[xOff+i+7])
	}
	return sum
}

func q4DotGroupX(raw []byte, rawOff int, xData []float64, xBase int) float64 {
	s := readF16(raw, rawOff)
	qw := unsafe.Slice((*uint64)(unsafe.Pointer(&raw[rawOff+2])), 2)
	var sum float64
	for w := 0; w < 2; w++ {
		chunk := qw[w]
		b0 := float64(int8(chunk&0x0F) - 8)
		b1 := float64(int8((chunk>>4)&0x0F) - 8)
		b2 := float64(int8((chunk>>8)&0x0F) - 8)
		b3 := float64(int8((chunk>>12)&0x0F) - 8)
		b4 := float64(int8((chunk>>16)&0x0F) - 8)
		b5 := float64(int8((chunk>>20)&0x0F) - 8)
		b6 := float64(int8((chunk>>24)&0x0F) - 8)
		b7 := float64(int8((chunk>>28)&0x0F) - 8)
		b8 := float64(int8((chunk>>32)&0x0F) - 8)
		b9 := float64(int8((chunk>>36)&0x0F) - 8)
		b10 := float64(int8((chunk>>40)&0x0F) - 8)
		b11 := float64(int8((chunk>>44)&0x0F) - 8)
		b12 := float64(int8((chunk>>48)&0x0F) - 8)
		b13 := float64(int8((chunk>>52)&0x0F) - 8)
		b14 := float64(int8((chunk>>56)&0x0F) - 8)
		b15 := float64(int8((chunk>>60)&0x0F) - 8)
		i := xBase + w*16
		sum += b0*xData[i] + b1*xData[i+1] + b2*xData[i+2] + b3*xData[i+3] +
			b4*xData[i+4] + b5*xData[i+5] + b6*xData[i+6] + b7*xData[i+7] +
			b8*xData[i+8] + b9*xData[i+9] + b10*xData[i+10] + b11*xData[i+11] +
			b12*xData[i+12] + b13*xData[i+13] + b14*xData[i+14] + b15*xData[i+15]
	}
	return s * sum
}

func q4_1DotGroupX(raw []byte, rawOff int, xData []float64, xBase int) float64 {
	d := readF16(raw, rawOff)
	m := readF16(raw, rawOff+2)
	qw := unsafe.Slice((*uint64)(unsafe.Pointer(&raw[rawOff+4])), 2)
	var qSum, xSum float64
	for w := 0; w < 2; w++ {
		chunk := qw[w]
		b0 := float64(chunk & 0x0F)
		b1 := float64((chunk >> 4) & 0x0F)
		b2 := float64((chunk >> 8) & 0x0F)
		b3 := float64((chunk >> 12) & 0x0F)
		b4 := float64((chunk >> 16) & 0x0F)
		b5 := float64((chunk >> 20) & 0x0F)
		b6 := float64((chunk >> 24) & 0x0F)
		b7 := float64((chunk >> 28) & 0x0F)
		b8 := float64((chunk >> 32) & 0x0F)
		b9 := float64((chunk >> 36) & 0x0F)
		b10 := float64((chunk >> 40) & 0x0F)
		b11 := float64((chunk >> 44) & 0x0F)
		b12 := float64((chunk >> 48) & 0x0F)
		b13 := float64((chunk >> 52) & 0x0F)
		b14 := float64((chunk >> 56) & 0x0F)
		b15 := float64((chunk >> 60) & 0x0F)
		i := xBase + w*16
		qSum += b0*xData[i] + b1*xData[i+1] + b2*xData[i+2] + b3*xData[i+3] +
			b4*xData[i+4] + b5*xData[i+5] + b6*xData[i+6] + b7*xData[i+7] +
			b8*xData[i+8] + b9*xData[i+9] + b10*xData[i+10] + b11*xData[i+11] +
			b12*xData[i+12] + b13*xData[i+13] + b14*xData[i+14] + b15*xData[i+15]
		xSum += xData[i] + xData[i+1] + xData[i+2] + xData[i+3] +
			xData[i+4] + xData[i+5] + xData[i+6] + xData[i+7] +
			xData[i+8] + xData[i+9] + xData[i+10] + xData[i+11] +
			xData[i+12] + xData[i+13] + xData[i+14] + xData[i+15]
	}
	return d*qSum + m*xSum
}

func q4DotGroup(raw []byte, rawOff int, x []float64, xOff int) float64 {
	s := readF16(raw, rawOff)
	qw := unsafe.Slice((*uint64)(unsafe.Pointer(&raw[rawOff+2])), 2)
	var sum float64
	for w := 0; w < 2; w++ {
		chunk := qw[w]
		b0 := float64(int8(chunk&0x0F) - 8)
		b1 := float64(int8((chunk>>4)&0x0F) - 8)
		b2 := float64(int8((chunk>>8)&0x0F) - 8)
		b3 := float64(int8((chunk>>12)&0x0F) - 8)
		b4 := float64(int8((chunk>>16)&0x0F) - 8)
		b5 := float64(int8((chunk>>20)&0x0F) - 8)
		b6 := float64(int8((chunk>>24)&0x0F) - 8)
		b7 := float64(int8((chunk>>28)&0x0F) - 8)
		b8 := float64(int8((chunk>>32)&0x0F) - 8)
		b9 := float64(int8((chunk>>36)&0x0F) - 8)
		b10 := float64(int8((chunk>>40)&0x0F) - 8)
		b11 := float64(int8((chunk>>44)&0x0F) - 8)
		b12 := float64(int8((chunk>>48)&0x0F) - 8)
		b13 := float64(int8((chunk>>52)&0x0F) - 8)
		b14 := float64(int8((chunk>>56)&0x0F) - 8)
		b15 := float64(int8((chunk>>60)&0x0F) - 8)
		i := xOff + w*16
		sum += b0*x[i] + b1*x[i+1] + b2*x[i+2] + b3*x[i+3] +
			b4*x[i+4] + b5*x[i+5] + b6*x[i+6] + b7*x[i+7] +
			b8*x[i+8] + b9*x[i+9] + b10*x[i+10] + b11*x[i+11] +
			b12*x[i+12] + b13*x[i+13] + b14*x[i+14] + b15*x[i+15]
	}
	return s * sum
}

func fnLinearQ8Fast(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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
		return errors.NewError("linear_q8: groups_per_row must be positive")
	}
	groupsPerRow := int(gpr)

	rawBytes := []byte(raw.Value)
	rowBytes := groupsPerRow * 34
	outFeatures := len(rawBytes) / rowBytes
	inFeatures := groupsPerRow * 32

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_q8: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_q8: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		data := make([]float64, 0, xRows*outFeatures)
		for xi := 0; xi < xRows; xi++ {
			xRowOff := xi * xCols
			for j := 0; j < outFeatures; j++ {
				rowRawOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q8DotGroupX(rawBytes, rowRawOff+g*34, xData, xRowOff+g*32)
				}
				data = append(data, sum)
			}
		}
		return object.NewFloatArray2D(data, xRows, outFeatures)
	}

	xMat, errObj := toFloatMatrix(args[0], "linear_q8", "x")
	if errObj != nil {
		return errObj
	}
	seqLen := len(xMat)
	if seqLen == 0 || outFeatures == 0 {
		return errors.NewError("linear_q8: inputs cannot be empty")
	}

	data := make([]float64, 0, seqLen*outFeatures)
	for xi := 0; xi < seqLen; xi++ {
		xRow := xMat[xi]
		for j := 0; j < outFeatures; j++ {
			rowRawOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				sum += q8DotGroup(rawBytes, rowRawOff+g*34, xRow, g*32)
			}
			data = append(data, sum)
		}
	}
	return object.NewFloatArray2D(data, seqLen, outFeatures)
}

func fnLinearRowQ8Fast(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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
		return errors.NewError("linear_row_q8: groups_per_row must be positive")
	}
	groupsPerRow := int(gpr)

	rawBytes := []byte(raw.Value)
	rowBytes := groupsPerRow * 34
	outFeatures := len(rawBytes) / rowBytes
	inFeatures := groupsPerRow * 32

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_row_q8: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_row_q8: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		lastOff := (xRows - 1) * xCols
		result := make([]float64, outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rowRawOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q8DotGroupX(rawBytes, rowRawOff+g*34, xData, lastOff+g*32)
				}
				result[j] = sum
			}
		})
		return object.NewFloatArray1D(result)
	}

	xMat, errObj := toFloatMatrix(args[0], "linear_row_q8", "x")
	if errObj != nil {
		return errObj
	}
	if len(xMat) == 0 || outFeatures == 0 {
		return errors.NewError("linear_row_q8: inputs cannot be empty")
	}
	lastRow := xMat[len(xMat)-1]
	if len(lastRow) != inFeatures {
		return errors.NewError("linear_row_q8: x columns (%d) must match in_features (%d)", len(lastRow), inFeatures)
	}

	result := make([]float64, outFeatures)
	parallelFor(outFeatures, func(start, end int) {
		for j := start; j < end; j++ {
			rowRawOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				sum += q8DotGroup(rawBytes, rowRawOff+g*34, lastRow, g*32)
			}
			result[j] = sum
		}
	})
	return object.NewFloatArray1D(result)
}

func fnLinearQ4Fast(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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
		return errors.NewError("linear_q4: groups_per_row must be positive")
	}
	groupsPerRow := int(gpr)

	rawBytes := []byte(raw.Value)
	rowBytes := groupsPerRow * 18
	outFeatures := len(rawBytes) / rowBytes
	inFeatures := groupsPerRow * 32

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_q4: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_q4: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		data := make([]float64, 0, xRows*outFeatures)
		for xi := 0; xi < xRows; xi++ {
			xRowOff := xi * xCols
			for j := 0; j < outFeatures; j++ {
				wRawOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q4DotGroupX(rawBytes, wRawOff+g*18, xData, xRowOff+g*32)
				}
				data = append(data, sum)
			}
		}
		return object.NewFloatArray2D(data, xRows, outFeatures)
	}

	xMat, errObj := toFloatMatrix(args[0], "linear_q4", "x")
	if errObj != nil {
		return errObj
	}
	seqLen := len(xMat)
	if seqLen == 0 || outFeatures == 0 {
		return errors.NewError("linear_q4: inputs cannot be empty")
	}

	data := make([]float64, 0, seqLen*outFeatures)
	for xi := 0; xi < seqLen; xi++ {
		xRow := xMat[xi]
		for j := 0; j < outFeatures; j++ {
			wRawOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				sum += q4DotGroup(rawBytes, wRawOff+g*18, xRow, g*32)
			}
			data = append(data, sum)
		}
	}
	return object.NewFloatArray2D(data, seqLen, outFeatures)
}

func fnLinearRowQ4Fast(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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
		return errors.NewError("linear_row_q4: groups_per_row must be positive")
	}
	groupsPerRow := int(gpr)

	rawBytes := []byte(raw.Value)
	rowBytes := groupsPerRow * 18
	outFeatures := len(rawBytes) / rowBytes
	inFeatures := groupsPerRow * 32

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_row_q4: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_row_q4: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		lastOff := (xRows - 1) * xCols
		result := make([]float64, outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				wRawOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q4DotGroupX(rawBytes, wRawOff+g*18, xData, lastOff+g*32)
				}
				result[j] = sum
			}
		})
		return object.NewFloatArray1D(result)
	}

	xMat, errObj := toFloatMatrix(args[0], "linear_row_q4", "x")
	if errObj != nil {
		return errObj
	}
	if len(xMat) == 0 || outFeatures == 0 {
		return errors.NewError("linear_row_q4: inputs cannot be empty")
	}
	lastRow := xMat[len(xMat)-1]
	if len(lastRow) != inFeatures {
		return errors.NewError("linear_row_q4: x columns (%d) must match in_features (%d)", len(lastRow), inFeatures)
	}

	result := make([]float64, outFeatures)
	parallelFor(outFeatures, func(start, end int) {
		for j := start; j < end; j++ {
			wRawOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				sum += q4DotGroup(rawBytes, wRawOff+g*18, lastRow, g*32)
			}
			result[j] = sum
		}
	})
	return object.NewFloatArray1D(result)
}
