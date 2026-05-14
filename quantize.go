package scriptlingllmlib

import (
	"context"
	"encoding/binary"
	"math"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func fnQuantizeQ8(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 3); err != nil {
		return err
	}
	flatData, errObj := toFloatList(args[0], "quantize_q8", "data")
	if errObj != nil {
		return errObj
	}
	rows, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	cols, err := args[2].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[2].Type().String())
	}
	nRows := int(rows)
	nCols := int(cols)
	if len(flatData) != nRows*nCols {
		return errors.NewError("quantize_q8: data length (%d) must equal rows*cols (%d)", len(flatData), nRows*nCols)
	}
	if nCols%32 != 0 {
		return errors.NewError("quantize_q8: cols (%d) must be divisible by 32", nCols)
	}
	groupsPerRow := nCols / 32
	totalGroups := nRows * groupsPerRow
	result := make([]byte, 0, totalGroups*34)

	for r := 0; r < nRows; r++ {
		rowOff := r * nCols
		for g := 0; g < groupsPerRow; g++ {
			base := rowOff + g*32
			var maxAbs float64
			for i := 0; i < 32; i++ {
				v := math.Abs(flatData[base+i])
				if v > maxAbs {
					maxAbs = v
				}
			}
			var scale float64
			if maxAbs > 0 {
				scale = maxAbs / 127.0
			}
			scaleBits := float64ToFloat16(scale)
			scaleBytes := make([]byte, 2)
			binary.LittleEndian.PutUint16(scaleBytes, scaleBits)
			result = append(result, scaleBytes...)
			invScale := 1.0 / scale
			for i := 0; i < 32; i++ {
				q := int8(flatData[base+i] * invScale)
				if flatData[base+i]*invScale > 127 {
					q = 127
				} else if flatData[base+i]*invScale < -128 {
					q = -128
				}
				result = append(result, byte(q))
			}
		}
	}

	return object.NewString(string(result))
}

func fnQuantizeQ8Rows(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	matrix, errObj := toFloatMatrix(args[0], "quantize_q8_rows", "data")
	if errObj != nil {
		return errObj
	}
	nCols64, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	nCols := int(nCols64)
	nRows := len(matrix)
	if nRows == 0 {
		return errors.NewError("quantize_q8_rows: data cannot be empty")
	}
	if nCols%32 != 0 {
		return errors.NewError("quantize_q8_rows: cols (%d) must be divisible by 32", nCols)
	}
	groupsPerRow := nCols / 32
	totalGroups := nRows * groupsPerRow
	result := make([]byte, 0, totalGroups*34)
	scaleBytes := make([]byte, 2)

	for r := 0; r < nRows; r++ {
		row := matrix[r]
		for g := 0; g < groupsPerRow; g++ {
			base := g * 32
			var maxAbs float64
			for i := 0; i < 32; i++ {
				v := math.Abs(row[base+i])
				if v > maxAbs {
					maxAbs = v
				}
			}
			var scale float64
			if maxAbs > 0 {
				scale = maxAbs / 127.0
			}
			binary.LittleEndian.PutUint16(scaleBytes, float64ToFloat16(scale))
			result = append(result, scaleBytes...)
			invScale := 1.0 / scale
			for i := 0; i < 32; i++ {
				q := int8(row[base+i] * invScale)
				if row[base+i]*invScale > 127 {
					q = 127
				} else if row[base+i]*invScale < -128 {
					q = -128
				}
				result = append(result, byte(q))
			}
		}
	}

	return object.NewString(string(result))
}

func float64ToFloat16(f float64) uint16 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0
	}
	bits := math.Float64bits(float64(f))
	sign := uint16((bits >> 48) & 0x8000)
	exp64 := (bits >> 52) & 0x7FF
	mant64 := bits & 0x000FFFFFFFFFFFFF

	if exp64 == 0 {
		return sign
	}
	if exp64 == 0x7FF {
		return sign | 0x7C00
	}

	exp16 := int16(exp64) - 1023 + 15
	if exp16 <= 0 {
		return sign
	}
	if exp16 >= 0x1F {
		return sign | 0x7C00
	}

	mant16 := uint16((mant64 >> 42) & 0x3FF)
	return sign | (uint16(exp16) << 10) | mant16
}
