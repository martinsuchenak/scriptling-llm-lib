package scriptlingllmlib

import (
	"context"
	"encoding/binary"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

const (
	QK_K = 256
)

// getScaleMinK4 unpacks the 6-bit scale and min for sub-block j from the 12-byte scales array.
// For j < 4: scale = scales[j] & 63, min = scales[j+4] & 63
// For j >= 4: upper 2 bits packed into top bits of earlier bytes
func getScaleMinK4(j int, scales []byte) (uint8, uint8) {
	if j < 4 {
		sc := scales[j] & 63
		m := scales[j+4] & 63
		return sc, m
	}
	sc := (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
	m := (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
	return sc, m
}

func fnLinearQ4K(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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
	blockSize := 144
	elementsPerBlock := 256
	groupsPerRow := int(gpr)
	rowBytes := groupsPerRow * blockSize
	outFeatures := len(rawBytes) / rowBytes
	inFeatures := groupsPerRow * elementsPerBlock

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_q4_k: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_q4_k: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		total := xRows * outFeatures
		dataSlice := make([]float64, total)
		parallelFor(total, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				xRowOff := xi * xCols
				rowRawOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					blkOff := rowRawOff + g*blockSize
					xOff := xRowOff + g*elementsPerBlock
					sum += q4kDotBlockFast(rawBytes, blkOff, xData, xOff)
				}
				dataSlice[xi*outFeatures+j] = sum
			}
		})
		return object.NewFloatArray2D(dataSlice, xRows, outFeatures)
	}

	xMat, errObj := toFloatMatrix(args[0], "linear_q4_k", "x")
	if errObj != nil {
		return errObj
	}
	seqLen := len(xMat)
	if seqLen == 0 || outFeatures == 0 {
		return errors.NewError("linear_q4_k: inputs cannot be empty")
	}

	data := make([]float64, 0, seqLen*outFeatures)
	for xi := 0; xi < seqLen; xi++ {
		xRow := xMat[xi]
		for j := 0; j < outFeatures; j++ {
			rowRawOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				blkOff := rowRawOff + g*blockSize
				xOff := g * elementsPerBlock
				sum += q4kDotBlockFast(rawBytes, blkOff, xRow, xOff)
			}
			data = append(data, sum)
		}
	}
	return object.NewFloatArray2D(data, seqLen, outFeatures)
}

func fnLinearRowQ4K(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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
	blockSize := 144
	elementsPerBlock := 256
	groupsPerRow := int(gpr)
	rowBytes := groupsPerRow * blockSize
	outFeatures := len(rawBytes) / rowBytes
	inFeatures := groupsPerRow * elementsPerBlock

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_row_q4_k: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_row_q4_k: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		lastOff := (xRows - 1) * xCols
		result := make([]float64, outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rowRawOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					blkOff := rowRawOff + g*blockSize
					xOff := lastOff + g*elementsPerBlock
					sum += q4kDotBlockFast(rawBytes, blkOff, xData, xOff)
				}
				result[j] = sum
			}
		})
		return object.NewFloatArray1D(result)
	}

	xMat, errObj := toFloatMatrix(args[0], "linear_row_q4_k", "x")
	if errObj != nil {
		return errObj
	}
	if len(xMat) == 0 || outFeatures == 0 {
		return errors.NewError("linear_row_q4_k: inputs cannot be empty")
	}
	lastRow := xMat[len(xMat)-1]
	if len(lastRow) != inFeatures {
		return errors.NewError("linear_row_q4_k: x columns (%d) must match in_features (%d)", len(lastRow), inFeatures)
	}

	result := make([]float64, outFeatures)
	parallelFor(outFeatures, func(start, end int) {
		for j := start; j < end; j++ {
			rowRawOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				blkOff := rowRawOff + g*blockSize
				xOff := g * elementsPerBlock
				sum += q4kDotBlockFast(rawBytes, blkOff, lastRow, xOff)
			}
			result[j] = sum
		}
	})
	return object.NewFloatArray1D(result)
}

// q4kDotBlock computes dot product of a Q4_K block with a float64 vector.
// Block layout (144 bytes, 256 elements):
//
//	[0..1]   d     : f16 super-block scale
//	[2..3]   dmin  : f16 super-block minimum
//	[4..15]  scales: 12 bytes packed 6-bit scales+mins for 8 sub-blocks
//	[16..143] qs   : 128 bytes = 256 nibbles (4-bit quantized values)
//
// Dequant formula per value: (d * scale_6bit) * quant_4bit - (dmin * min_6bit)
// Processing: 4 groups of 64, each group uses 2 sub-blocks (is, is+1):
//   - 32 low nibbles with (d*sc0) scale and (dmin*m0) min
//   - 32 high nibbles with (d*sc1) scale and (dmin*m1) min
func q4kDotBlock(raw []byte, blkOff int, x []float64, xOff int) float64 {
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
		m1val := dmin * float64(m0)
		d2 := d * float64(sc1)
		m2val := dmin * float64(m1)

		qBase := qsOff + qPos
		xBase := xOff + group*64

		for l := 0; l < 32; l++ {
			q := float64(raw[qBase+l] & 0xF)
			sum += (d1*q - m1val) * x[xBase+l]
		}
		for l := 0; l < 32; l++ {
			q := float64(raw[qBase+l] >> 4)
			sum += (d2*q - m2val) * x[xBase+32+l]
		}

		qPos += 32
		is += 2
	}

	return sum
}

func q4kDotBlockSlow(raw []byte, blkOff int, x []float64, xOff int) float64 {
	return q4kDotBlockFast(raw, blkOff, x, xOff)
}

func fnDequantizeQ4K(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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
		off := b * 144
		d := float16ToFloat64(binary.LittleEndian.Uint16(rawBytes[off:]))
		dmin := float16ToFloat64(binary.LittleEndian.Uint16(rawBytes[off+2:]))
		scalesOff := off + 4
		qsOff := off + 16
		outOff := b * 256

		is := 0
		qPos := 0
		for group := 0; group < 4; group++ {
			sc0, m0 := getScaleMinK4(is, rawBytes[scalesOff:])
			sc1, m1 := getScaleMinK4(is+1, rawBytes[scalesOff:])
			d1 := d * float64(sc0)
			m1val := dmin * float64(m0)
			d2 := d * float64(sc1)
			m2val := dmin * float64(m1)

			qBase := qsOff + qPos
			for l := 0; l < 32; l++ {
				result[outOff+group*64+l] = d1*float64(rawBytes[qBase+l]&0xF) - m1val
			}
			for l := 0; l < 32; l++ {
				result[outOff+group*64+32+l] = d2*float64(rawBytes[qBase+l]>>4) - m2val
			}

			qPos += 32
			is += 2
		}
	}

	return object.NewFloatArray1D(result)
}
