package scriptlingllmlib

import (
	"context"
	"math"
	"sync"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func fnFusedQKV(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 4); err != nil {
		return err
	}

	xData, xRows, xCols, xOK := object.GetFloatMatrix(args[0])
	if !xOK {
		return errors.NewTypeError("FLOAT_MATRIX", args[0].Type().String())
	}
	if xRows == 0 {
		return errors.NewError("fused_qkv: x cannot be empty")
	}

	var qOut, kOut, vOut []float64
	var qRows, qCols, kRows, kCols, vRows, vCols int

	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		qOut, qRows, qCols = fusedMatmul(args[1], xData, xRows, xCols)
		wg.Done()
	}()
	go func() {
		kOut, kRows, kCols = fusedMatmul(args[2], xData, xRows, xCols)
		wg.Done()
	}()
	go func() {
		vOut, vRows, vCols = fusedMatmul(args[3], xData, xRows, xCols)
		wg.Done()
	}()
	wg.Wait()

	if qOut == nil || kOut == nil || vOut == nil {
		return errors.NewError("fused_qkv: projection failed")
	}

	return &object.List{Elements: []object.Object{
		object.NewFloatArray2D(qOut, qRows, qCols),
		object.NewFloatArray2D(kOut, kRows, kCols),
		object.NewFloatArray2D(vOut, vRows, vCols),
	}}
}

func fusedMatmul(w object.Object, xData []float64, xRows, xCols int) ([]float64, int, int) {
	if raw, ok := w.(*object.String); ok {
		return fusedMatmulQ8(xData, xRows, xCols, raw)
	}
	if d, ok := w.(*object.Dict); ok {
		if d.HasByString("q5") {
			return fusedMatmulQ5(xData, xRows, xCols, d)
		}
		if d.HasByString("q8") {
			pair, _ := d.GetByString("raw")
			if raw, ok := pair.Value.(*object.String); ok {
				return fusedMatmulQ8(xData, xRows, xCols, raw)
			}
		}
	}
	if wData, wRows, wCols, ok := object.GetFloatMatrix(w); ok {
		return fusedMatmulFloat(xData, xRows, xCols, wData, wRows, wCols)
	}
	return nil, 0, 0
}

func fusedMatmulQ5(xData []float64, xRows, xCols int, d *object.Dict) ([]float64, int, int) {
	rawPair, _ := d.GetByString("raw")
	raw, ok := rawPair.Value.(*object.String)
	if !ok {
		return nil, 0, 0
	}
	gprPair, _ := d.GetByString("groups_per_row")
	gprVal, err := gprPair.Value.AsInt()
	if err != nil {
		return nil, 0, 0
	}
	groupsPerRow := int(gprVal)

	rawBytes := []byte(raw.Value)
	rowBytes := groupsPerRow * 22
	outFeatures := len(rawBytes) / rowBytes

	result := make([]float64, xRows*outFeatures)
	parallelFor(xRows*outFeatures, func(start, end int) {
		for idx := start; idx < end; idx++ {
			xi := idx / outFeatures
			j := idx % outFeatures
			xOff := xi * xCols
			wRawOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				sum += q5DotGroupX(rawBytes, wRawOff+g*22, xData, xOff+g*32)
			}
			result[xi*outFeatures+j] = sum
		}
	})
	return result, xRows, outFeatures
}

func fusedMatmulQ8(xData []float64, xRows, xCols int, raw *object.String) ([]float64, int, int) {
	rawBytes := []byte(raw.Value)
	groupsPerRow := xCols / 32
	if groupsPerRow == 0 {
		return nil, 0, 0
	}
	rowBytes := groupsPerRow * 34
	if len(rawBytes)%rowBytes != 0 {
		return nil, 0, 0
	}
	outFeatures := len(rawBytes) / rowBytes
	result := make([]float64, xRows*outFeatures)
	parallelFor(xRows*outFeatures, func(start, end int) {
		for idx := start; idx < end; idx++ {
			xi := idx / outFeatures
			j := idx % outFeatures
			xOff := xi * xCols
			rOff := j * rowBytes
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				sum += q8DotGroupX(rawBytes, rOff+g*34, xData, xOff+g*32)
			}
			result[xi*outFeatures+j] = sum
		}
	})
	return result, xRows, outFeatures
}

func fusedMatmulFloat(xData []float64, xRows, xCols int, wData []float64, wRows, wCols int) ([]float64, int, int) {
	result := make([]float64, xRows*wRows)
	parallelFor(xRows*wRows, func(start, end int) {
		for idx := start; idx < end; idx++ {
			xi := idx / wRows
			j := idx % wRows
			xOff := xi * xCols
			wOff := j * wCols
			var sum float64
			for l := 0; l < xCols; l++ {
				sum += xData[xOff+l] * wData[wOff+l]
			}
			result[xi*wRows+j] = sum
		}
	})
	return result, xRows, wRows
}

func fnFusedFFN(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 4); err != nil {
		return err
	}

	xData, xRows, xCols, xOK := object.GetFloatMatrix(args[0])
	if !xOK {
		return errors.NewTypeError("FLOAT_MATRIX", args[0].Type().String())
	}
	if xRows == 0 {
		return errors.NewError("fused_ffn: x cannot be empty")
	}

	var gateData, upData []float64
	var gateRows, gateCols, upRows, upCols int

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		gateData, gateRows, gateCols = fusedMatmul(args[1], xData, xRows, xCols)
		wg.Done()
	}()
	go func() {
		upData, upRows, upCols = fusedMatmul(args[2], xData, xRows, xCols)
		wg.Done()
	}()
	wg.Wait()

	if gateData == nil || upData == nil {
		return errors.NewError("fused_ffn: projection failed")
	}

	hidden := gateRows * gateCols
	siluData := make([]float64, hidden)
	for i := 0; i < hidden; i++ {
		g := gateData[i]
		s := 1.0 / (1.0 + math.Exp(-g))
		siluData[i] = g * s * upData[i]
	}

	downData, downRows, downCols := fusedMatmul(args[3], siluData, gateRows, gateCols)
	if downData == nil {
		return errors.NewError("fused_ffn: down projection failed")
	}

	_ = upRows
	_ = upCols
	return object.NewFloatArray2D(downData, downRows, downCols)
}

func fnFusedRopeBatch(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}

	startPos, err := args[1].AsInt()
	if err != nil {
		return err
	}

	freqBase := kwargs.MustGetFloat("freq_base", 10000.0)
	ropeDim := int(kwargs.MustGetInt("rope_dim", 0))

	headList, ok := args[0].(*object.List)
	if !ok {
		return errors.NewTypeError("LIST", args[0].Type().String())
	}

	nHeads := len(headList.Elements)
	results := make([]object.Object, nHeads)

	halfDim := ropeDim / 2
	freqs := make([]float64, halfDim)
	for i := 0; i < halfDim; i++ {
		freqs[i] = 1.0 / math.Pow(freqBase, float64(2*i)/float64(ropeDim))
	}

	for h := 0; h < nHeads; h++ {
		hData, hRows, hCols, ok := object.GetFloatMatrix(headList.Elements[h])
		if !ok {
			return errors.NewError("fused_rope_batch: head %d is not a matrix", h)
		}

		out := make([]float64, hRows*hCols)
		for row := 0; row < hRows; row++ {
			pos := int(startPos) + row
			rOff := row * hCols
			for i := 0; i < halfDim; i++ {
				angle := freqs[i] * float64(pos)
				cosA := math.Cos(angle)
				sinA := math.Sin(angle)
				x0 := hData[rOff+2*i]
				x1 := hData[rOff+2*i+1]
				out[rOff+2*i] = x0*cosA - x1*sinA
				out[rOff+2*i+1] = x0*sinA + x1*cosA
			}
			if ropeDim > 0 && ropeDim < hCols {
				for d := ropeDim; d < hCols; d++ {
					out[rOff+d] = hData[rOff+d]
				}
			}
		}
		results[h] = object.NewFloatArray2D(out, hRows, hCols)
	}

	return &object.List{Elements: results}
}

func getHeadData(obj object.Object) ([]float64, int, int) {
	if data, rows, cols, ok := object.GetFloatMatrix(obj); ok {
		return data, rows, cols
	}
	mat, errObj := toFloatMatrix(obj, "fused_attention", "head")
	if errObj != nil {
		return nil, 0, 0
	}
	rows := len(mat)
	if rows == 0 {
		return nil, 0, 0
	}
	cols := len(mat[0])
	data := make([]float64, 0, rows*cols)
	for _, row := range mat {
		data = append(data, row...)
	}
	return data, rows, cols
}

func fnFusedAttention(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 3, 4); err != nil {
		return err
	}
	causal := true
	if len(args) == 4 {
		c, e := args[3].AsBool()
		if e != nil {
			return e
		}
		causal = c
	}

	qList, ok := args[0].(*object.List)
	if !ok {
		return errors.NewTypeError("LIST", args[0].Type().String())
	}
	kList, ok := args[1].(*object.List)
	if !ok {
		return errors.NewTypeError("LIST", args[1].Type().String())
	}
	vList, ok := args[2].(*object.List)
	if !ok {
		return errors.NewTypeError("LIST", args[2].Type().String())
	}

	nHeads := len(qList.Elements)
	if nHeads == 0 {
		return errors.NewError("fused_attention: empty head lists")
	}

	results := make([]object.Object, nHeads)

	for h := 0; h < nHeads; h++ {
		qData, qRows, qCols := getHeadData(qList.Elements[h])
		kData, kRows, kCols := getHeadData(kList.Elements[h])
		vData, vRows, vCols := getHeadData(vList.Elements[h])

		if qData == nil || kData == nil || vData == nil {
			return errors.NewError("fused_attention: head %d is not a matrix", h)
		}
		if kCols != qCols || vCols != qCols {
			return errors.NewError("fused_attention: head %d dimension mismatch", h)
		}
		if kRows != vRows {
			return errors.NewError("fused_attention: head %d k/v row mismatch", h)
		}

		dk := qCols
		scale := 1.0 / math.Sqrt(float64(dk))
		out := make([]float64, qRows*dk)

		for qi := 0; qi < qRows; qi++ {
			qOff := qi * dk
			scores := make([]float64, kRows)
			for ki := 0; ki < kRows; ki++ {
				kOff := ki * dk
				var dot float64
				for d := 0; d < dk; d++ {
					dot += qData[qOff+d] * kData[kOff+d]
				}
				scores[ki] = dot * scale
			}
			if causal && qRows > 1 {
				for ki := qi + 1; ki < kRows; ki++ {
					scores[ki] = math.Inf(-1)
				}
			}
			maxScore := scores[0]
			for _, s := range scores[1:] {
				if s > maxScore {
					maxScore = s
				}
			}
			var sumExp float64
			exps := make([]float64, kRows)
			for i, s := range scores {
				exps[i] = math.Exp(s - maxScore)
				sumExp += exps[i]
			}
			invSum := 1.0 / sumExp
			rowOff := qi * dk
			for d := 0; d < dk; d++ {
				var val float64
				for ki := 0; ki < kRows; ki++ {
					val += exps[ki] * invSum * vData[ki*dk+d]
				}
				out[rowOff+d] = val
			}
		}
		results[h] = object.NewFloatArray2D(out, qRows, dk)
	}

	return &object.List{Elements: results}
}
