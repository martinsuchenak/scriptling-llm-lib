package scriptlingllmlib

import (
	"context"
	"math"
	"sync"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func fnFusedBlock(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 13, 15); err != nil {
		return err
	}

	xData, xRows, xCols, xOK := object.GetFloatMatrix(args[0])
	if !xOK {
		xMat, errObj := toFloatMatrix(args[0], "fused_block", "x")
		if errObj != nil {
			return errObj
		}
		if len(xMat) == 0 {
			return errors.NewError("fused_block: x cannot be empty")
		}
		xRows = len(xMat)
		xCols = len(xMat[0])
		xData = make([]float64, 0, xRows*xCols)
		for _, row := range xMat {
			xData = append(xData, row...)
		}
	}
	if xRows == 0 {
		return errors.NewError("fused_block: x cannot be empty")
	}

	attnNormW, err := toFloatList(args[1], "fused_block", "attn_norm_w")
	if err != nil {
		return err
	}
	wQ := args[2]
	wK := args[3]
	wV := args[4]
	wO := args[5]

	ffnNormW, err := toFloatList(args[6], "fused_block", "ffn_norm_w")
	if err != nil {
		return err
	}
	wGate := args[7]
	wUp := args[8]
	wDown := args[9]

	nHeads := int(mustGetIntArg(args[10], 1))
	nKVHeads := int(mustGetIntArg(args[11], 1))
	dK := int(mustGetIntArg(args[12], 1))

	startPos := int64(0)
	if len(args) > 13 {
		startPos, _ = args[13].AsInt()
	}
	causal := true
	if len(args) > 14 {
		causal, _ = args[14].AsBool()
	}

	freqBase := kwargs.MustGetFloat("freq_base", 10000.0)
	ropeDim := int(kwargs.MustGetInt("rope_dim", 0))
	eps := kwargs.MustGetFloat("eps", 1e-5)

	// Optional cached K/V: list of FloatArrays, one per kv_head
	cachedK, cachedKRows := getKwargHeadList(kwargs, "cached_k")
	cachedV, cachedVRows := getKwargHeadList(kwargs, "cached_v")

	// Step 1: RMS norm (attention)
	normed := make([]float64, xRows*xCols)
	rmsNormFlat(xData, attnNormW, eps, xRows, xCols, normed)

	// Step 2: Parallel QKV projection
	var qData, kData, vData []float64
	var qRows, qCols, kRows, kCols, vRows, vCols int
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		qData, qRows, qCols = fusedMatmul(wQ, normed, xRows, xCols)
		wg.Done()
	}()
	go func() {
		kData, kRows, kCols = fusedMatmul(wK, normed, xRows, xCols)
		wg.Done()
	}()
	go func() {
		vData, vRows, vCols = fusedMatmul(wV, normed, xRows, xCols)
		wg.Done()
	}()
	wg.Wait()

	if qData == nil || kData == nil || vData == nil {
		return errors.NewError("fused_block: QKV projection failed")
	}

	// Step 3: Split heads
	qHeads := splitHeadsData(qData, qRows, qCols, nHeads)
	kHeads := splitHeadsData(kData, kRows, kCols, nKVHeads)
	vHeads := splitHeadsData(vData, vRows, vCols, nKVHeads)

	// Step 4: RoPE
	if ropeDim > 0 && dK > 0 {
		halfDim := ropeDim / 2
		freqs := make([]float64, halfDim)
		for i := 0; i < halfDim; i++ {
			freqs[i] = 1.0 / math.Pow(freqBase, 2.0*float64(i)/float64(ropeDim))
		}
		for h := 0; h < nHeads; h++ {
			applyRopeInPlace(qHeads[h], qRows, dK, int(startPos), freqs, halfDim)
		}
		for h := 0; h < nKVHeads; h++ {
			applyRopeInPlace(kHeads[h], kRows, dK, int(startPos), freqs, halfDim)
		}
	}

	// Step 5: Merge with cached K/V
	if cachedK != nil && len(cachedK) == nKVHeads {
		for h := 0; h < nKVHeads; h++ {
			newK := kHeads[h]
			old := cachedK[h]
			totalRows := cachedKRows + kRows
			merged := make([]float64, totalRows*dK)
			copy(merged, old)
			copy(merged[cachedKRows*dK:], newK)
			kHeads[h] = merged
		}
	}
	if cachedV != nil && len(cachedV) == nKVHeads {
		for h := 0; h < nKVHeads; h++ {
			newV := vHeads[h]
			old := cachedV[h]
			totalRows := cachedVRows + vRows
			merged := make([]float64, totalRows*dK)
			copy(merged, old)
			copy(merged[cachedVRows*dK:], newV)
			vHeads[h] = merged
		}
	}

	// Repeat KV if GQA
	nRep := nHeads / nKVHeads
	if nRep > 1 {
		kHeads = repeatKVData(kHeads, nRep)
		vHeads = repeatKVData(vHeads, nRep)
	}

	// Step 6: Multi-head attention
	kvLen := len(kHeads[0]) / dK
	attnOut := make([]float64, xRows*nHeads*dK)
	for h := 0; h < nHeads; h++ {
		fusedAttentionHead(qHeads[h], kHeads[h], vHeads[h], qRows, dK, kvLen, causal, 0, attnOut, h*qRows*dK)
	}

	// Step 7: Merge heads
	merged := mergeHeadsData(attnOut, xRows, nHeads, dK)

	// Step 8: Output projection
	oData, _, _ := fusedMatmul(wO, merged, xRows, nHeads*dK)
	if oData == nil {
		return errors.NewError("fused_block: output projection failed")
	}

	// Step 9: Residual add
	residual := make([]float64, xRows*xCols)
	for i := 0; i < xRows*xCols; i++ {
		residual[i] = xData[i] + oData[i]
	}

	// Step 10: RMS norm (FFN)
	ffnNormed := make([]float64, xRows*xCols)
	rmsNormFlat(residual, ffnNormW, eps, xRows, xCols, ffnNormed)

	// Step 11: Fused FFN
	var gateData, upData []float64
	var gateRows, gateCols int
	wg.Add(2)
	go func() {
		gateData, gateRows, gateCols = fusedMatmul(wGate, ffnNormed, xRows, xCols)
		wg.Done()
	}()
	go func() {
		upData, _, _ = fusedMatmul(wUp, ffnNormed, xRows, xCols)
		wg.Done()
	}()
	wg.Wait()

	if gateData == nil || upData == nil {
		return errors.NewError("fused_block: FFN gate/up failed")
	}

	hidden := gateRows * gateCols
	siluData := make([]float64, hidden)
	for i := 0; i < hidden; i++ {
		g := gateData[i]
		s := 1.0 / (1.0 + math.Exp(-g))
		siluData[i] = g * s * upData[i]
	}

	downData, _, _ := fusedMatmul(wDown, siluData, gateRows, gateCols)
	if downData == nil {
		return errors.NewError("fused_block: down projection failed")
	}

	// Step 12: Final residual add
	result := make([]float64, xRows*xCols)
	for i := 0; i < xRows*xCols; i++ {
		result[i] = residual[i] + downData[i]
	}

	// Build return: [result, new_k_heads, new_v_heads]
	newKList := make([]object.Object, nKVHeads)
	newVList := make([]object.Object, nKVHeads)
	for h := 0; h < nKVHeads; h++ {
		newKList[h] = object.NewFloatArray2D(kHeads[h], kRows+cachedKRows, dK)
		newVList[h] = object.NewFloatArray2D(vHeads[h], vRows+cachedVRows, dK)
	}

	return &object.List{Elements: []object.Object{
		object.NewFloatArray2D(result, xRows, xCols),
		&object.List{Elements: newKList},
		&object.List{Elements: newVList},
	}}
}

func rmsNormFlat(xData, w []float64, eps float64, rows, cols int, out []float64) {
	for r := 0; r < rows; r++ {
		off := r * cols
		var ss float64
		for j := 0; j < cols; j++ {
			v := xData[off+j]
			ss += v * v
		}
		inv := 1.0 / math.Sqrt(ss/float64(cols)+eps)
		for j := 0; j < cols; j++ {
			out[off+j] = xData[off+j] * inv * w[j]
		}
	}
}

func applyRopeInPlace(data []float64, rows, dK, startPos int, freqs []float64, halfDim int) {
	for r := 0; r < rows; r++ {
		pos := startPos + r
		off := r * dK
		for i := 0; i < halfDim; i++ {
			angle := freqs[i] * float64(pos)
			cosA := math.Cos(angle)
			sinA := math.Sin(angle)
			x0 := data[off+2*i]
			x1 := data[off+2*i+1]
			data[off+2*i] = x0*cosA - x1*sinA
			data[off+2*i+1] = x0*sinA + x1*cosA
		}
	}
}

func fusedAttentionHead(qData, kData, vData []float64, qRows, dK, kRows int, causal bool, cacheLen int, out []float64, outOff int) {
	scale := 1.0 / math.Sqrt(float64(dK))
	for qi := 0; qi < qRows; qi++ {
		qOff := qi * dK
		scores := make([]float64, kRows)
		for ki := 0; ki < kRows; ki++ {
			kOff := ki * dK
			var dot float64
			for d := 0; d < dK; d++ {
				dot += qData[qOff+d] * kData[kOff+d]
			}
			scores[ki] = dot * scale
		}
		if causal {
			limit := cacheLen + qi + 1
			for ki := limit; ki < kRows; ki++ {
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
		rowOff := outOff + qi*dK
		for d := 0; d < dK; d++ {
			var val float64
			for ki := 0; ki < kRows; ki++ {
				val += exps[ki] * invSum * vData[ki*dK+d]
			}
			out[rowOff+d] = val
		}
	}
}

func getKwargHeadList(kwargs object.Kwargs, key string) ([][]float64, int) {
	if !kwargs.Has(key) {
		return nil, 0
	}
	val := kwargs.Get(key)
	list, ok := val.(*object.List)
	if !ok {
		return nil, 0
	}
	n := len(list.Elements)
	if n == 0 {
		return nil, 0
	}
	result := make([][]float64, n)
	var rows int
	for i, el := range list.Elements {
		data, r, _, ok := object.GetFloatMatrix(el)
		if !ok {
			return nil, 0
		}
		result[i] = data
		rows = r
	}
	return result, rows
}

func splitHeadsData(data []float64, rows, cols, nHeads int) [][]float64 {
	dK := cols / nHeads
	heads := make([][]float64, nHeads)
	for h := 0; h < nHeads; h++ {
		headData := make([]float64, rows*dK)
		for r := 0; r < rows; r++ {
			srcOff := r*cols + h*dK
			dstOff := r * dK
			copy(headData[dstOff:dstOff+dK], data[srcOff:srcOff+dK])
		}
		heads[h] = headData
	}
	return heads
}

func repeatKVData(heads [][]float64, nRep int) [][]float64 {
	expanded := make([][]float64, 0, len(heads)*nRep)
	for _, h := range heads {
		for i := 0; i < nRep; i++ {
			expanded = append(expanded, h)
		}
	}
	return expanded
}

func mergeHeadsData(data []float64, seqLen, nHeads, dK int) []float64 {
	totalCols := nHeads * dK
	result := make([]float64, seqLen*totalCols)
	for s := 0; s < seqLen; s++ {
		for h := 0; h < nHeads; h++ {
			srcOff := h*seqLen*dK + s*dK
			dstOff := s*totalCols + h*dK
			copy(result[dstOff:dstOff+dK], data[srcOff:srcOff+dK])
		}
	}
	return result
}

func mustGetIntArg(arg object.Object, defaultVal int64) int64 {
	if arg == nil {
		return defaultVal
	}
	v, e := arg.AsInt()
	if e != nil {
		return defaultVal
	}
	return v
}
