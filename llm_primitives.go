package scriptlingllmlib

import (
	"context"
	"encoding/binary"
	"math"
	"sort"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func fnRmsNorm(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 2, 3); err != nil {
		return err
	}
	w, errObj := toFloatList(args[1], "rms_norm", "weight")
	if errObj != nil {
		return errObj
	}
	eps := 1e-5
	if len(args) == 3 {
		e, err := args[2].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[2].Type().String())
		}
		eps = e
	}

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 {
			return object.NewFloatArray2D(nil, 0, 0)
		}
		if xCols != len(w) {
			return errors.NewError("rms_norm: x columns (%d) must match weight length (%d)", xCols, len(w))
		}
		data := make([]float64, 0, xRows*xCols)
		for i := 0; i < xRows; i++ {
			off := i * xCols
			var ss float64
			for j := 0; j < xCols; j++ {
				v := xData[off+j]
				ss += v * v
			}
			inv := 1.0 / math.Sqrt(ss/float64(xCols)+eps)
			for j := 0; j < xCols; j++ {
				data = append(data, xData[off+j]*inv*w[j])
			}
		}
		return object.NewFloatArray2D(data, xRows, xCols)
	}

	x, errObj := toFloatMatrix(args[0], "rms_norm", "x")
	if errObj != nil {
		return errObj
	}
	rows := len(x)
	if rows == 0 {
		return object.NewFloatArray2D(nil, 0, 0)
	}
	cols := len(x[0])
	data := make([]float64, 0, rows*cols)
	for _, row := range x {
		if len(row) != len(w) {
			return errors.NewError("rms_norm: x columns (%d) must match weight length (%d)", len(row), len(w))
		}
		var ss float64
		for _, v := range row {
			ss += v * v
		}
		ss /= float64(len(row))
		inv := 1.0 / math.Sqrt(ss+eps)
		for j, v := range row {
			data = append(data, v*inv*w[j])
		}
	}
	return object.NewFloatArray2D(data, rows, cols)
}

func fnRope(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 1, 2); err != nil {
		return err
	}
	startPos := int64(0)
	if len(args) == 2 {
		sp, err := args[1].AsInt()
		if err != nil {
			return errors.NewTypeError("INTEGER", args[1].Type().String())
		}
		startPos = sp
	}

	freqBase := 10000.0
	if kwargs.Has("freq_base") {
		freqBase = kwargs.MustGetFloat("freq_base", 10000.0)
	}
	ropeDim := 0
	if kwargs.Has("rope_dim") {
		ropeDim = int(kwargs.MustGetInt("rope_dim", 0))
	}

	validateEven := func(cols int) object.Object {
		effectiveDim := cols
		if ropeDim > 0 && ropeDim < cols {
			effectiveDim = ropeDim
		}
		if effectiveDim%2 != 0 {
			return errors.NewError("rope: dimension must be even, got %d", effectiveDim)
		}
		return nil
	}

	applyRope := func(xData []float64, xRows, xCols int) []float64 {
		effectiveDim := xCols
		if ropeDim > 0 && ropeDim < xCols {
			effectiveDim = ropeDim
		}
		halfDim := effectiveDim / 2
		data := make([]float64, 0, xRows*xCols)
		for seqIdx := 0; seqIdx < xRows; seqIdx++ {
			pos := float64(startPos) + float64(seqIdx)
			off := seqIdx * xCols
			for i := 0; i < halfDim; i++ {
				freq := 1.0 / math.Pow(freqBase, 2.0*float64(i)/float64(effectiveDim))
				angle := freq * pos
				cosA := math.Cos(angle)
				sinA := math.Sin(angle)
				data = append(data, xData[off+2*i]*cosA-xData[off+2*i+1]*sinA)
				data = append(data, xData[off+2*i]*sinA+xData[off+2*i+1]*cosA)
			}
			if effectiveDim < xCols {
				for i := effectiveDim; i < xCols; i++ {
					data = append(data, xData[off+i])
				}
			}
		}
		return data
	}

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 {
			return object.NewFloatArray2D(nil, 0, 0)
		}
		if errObj := validateEven(xCols); errObj != nil {
			return errObj
		}
		data := applyRope(xData, xRows, xCols)
		return object.NewFloatArray2D(data, xRows, xCols)
	}

	x, errObj := toFloatMatrix(args[0], "rope", "x")
	if errObj != nil {
		return errObj
	}
	if len(x) == 0 {
		return object.NewFloatArray2D(nil, 0, 0)
	}
	if errObj := validateEven(len(x[0])); errObj != nil {
		return errObj
	}
	dk := len(x[0])
	seqLen := len(x)

	flatData := make([]float64, 0, seqLen*dk)
	for _, row := range x {
		flatData = append(flatData, row...)
	}
	data := applyRope(flatData, seqLen, dk)
	return object.NewFloatArray2D(data, seqLen, dk)
}

func fnSiluGate(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}

	gateData, gateRows, gateCols, gateOK := object.GetFloatMatrix(args[0])
	upData, upRows, upCols, upOK := object.GetFloatMatrix(args[1])

	if gateOK && upOK {
		if gateRows != upRows {
			return errors.NewError("silu_gate: gate and up must have the same number of rows")
		}
		if gateRows == 0 {
			return object.NewFloatArray2D(nil, 0, 0)
		}
		if gateCols != upCols {
			return errors.NewError("silu_gate: gate and up must have the same number of columns")
		}
		data := make([]float64, 0, gateRows*gateCols)
		for i := 0; i < gateRows; i++ {
			off := i * gateCols
			for j := 0; j < gateCols; j++ {
				g := gateData[off+j]
				s := 1.0 / (1.0 + math.Exp(-g))
				data = append(data, g*s*upData[off+j])
			}
		}
		return object.NewFloatArray2D(data, gateRows, gateCols)
	}

	gate, errObj := toFloatMatrix(args[0], "silu_gate", "gate")
	if errObj != nil {
		return errObj
	}
	up, errObj := toFloatMatrix(args[1], "silu_gate", "up")
	if errObj != nil {
		return errObj
	}
	if len(gate) != len(up) {
		return errors.NewError("silu_gate: gate and up must have the same number of rows")
	}
	if len(gate) == 0 {
		return object.NewFloatArray2D(nil, 0, 0)
	}
	if len(gate[0]) != len(up[0]) {
		return errors.NewError("silu_gate: gate and up must have the same number of columns")
	}

	rows := len(gate)
	cols := len(gate[0])
	data := make([]float64, 0, rows*cols)
	for i := range gate {
		for j := range gate[i] {
			s := 1.0 / (1.0 + math.Exp(-gate[i][j]))
			data = append(data, gate[i][j]*s*up[i][j])
		}
	}
	return object.NewFloatArray2D(data, rows, cols)
}

func fnAttention(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 3, 4); err != nil {
		return err
	}
	causal := true
	if len(args) == 4 {
		c, err := args[3].AsBool()
		if err != nil {
			return errors.NewTypeError("BOOLEAN", args[3].Type().String())
		}
		causal = c
	}

	qData, qRows, qCols, qOK := object.GetFloatMatrix(args[0])
	kData, kRows, kCols, kOK := object.GetFloatMatrix(args[1])
	vData, vRows, vCols, vOK := object.GetFloatMatrix(args[2])

	if qOK && kOK && vOK {
		if qRows == 0 || kRows == 0 || vRows == 0 {
			return errors.NewError("attention: inputs cannot be empty")
		}
		if kCols != qCols || vCols != qCols {
			return errors.NewError("attention: q, k, v must have the same inner dimension")
		}
		if kRows != vRows {
			return errors.NewError("attention: k and v must have the same number of rows")
		}
		dk := qCols
		scale := 1.0 / math.Sqrt(float64(dk))
		data := make([]float64, 0, qRows*dk)
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
			for d := 0; d < dk; d++ {
				var val float64
				for ki := 0; ki < kRows; ki++ {
					val += exps[ki] * invSum * vData[ki*dk+d]
				}
				data = append(data, val)
			}
		}
		return object.NewFloatArray2D(data, qRows, dk)
	}

	q, errObj := toFloatMatrix(args[0], "attention", "q")
	if errObj != nil {
		return errObj
	}
	k, errObj := toFloatMatrix(args[1], "attention", "k")
	if errObj != nil {
		return errObj
	}
	v, errObj := toFloatMatrix(args[2], "attention", "v")
	if errObj != nil {
		return errObj
	}

	if len(q) == 0 || len(k) == 0 || len(v) == 0 {
		return errors.NewError("attention: inputs cannot be empty")
	}
	dk := len(q[0])
	if len(k[0]) != dk || len(v[0]) != dk {
		return errors.NewError("attention: q, k, v must have the same inner dimension")
	}
	if len(k) != len(v) {
		return errors.NewError("attention: k and v must have the same number of rows")
	}

	qLen := len(q)
	kvLen := len(k)
	scale := 1.0 / math.Sqrt(float64(dk))

	data := make([]float64, 0, qLen*dk)
	for qi := 0; qi < qLen; qi++ {
		scores := make([]float64, kvLen)
		for ki := 0; ki < kvLen; ki++ {
			var dot float64
			for d := 0; d < dk; d++ {
				dot += q[qi][d] * k[ki][d]
			}
			scores[ki] = dot * scale
		}

		if causal && qLen > 1 {
			for ki := qi + 1; ki < kvLen; ki++ {
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
		exps := make([]float64, kvLen)
		for i, s := range scores {
			exps[i] = math.Exp(s - maxScore)
			sumExp += exps[i]
		}
		invSum := 1.0 / sumExp

		for d := 0; d < dk; d++ {
			var val float64
			for ki := 0; ki < kvLen; ki++ {
				val += exps[ki] * invSum * v[ki][d]
			}
			data = append(data, val)
		}
	}
	return object.NewFloatArray2D(data, qLen, dk)
}

func fnLinear(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 2, 3); err != nil {
		return err
	}
	var bias []float64
	if len(args) == 3 {
		var errObj object.Object
		bias, errObj = toFloatList(args[2], "linear", "bias")
		if errObj != nil {
			return errObj
		}
	}

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		wData, wRows, wCols, _ := object.GetFloatMatrix(args[1])
		if wData == nil {
			wMat, errObj := toFloatMatrix(args[1], "linear", "weight")
			if errObj != nil {
				return errObj
			}
			wRows = len(wMat)
			if wRows == 0 {
				return errors.NewError("linear: inputs cannot be empty")
			}
			wCols = len(wMat[0])
			flatW := make([]float64, 0, wRows*wCols)
			for _, r := range wMat {
				flatW = append(flatW, r...)
			}
			wData = flatW
		}
		if xRows == 0 || wRows == 0 {
			return errors.NewError("linear: inputs cannot be empty")
		}
		if wCols != xCols {
			return errors.NewError("linear: weight columns (%d) must match x columns (%d)", wCols, xCols)
		}
		data := make([]float64, xRows*wRows)
		total := xRows * wRows
		parallelFor(total, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / wRows
				j := idx % wRows
				xOff := xi * xCols
				wOff := j * wCols
				var sum float64
				for l := 0; l < xCols; l++ {
					sum += xData[xOff+l] * wData[wOff+l]
				}
				if bias != nil {
					sum += bias[j]
				}
				data[xi*wRows+j] = sum
			}
		})
		return object.NewFloatArray2D(data, xRows, wRows)
	}

	x, errObj := toFloatMatrix(args[0], "linear", "x")
	if errObj != nil {
		return errObj
	}
	weight, errObj := toFloatMatrix(args[1], "linear", "weight")
	if errObj != nil {
		return errObj
	}

	if len(x) == 0 || len(weight) == 0 {
		return errors.NewError("linear: inputs cannot be empty")
	}
	inFeatures := len(x[0])
	if len(weight[0]) != inFeatures {
		return errors.NewError("linear: weight columns (%d) must match x columns (%d)", len(weight[0]), inFeatures)
	}
	outFeatures := len(weight)
	seqLen := len(x)

	data := make([]float64, 0, seqLen*outFeatures)
	for _, row := range x {
		for j := 0; j < outFeatures; j++ {
			var sum float64
			for l := 0; l < inFeatures; l++ {
				sum += row[l] * weight[j][l]
			}
			if bias != nil {
				sum += bias[j]
			}
			data = append(data, sum)
		}
	}
	return object.NewFloatArray2D(data, seqLen, outFeatures)
}

func fnLinearRow(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 2, 3); err != nil {
		return err
	}
	var bias []float64
	if len(args) == 3 {
		var errObj object.Object
		bias, errObj = toFloatList(args[2], "linear_row", "bias")
		if errObj != nil {
			return errObj
		}
	}

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		wData, wRows, wCols, _ := object.GetFloatMatrix(args[1])
		if wData == nil {
			wMat, errObj := toFloatMatrix(args[1], "linear_row", "weight")
			if errObj != nil {
				return errObj
			}
			if len(wMat) == 0 {
				return errors.NewError("linear_row: inputs cannot be empty")
			}
			wRows = len(wMat)
			wCols = len(wMat[0])
			flatW := make([]float64, 0, wRows*wCols)
			for _, r := range wMat {
				flatW = append(flatW, r...)
			}
			wData = flatW
		}
		if xRows == 0 || wRows == 0 {
			return errors.NewError("linear_row: inputs cannot be empty")
		}
		if wCols != xCols {
			return errors.NewError("linear_row: weight columns (%d) must match x columns (%d)", wCols, xCols)
		}
		lastOff := (xRows - 1) * xCols
		result := make([]float64, wRows)
		parallelFor(wRows, func(start, end int) {
			for j := start; j < end; j++ {
				wOff := j * wCols
				var sum float64
				for l := 0; l < xCols; l++ {
					sum += xData[lastOff+l] * wData[wOff+l]
				}
				if bias != nil {
					sum += bias[j]
				}
				result[j] = sum
			}
		})
		return object.NewFloatArray1D(result)
	}

	x, errObj := toFloatMatrix(args[0], "linear_row", "x")
	if errObj != nil {
		return errObj
	}
	weight, errObj := toFloatMatrix(args[1], "linear_row", "weight")
	if errObj != nil {
		return errObj
	}

	if len(x) == 0 || len(weight) == 0 {
		return errors.NewError("linear_row: inputs cannot be empty")
	}
	inFeatures := len(x[0])
	if len(weight[0]) != inFeatures {
		return errors.NewError("linear_row: weight columns (%d) must match x columns (%d)", len(weight[0]), inFeatures)
	}
	outFeatures := len(weight)

	lastRow := x[len(x)-1]
	result := make([]float64, outFeatures)
	for j := 0; j < outFeatures; j++ {
		var sum float64
		for l := 0; l < inFeatures; l++ {
			sum += lastRow[l] * weight[j][l]
		}
		if bias != nil {
			sum += bias[j]
		}
		result[j] = sum
	}
	return object.NewFloatArray1D(result)
}

type indexedFloat struct {
	index int
	value float64
}

func fnTopK(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	logits, errObj := toFloatList(args[0], "top_k", "logits")
	if errObj != nil {
		return errObj
	}
	k, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	n := len(logits)
	if k <= 0 {
		return errors.NewError("top_k: k must be positive")
	}
	if int(k) > n {
		k = int64(n)
	}

	indexed := make([]indexedFloat, n)
	for i, v := range logits {
		indexed[i] = indexedFloat{index: i, value: v}
	}

	top := make([]indexedFloat, k)
	copy(top, indexed[:k])

	sort.Slice(top, func(i, j int) bool {
		return top[i].value > top[j].value
	})

	for i := int(k); i < n; i++ {
		if indexed[i].value > top[k-1].value {
			pos := sort.Search(int(k), func(j int) bool {
				return top[j].value < indexed[i].value
			})
			copy(top[pos+1:], top[pos:])
			top[pos] = indexed[i]
		}
	}

	result := make([]object.Object, k)
	for i, tv := range top {
		result[i] = &object.List{Elements: []object.Object{
			object.NewInteger(int64(tv.index)),
			&object.Float{Value: tv.value},
		}}
	}
	return &object.List{Elements: result}
}

func fnDequantizeQ8(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 3); err != nil {
		return err
	}
	dataList, ok := args[0].(*object.List)
	if !ok {
		return errors.NewTypeError("LIST", args[0].Type().String())
	}
	scales, errObj := toFloatList(args[1], "dequantize_q8", "scales")
	if errObj != nil {
		return errObj
	}
	groupSize, err := args[2].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[2].Type().String())
	}
	if groupSize <= 0 {
		return errors.NewError("dequantize_q8: group_size must be positive")
	}

	n := len(dataList.Elements)
	if n == 0 {
		return object.NewFloatArray1D(nil)
	}
	numGroups := (n + int(groupSize) - 1) / int(groupSize)
	if len(scales) < numGroups {
		return errors.NewError("dequantize_q8: not enough scales (%d) for %d groups", len(scales), numGroups)
	}

	data := make([]int8, n)
	for i, el := range dataList.Elements {
		v, err := el.AsInt()
		if err != nil {
			return errors.NewTypeError("INTEGER", el.Type().String())
		}
		if v < -128 || v > 127 {
			return errors.NewError("dequantize_q8: data[%d] = %d is out of int8 range", i, v)
		}
		data[i] = int8(v)
	}

	result := make([]float64, n)
	for i, d := range data {
		groupIdx := i / int(groupSize)
		result[i] = float64(d) * scales[groupIdx]
	}
	return object.NewFloatArray1D(result)
}

func fnDequantizeQ8_0(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	raw, ok := args[0].(*object.String)
	if !ok {
		return errors.NewTypeError("STRING", args[0].Type().String())
	}
	nGroups, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	if nGroups <= 0 {
		return errors.NewError("dequantize_q8_0: n_groups must be positive")
	}

	rawStr := raw.Value
	expectedLen := int(nGroups) * 34
	if len(rawStr) < expectedLen {
		return errors.NewError("dequantize_q8_0: raw data too short (%d bytes, need %d)", len(rawStr), expectedLen)
	}

	nElements := int(nGroups) * 32
	result := make([]float64, nElements)
	rawBytes := []byte(rawStr)
	idx := 0
	for g := 0; g < int(nGroups); g++ {
		base := g * 34
		scaleBits := binary.LittleEndian.Uint16(rawBytes[base : base+2])
		scale := float16ToFloat64(scaleBits)
		for i := 0; i < 32; i++ {
			q := int8(rawBytes[base+2+i])
			result[idx] = float64(q) * scale
			idx++
		}
	}
	return object.NewFloatArray1D(result)
}

func float16ToFloat64(bits uint16) float64 {
	sign := float64(1)
	if bits&0x8000 != 0 {
		sign = -1
	}
	exp := int((bits >> 10) & 0x1f)
	frac := float64(bits & 0x3ff)
	switch {
	case exp == 0 && frac == 0:
		return sign * 0
	case exp == 0:
		return sign * math.Ldexp(frac/1024.0, -14)
	case exp == 31 && frac == 0:
		return sign * math.Inf(1)
	case exp == 31:
		return math.NaN()
	default:
		return sign * math.Ldexp(1.0+frac/1024.0, exp-15)
	}
}

func fnLinearQ8(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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

	scales := make([]float64, outFeatures*groupsPerRow)
	quantData := make([]int8, outFeatures*inFeatures)
	for j := 0; j < outFeatures; j++ {
		rowOff := j * rowBytes
		for g := 0; g < groupsPerRow; g++ {
			base := rowOff + g*34
			scales[j*groupsPerRow+g] = float16ToFloat64(binary.LittleEndian.Uint16(rawBytes[base : base+2]))
			dataOff := j*inFeatures + g*32
			for i := 0; i < 32; i++ {
				quantData[dataOff+i] = int8(rawBytes[base+2+i])
			}
		}
	}

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_q8: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_q8: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		data := make([]float64, 0, xRows*outFeatures)
		for xi := 0; xi < xRows; xi++ {
			rowOff2 := xi * xCols
			for j := 0; j < outFeatures; j++ {
				wOff := j * inFeatures
				sOff := j * groupsPerRow
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					s := scales[sOff+g]
					dOff := wOff + g*32
					xBase := rowOff2 + g*32
					sum += float64(quantData[dOff])*s*xData[xBase] +
						float64(quantData[dOff+1])*s*xData[xBase+1] +
						float64(quantData[dOff+2])*s*xData[xBase+2] +
						float64(quantData[dOff+3])*s*xData[xBase+3] +
						float64(quantData[dOff+4])*s*xData[xBase+4] +
						float64(quantData[dOff+5])*s*xData[xBase+5] +
						float64(quantData[dOff+6])*s*xData[xBase+6] +
						float64(quantData[dOff+7])*s*xData[xBase+7] +
						float64(quantData[dOff+8])*s*xData[xBase+8] +
						float64(quantData[dOff+9])*s*xData[xBase+9] +
						float64(quantData[dOff+10])*s*xData[xBase+10] +
						float64(quantData[dOff+11])*s*xData[xBase+11] +
						float64(quantData[dOff+12])*s*xData[xBase+12] +
						float64(quantData[dOff+13])*s*xData[xBase+13] +
						float64(quantData[dOff+14])*s*xData[xBase+14] +
						float64(quantData[dOff+15])*s*xData[xBase+15] +
						float64(quantData[dOff+16])*s*xData[xBase+16] +
						float64(quantData[dOff+17])*s*xData[xBase+17] +
						float64(quantData[dOff+18])*s*xData[xBase+18] +
						float64(quantData[dOff+19])*s*xData[xBase+19] +
						float64(quantData[dOff+20])*s*xData[xBase+20] +
						float64(quantData[dOff+21])*s*xData[xBase+21] +
						float64(quantData[dOff+22])*s*xData[xBase+22] +
						float64(quantData[dOff+23])*s*xData[xBase+23] +
						float64(quantData[dOff+24])*s*xData[xBase+24] +
						float64(quantData[dOff+25])*s*xData[xBase+25] +
						float64(quantData[dOff+26])*s*xData[xBase+26] +
						float64(quantData[dOff+27])*s*xData[xBase+27] +
						float64(quantData[dOff+28])*s*xData[xBase+28] +
						float64(quantData[dOff+29])*s*xData[xBase+29] +
						float64(quantData[dOff+30])*s*xData[xBase+30] +
						float64(quantData[dOff+31])*s*xData[xBase+31]
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
	for i, row := range xMat {
		if len(row) != inFeatures {
			return errors.NewError("linear_q8: x row %d has %d columns, want %d", i, len(row), inFeatures)
		}
	}

	data := make([]float64, 0, seqLen*outFeatures)
	for xi := 0; xi < seqLen; xi++ {
		xRow := xMat[xi]
		for j := 0; j < outFeatures; j++ {
			wOff := j * inFeatures
			sOff := j * groupsPerRow
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				s := scales[sOff+g]
				dOff := wOff + g*32
				xOff2 := g * 32
				sum += float64(quantData[dOff])*s*xRow[xOff2] +
					float64(quantData[dOff+1])*s*xRow[xOff2+1] +
					float64(quantData[dOff+2])*s*xRow[xOff2+2] +
					float64(quantData[dOff+3])*s*xRow[xOff2+3] +
					float64(quantData[dOff+4])*s*xRow[xOff2+4] +
					float64(quantData[dOff+5])*s*xRow[xOff2+5] +
					float64(quantData[dOff+6])*s*xRow[xOff2+6] +
					float64(quantData[dOff+7])*s*xRow[xOff2+7] +
					float64(quantData[dOff+8])*s*xRow[xOff2+8] +
					float64(quantData[dOff+9])*s*xRow[xOff2+9] +
					float64(quantData[dOff+10])*s*xRow[xOff2+10] +
					float64(quantData[dOff+11])*s*xRow[xOff2+11] +
					float64(quantData[dOff+12])*s*xRow[xOff2+12] +
					float64(quantData[dOff+13])*s*xRow[xOff2+13] +
					float64(quantData[dOff+14])*s*xRow[xOff2+14] +
					float64(quantData[dOff+15])*s*xRow[xOff2+15] +
					float64(quantData[dOff+16])*s*xRow[xOff2+16] +
					float64(quantData[dOff+17])*s*xRow[xOff2+17] +
					float64(quantData[dOff+18])*s*xRow[xOff2+18] +
					float64(quantData[dOff+19])*s*xRow[xOff2+19] +
					float64(quantData[dOff+20])*s*xRow[xOff2+20] +
					float64(quantData[dOff+21])*s*xRow[xOff2+21] +
					float64(quantData[dOff+22])*s*xRow[xOff2+22] +
					float64(quantData[dOff+23])*s*xRow[xOff2+23] +
					float64(quantData[dOff+24])*s*xRow[xOff2+24] +
					float64(quantData[dOff+25])*s*xRow[xOff2+25] +
					float64(quantData[dOff+26])*s*xRow[xOff2+26] +
					float64(quantData[dOff+27])*s*xRow[xOff2+27] +
					float64(quantData[dOff+28])*s*xRow[xOff2+28] +
					float64(quantData[dOff+29])*s*xRow[xOff2+29] +
					float64(quantData[dOff+30])*s*xRow[xOff2+30] +
					float64(quantData[dOff+31])*s*xRow[xOff2+31]
			}
			data = append(data, sum)
		}
	}
	return object.NewFloatArray2D(data, seqLen, outFeatures)
}

func fnLinearRowQ8(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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

	scales := make([]float64, outFeatures*groupsPerRow)
	quantData := make([]int8, outFeatures*inFeatures)
	for j := 0; j < outFeatures; j++ {
		rowOff := j * rowBytes
		for g := 0; g < groupsPerRow; g++ {
			base := rowOff + g*34
			scales[j*groupsPerRow+g] = float16ToFloat64(binary.LittleEndian.Uint16(rawBytes[base : base+2]))
			dataOff := j*inFeatures + g*32
			for i := 0; i < 32; i++ {
				quantData[dataOff+i] = int8(rawBytes[base+2+i])
			}
		}
	}

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_row_q8: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_row_q8: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		lastOff := (xRows - 1) * xCols
		result := make([]float64, outFeatures)
		for j := 0; j < outFeatures; j++ {
			wOff := j * inFeatures
			sOff := j * groupsPerRow
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				s := scales[sOff+g]
				dOff := wOff + g*32
				xBase := lastOff + g*32
				sum += float64(quantData[dOff])*s*xData[xBase] +
					float64(quantData[dOff+1])*s*xData[xBase+1] +
					float64(quantData[dOff+2])*s*xData[xBase+2] +
					float64(quantData[dOff+3])*s*xData[xBase+3] +
					float64(quantData[dOff+4])*s*xData[xBase+4] +
					float64(quantData[dOff+5])*s*xData[xBase+5] +
					float64(quantData[dOff+6])*s*xData[xBase+6] +
					float64(quantData[dOff+7])*s*xData[xBase+7] +
					float64(quantData[dOff+8])*s*xData[xBase+8] +
					float64(quantData[dOff+9])*s*xData[xBase+9] +
					float64(quantData[dOff+10])*s*xData[xBase+10] +
					float64(quantData[dOff+11])*s*xData[xBase+11] +
					float64(quantData[dOff+12])*s*xData[xBase+12] +
					float64(quantData[dOff+13])*s*xData[xBase+13] +
					float64(quantData[dOff+14])*s*xData[xBase+14] +
					float64(quantData[dOff+15])*s*xData[xBase+15] +
					float64(quantData[dOff+16])*s*xData[xBase+16] +
					float64(quantData[dOff+17])*s*xData[xBase+17] +
					float64(quantData[dOff+18])*s*xData[xBase+18] +
					float64(quantData[dOff+19])*s*xData[xBase+19] +
					float64(quantData[dOff+20])*s*xData[xBase+20] +
					float64(quantData[dOff+21])*s*xData[xBase+21] +
					float64(quantData[dOff+22])*s*xData[xBase+22] +
					float64(quantData[dOff+23])*s*xData[xBase+23] +
					float64(quantData[dOff+24])*s*xData[xBase+24] +
					float64(quantData[dOff+25])*s*xData[xBase+25] +
					float64(quantData[dOff+26])*s*xData[xBase+26] +
					float64(quantData[dOff+27])*s*xData[xBase+27] +
					float64(quantData[dOff+28])*s*xData[xBase+28] +
					float64(quantData[dOff+29])*s*xData[xBase+29] +
					float64(quantData[dOff+30])*s*xData[xBase+30] +
					float64(quantData[dOff+31])*s*xData[xBase+31]
			}
			result[j] = sum
		}
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
	for j := 0; j < outFeatures; j++ {
		wOff := j * inFeatures
		sOff := j * groupsPerRow
		var sum float64
		for g := 0; g < groupsPerRow; g++ {
			s := scales[sOff+g]
			dOff := wOff + g*32
			xOff := g * 32
			sum += float64(quantData[dOff])*s*lastRow[xOff] +
				float64(quantData[dOff+1])*s*lastRow[xOff+1] +
				float64(quantData[dOff+2])*s*lastRow[xOff+2] +
				float64(quantData[dOff+3])*s*lastRow[xOff+3] +
				float64(quantData[dOff+4])*s*lastRow[xOff+4] +
				float64(quantData[dOff+5])*s*lastRow[xOff+5] +
				float64(quantData[dOff+6])*s*lastRow[xOff+6] +
				float64(quantData[dOff+7])*s*lastRow[xOff+7] +
				float64(quantData[dOff+8])*s*lastRow[xOff+8] +
				float64(quantData[dOff+9])*s*lastRow[xOff+9] +
				float64(quantData[dOff+10])*s*lastRow[xOff+10] +
				float64(quantData[dOff+11])*s*lastRow[xOff+11] +
				float64(quantData[dOff+12])*s*lastRow[xOff+12] +
				float64(quantData[dOff+13])*s*lastRow[xOff+13] +
				float64(quantData[dOff+14])*s*lastRow[xOff+14] +
				float64(quantData[dOff+15])*s*lastRow[xOff+15] +
				float64(quantData[dOff+16])*s*lastRow[xOff+16] +
				float64(quantData[dOff+17])*s*lastRow[xOff+17] +
				float64(quantData[dOff+18])*s*lastRow[xOff+18] +
				float64(quantData[dOff+19])*s*lastRow[xOff+19] +
				float64(quantData[dOff+20])*s*lastRow[xOff+20] +
				float64(quantData[dOff+21])*s*lastRow[xOff+21] +
				float64(quantData[dOff+22])*s*lastRow[xOff+22] +
				float64(quantData[dOff+23])*s*lastRow[xOff+23] +
				float64(quantData[dOff+24])*s*lastRow[xOff+24] +
				float64(quantData[dOff+25])*s*lastRow[xOff+25] +
				float64(quantData[dOff+26])*s*lastRow[xOff+26] +
				float64(quantData[dOff+27])*s*lastRow[xOff+27] +
				float64(quantData[dOff+28])*s*lastRow[xOff+28] +
				float64(quantData[dOff+29])*s*lastRow[xOff+29] +
				float64(quantData[dOff+30])*s*lastRow[xOff+30] +
				float64(quantData[dOff+31])*s*lastRow[xOff+31]
		}
		result[j] = sum
	}
	return object.NewFloatArray1D(result)
}

func dequantizeQ4_0Block(raw []byte, off int) (scale float64, values [32]int8) {
	scale = float16ToFloat64(binary.LittleEndian.Uint16(raw[off : off+2]))
	for i := 0; i < 16; i++ {
		b := raw[off+2+i]
		values[i*2] = int8(b&0x0F) - 8
		values[i*2+1] = int8((b>>4)&0x0F) - 8
	}
	return
}

func fnDequantizeQ4_0(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	raw, ok := args[0].(*object.String)
	if !ok {
		return errors.NewTypeError("STRING", args[0].Type().String())
	}
	nGroups, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	rawBytes := []byte(raw.Value)
	result := make([]float64, int(nGroups)*32)
	for g := 0; g < int(nGroups); g++ {
		scale, values := dequantizeQ4_0Block(rawBytes, g*18)
		off := g * 32
		for i := 0; i < 32; i++ {
			result[off+i] = float64(values[i]) * scale
		}
	}
	return object.NewFloatArray1D(result)
}

func fnLinearQ4(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
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

	scales := make([]float64, outFeatures*groupsPerRow)
	quantData := make([]int8, outFeatures*inFeatures)
	for j := 0; j < outFeatures; j++ {
		rOff := j * rowBytes
		for g := 0; g < groupsPerRow; g++ {
			base := rOff + g*18
			s, vals := dequantizeQ4_0Block(rawBytes, base)
			scales[j*groupsPerRow+g] = s
			dataOff := j*inFeatures + g*32
			for i := 0; i < 32; i++ {
				quantData[dataOff+i] = vals[i]
			}
		}
	}

	if xData, xRows, xCols, ok := object.GetFloatMatrix(args[0]); ok {
		if xRows == 0 || outFeatures == 0 {
			return errors.NewError("linear_q4: inputs cannot be empty")
		}
		if xCols != inFeatures {
			return errors.NewError("linear_q4: x columns (%d) must match in_features (%d)", xCols, inFeatures)
		}
		data := make([]float64, 0, xRows*outFeatures)
		for xi := 0; xi < xRows; xi++ {
			rowOff := xi * xCols
			for j := 0; j < outFeatures; j++ {
				wOff := j * inFeatures
				sOff := j * groupsPerRow
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					s := scales[sOff+g]
					dOff := wOff + g*32
					xBase := rowOff + g*32
					for i := 0; i < 32; i++ {
						sum += float64(quantData[dOff+i]) * s * xData[xBase+i]
					}
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
	for i, row := range xMat {
		if len(row) != inFeatures {
			return errors.NewError("linear_q4: x row %d has %d columns, want %d", i, len(row), inFeatures)
		}
	}
	data := make([]float64, 0, seqLen*outFeatures)
	for xi := 0; xi < seqLen; xi++ {
		xRow := xMat[xi]
		for j := 0; j < outFeatures; j++ {
			wOff := j * inFeatures
			sOff := j * groupsPerRow
			var sum float64
			for g := 0; g < groupsPerRow; g++ {
				s := scales[sOff+g]
				dOff := wOff + g*32
				xOff2 := g * 32
				for i := 0; i < 32; i++ {
					sum += float64(quantData[dOff+i]) * s * xRow[xOff2+i]
				}
			}
			data = append(data, sum)
		}
	}
	return object.NewFloatArray2D(data, seqLen, outFeatures)
}
