package scriptlingllmlib

import (
	"context"
	"encoding/binary"
	"math"
	"sort"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

// fnRmsNorm implements llm.rms_norm: RMS normalization.
// For each row: divide by root-mean-square, then multiply element-wise by weight.
// Used by LLaMA, Mistral, Phi, Qwen. Called twice per transformer layer.
func fnRmsNorm(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 2, 3); err != nil {
		return err
	}
	x, errObj := toFloatMatrix(args[0], "rms_norm", "x")
	if errObj != nil {
		return errObj
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

	result := make([][]float64, len(x))
	for i, row := range x {
		if len(row) != len(w) {
			return errors.NewError("rms_norm: x columns (%d) must match weight length (%d)", len(row), len(w))
		}
		var ss float64
		for _, v := range row {
			ss += v * v
		}
		ss /= float64(len(row))
		inv := 1.0 / math.Sqrt(ss+eps)
		result[i] = make([]float64, len(row))
		for j, v := range row {
			result[i][j] = v * inv * w[j]
		}
	}
	return floatMatrixToObject(result)
}

// fnRope implements llm.rope: Rotary Position Embeddings.
// Applies position-dependent rotation to pairs of dimensions (2i, 2i+1).
// Standard positional encoding for most modern transformers.
func fnRope(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 1, 2); err != nil {
		return err
	}
	x, errObj := toFloatMatrix(args[0], "rope", "x")
	if errObj != nil {
		return errObj
	}
	startPos := int64(0)
	if len(args) == 2 {
		sp, err := args[1].AsInt()
		if err != nil {
			return errors.NewTypeError("INTEGER", args[1].Type().String())
		}
		startPos = sp
	}

	if len(x) == 0 {
		return &object.List{Elements: []object.Object{}}
	}
	dk := len(x[0])
	if dk%2 != 0 {
		return errors.NewError("rope: last dimension must be even (got %d)", dk)
	}
	halfDim := dk / 2

	result := make([][]float64, len(x))
	for seqIdx, row := range x {
		pos := float64(startPos) + float64(seqIdx)
		result[seqIdx] = make([]float64, dk)
		for i := 0; i < halfDim; i++ {
			freq := 1.0 / math.Pow(10000.0, 2.0*float64(i)/float64(dk))
			angle := freq * pos
			cosA := math.Cos(angle)
			sinA := math.Sin(angle)
			result[seqIdx][2*i] = row[2*i]*cosA - row[2*i+1]*sinA
			result[seqIdx][2*i+1] = row[2*i]*sinA + row[2*i+1]*cosA
		}
	}
	return floatMatrixToObject(result)
}

// fnSiluGate implements llm.silu_gate: fused SiLU activation + element-wise multiply.
// Computes silu(gate) * up. The core of SwiGLU used in modern FFNs.
func fnSiluGate(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
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
		return &object.List{Elements: []object.Object{}}
	}
	if len(gate[0]) != len(up[0]) {
		return errors.NewError("silu_gate: gate and up must have the same number of columns")
	}

	result := make([][]float64, len(gate))
	for i := range gate {
		result[i] = make([]float64, len(gate[i]))
		for j := range gate[i] {
			s := 1.0 / (1.0 + math.Exp(-gate[i][j]))
			result[i][j] = gate[i][j] * s * up[i][j]
		}
	}
	return floatMatrixToObject(result)
}

// fnAttention implements llm.attention: scaled dot-product attention.
// Computes softmax(Q @ K^T / sqrt(d_k)) @ V with optional causal masking.
func fnAttention(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 3, 4); err != nil {
		return err
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
	causal := true
	if len(args) == 4 {
		c, err := args[3].AsBool()
		if err != nil {
			return errors.NewTypeError("BOOLEAN", args[3].Type().String())
		}
		causal = c
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

	output := make([][]float64, qLen)
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
		weights := make([]float64, kvLen)
		for i, e := range exps {
			weights[i] = e * invSum
		}

		output[qi] = make([]float64, dk)
		for d := 0; d < dk; d++ {
			var val float64
			for ki := 0; ki < kvLen; ki++ {
				val += weights[ki] * v[ki][d]
			}
			output[qi][d] = val
		}
	}
	return floatMatrixToObject(output)
}

// fnLinear implements llm.linear: fused matmul + optional bias add.
// Computes x @ weight.T + bias where weight has shape (out, in).
func fnLinear(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 2, 3); err != nil {
		return err
	}
	x, errObj := toFloatMatrix(args[0], "linear", "x")
	if errObj != nil {
		return errObj
	}
	weight, errObj := toFloatMatrix(args[1], "linear", "weight")
	if errObj != nil {
		return errObj
	}
	var bias []float64
	if len(args) == 3 {
		bias, errObj = toFloatList(args[2], "linear", "bias")
		if errObj != nil {
			return errObj
		}
	}

	if len(x) == 0 || len(weight) == 0 {
		return errors.NewError("linear: inputs cannot be empty")
	}
	inFeatures := len(x[0])
	if len(weight[0]) != inFeatures {
		return errors.NewError("linear: weight columns (%d) must match x columns (%d)", len(weight[0]), inFeatures)
	}
	outFeatures := len(weight)

	result := make([][]float64, len(x))
	for i, row := range x {
		result[i] = make([]float64, outFeatures)
		for j := 0; j < outFeatures; j++ {
			var sum float64
			for l := 0; l < inFeatures; l++ {
				sum += row[l] * weight[j][l]
			}
			if bias != nil {
				sum += bias[j]
			}
			result[i][j] = sum
		}
	}
	return floatMatrixToObject(result)
}

// fnLinearRow implements llm.linear_row: last-row-only linear.
// Same as linear() but computes only the last row, saving (seq_len-1)*out_features ops.
func fnLinearRow(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 2, 3); err != nil {
		return err
	}
	x, errObj := toFloatMatrix(args[0], "linear_row", "x")
	if errObj != nil {
		return errObj
	}
	weight, errObj := toFloatMatrix(args[1], "linear_row", "weight")
	if errObj != nil {
		return errObj
	}
	var bias []float64
	if len(args) == 3 {
		bias, errObj = toFloatList(args[2], "linear_row", "bias")
		if errObj != nil {
			return errObj
		}
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
	return floatListToObject(result)
}

// indexedFloat pairs an element index with its float value for partial sorting.
type indexedFloat struct {
	index int
	value float64
}

// fnTopK implements llm.top_k: O(n) partial sort for top-k selection.
// Uses a maintained top-k buffer with binary search insertion.
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

// fnDequantizeQ8 implements llm.dequantize_q8: int8 dequantization with per-group scales.
// Compatible with the Q8_0 format used by llama.cpp/GGUF.
// Each value: float = int8_value * scale[group_index].
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
		return &object.List{Elements: []object.Object{}}
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
	return floatListToObject(result)
}

// fnDequantizeQ8_0 implements llm.dequantize_q8_0: native GGUF Q8_0 block dequantization.
// Takes raw block data (from fs.read_bytes) and number of groups.
// Each Q8_0 block is 34 bytes: 2-byte f16 scale + 32-byte int8 values.
// Returns n_groups * 32 dequantized floats.
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
	return floatListToObject(result)
}

// float16ToFloat64 converts IEEE 754 half-precision bits to float64.
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

// fnLinearQ8 implements llm.linear_q8: quantized matmul for Q8_0 weights.
// Computes x @ weight.T where weight is stored as raw Q8_0 blocks.
// Optimized: fast f16 decode, pre-multiplied scales, reduced allocations.
func fnLinearQ8(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 3); err != nil {
		return err
	}
	xList, ok := args[0].(*object.List)
	if !ok {
		return errors.NewTypeError("LIST", args[0].Type().String())
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
	seqLen := len(xList.Elements)
	if seqLen == 0 || outFeatures == 0 {
		return errors.NewError("linear_q8: inputs cannot be empty")
	}
	inFeatures := groupsPerRow * 32

	xFlat := make([]float64, seqLen*inFeatures)
	for xi, rowObj := range xList.Elements {
		row, ok := rowObj.(*object.List)
		if !ok {
			return errors.NewError("linear_q8: x must be a list of lists")
		}
		if len(row.Elements) != inFeatures {
			return errors.NewError("linear_q8: x columns (%d) must match in_features (%d)", len(row.Elements), inFeatures)
		}
		off := xi * inFeatures
		for i, el := range row.Elements {
			f, e := el.AsFloat()
			if e != nil {
				return errors.NewTypeError("INTEGER or FLOAT", el.Type().String())
			}
			xFlat[off+i] = f
		}
	}

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

	resultRows := make([]object.Object, seqLen)
	for xi := 0; xi < seqLen; xi++ {
		xOff := xi * inFeatures
		out := make([]float64, outFeatures)
		xRow := xFlat[xOff : xOff+inFeatures]
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
			out[j] = sum
		}
		rowElems := make([]object.Object, outFeatures)
		for j, v := range out {
			rowElems[j] = &object.Float{Value: v}
		}
		resultRows[xi] = &object.List{Elements: rowElems}
	}
	return &object.List{Elements: resultRows}
}

// fnLinearRowQ8 implements llm.linear_row_q8: last-row-only quantized matmul.
// Same as linear_q8 but computes only the last row of x, returning a vector.
func fnLinearRowQ8(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 3); err != nil {
		return err
	}
	xList, ok := args[0].(*object.List)
	if !ok {
		return errors.NewTypeError("LIST", args[0].Type().String())
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
	if len(xList.Elements) == 0 || outFeatures == 0 {
		return errors.NewError("linear_row_q8: inputs cannot be empty")
	}
	inFeatures := groupsPerRow * 32

	lastRowObj := xList.Elements[len(xList.Elements)-1]
	lastRowList, ok := lastRowObj.(*object.List)
	if !ok {
		return errors.NewError("linear_row_q8: x must be a list of lists")
	}
	if len(lastRowList.Elements) != inFeatures {
		return errors.NewError("linear_row_q8: x columns (%d) must match in_features (%d)", len(lastRowList.Elements), inFeatures)
	}
	lastRow := make([]float64, inFeatures)
	for i, el := range lastRowList.Elements {
		f, e := el.AsFloat()
		if e != nil {
			return errors.NewTypeError("INTEGER or FLOAT", el.Type().String())
		}
		lastRow[i] = f
	}

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

	result := make([]object.Object, outFeatures)
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
		result[j] = &object.Float{Value: sum}
	}
	return &object.List{Elements: result}
}
