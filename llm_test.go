package scriptlingllmlib

import (
	"context"
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func evalFloat(t *testing.T, obj object.Object) float64 {
	t.Helper()
	f, err := obj.AsFloat()
	if err != nil {
		t.Fatalf("expected float, got error: %v", err)
	}
	return f
}

func evalInt(t *testing.T, obj object.Object) int64 {
	t.Helper()
	i, err := obj.AsInt()
	if err != nil {
		t.Fatalf("expected int, got error: %v", err)
	}
	return i
}

func evalList(t *testing.T, obj object.Object) []object.Object {
	t.Helper()
	l, ok := obj.(*object.List)
	if !ok {
		t.Fatalf("expected list, got %s", obj.Type().String())
	}
	return l.Elements
}

func evalFloatList(t *testing.T, obj object.Object) []float64 {
	t.Helper()
	if fa, ok := obj.(*object.FloatArray); ok && !fa.Is2D() {
		result := make([]float64, len(fa.Data))
		copy(result, fa.Data)
		return result
	}
	elems := evalList(t, obj)
	vals := make([]float64, len(elems))
	for i, e := range elems {
		vals[i] = evalFloat(t, e)
	}
	return vals
}

func evalFloatMatrix(t *testing.T, obj object.Object) [][]float64 {
	t.Helper()
	if fa, ok := obj.(*object.FloatArray); ok && fa.Is2D() {
		rows := fa.Rows()
		result := make([][]float64, rows)
		for i := 0; i < rows; i++ {
			row := fa.Row(i)
			rowCopy := make([]float64, len(row))
			copy(rowCopy, row)
			result[i] = rowCopy
		}
		return result
	}
	rows := evalList(t, obj)
	mat := make([][]float64, len(rows))
	for i, r := range rows {
		mat[i] = evalFloatList(t, r)
	}
	return mat
}

func assertError(t *testing.T, obj object.Object, msg string) {
	t.Helper()
	if _, ok := obj.(*object.Error); !ok {
		t.Fatalf("expected error containing %q, got %s", msg, obj.Inspect())
	}
}

func assertEmptyListOrFloatArray(t *testing.T, obj object.Object) {
	t.Helper()
	if fa, ok := obj.(*object.FloatArray); ok {
		if len(fa.Data) != 0 {
			t.Fatalf("expected empty, got %s", obj.Inspect())
		}
		return
	}
	if l, ok := obj.(*object.List); ok {
		if len(l.Elements) != 0 {
			t.Fatalf("expected empty, got %s", obj.Inspect())
		}
		return
	}
	t.Fatalf("expected empty list or FloatArray, got %s", obj.Type().String())
}

func floatList(vals ...float64) object.Object {
	elems := make([]object.Object, len(vals))
	for i, v := range vals {
		elems[i] = &object.Float{Value: v}
	}
	return &object.List{Elements: elems}
}

func floatMatrix(rows ...[]float64) object.Object {
	elems := make([]object.Object, len(rows))
	for i, r := range rows {
		elems[i] = floatList(r...)
	}
	return &object.List{Elements: elems}
}

func intList(vals ...int64) object.Object {
	elems := make([]object.Object, len(vals))
	for i, v := range vals {
		elems[i] = object.NewInteger(v)
	}
	return &object.List{Elements: elems}
}

var ctx = context.Background()
var noopKwargs = object.Kwargs{}

func TestArgmax(t *testing.T) {
	result := fnArgmax(ctx, noopKwargs, floatList(1.0, 3.0, 2.0))
	if evalInt(t, result) != 1 {
		t.Errorf("argmax([1,3,2]) = %d, want 1", evalInt(t, result))
	}

	result = fnArgmax(ctx, noopKwargs, floatList(5.0))
	if evalInt(t, result) != 0 {
		t.Errorf("argmax([5]) = %d, want 0", evalInt(t, result))
	}

	assertError(t, fnArgmax(ctx, noopKwargs), "1 argument")
	assertError(t, fnArgmax(ctx, noopKwargs, floatList()), "empty")
	assertError(t, fnArgmax(ctx, noopKwargs, &object.String{Value: "x"}), "LIST")
}

func TestArgmin(t *testing.T) {
	result := fnArgmin(ctx, noopKwargs, floatList(3.0, 1.0, 2.0))
	if evalInt(t, result) != 1 {
		t.Errorf("argmin([3,1,2]) = %d, want 1", evalInt(t, result))
	}

	result = fnArgmin(ctx, noopKwargs, floatList(-1.0, 0.0, 1.0))
	if evalInt(t, result) != 0 {
		t.Errorf("argmin([-1,0,1]) = %d, want 0", evalInt(t, result))
	}

	assertError(t, fnArgmin(ctx, noopKwargs, floatList()), "empty")
}

func TestTopk(t *testing.T) {
	result := fnTopk(ctx, noopKwargs, floatList(1.0, 5.0, 3.0, 4.0, 2.0), object.NewInteger(3))
	elems := evalList(t, result)
	if len(elems) != 3 {
		t.Fatalf("topk len = %d, want 3", len(elems))
	}

	pairs := make([]struct {
		idx int64
		val float64
	}, 3)
	for i, e := range elems {
		pair := evalList(t, e)
		pairs[i].idx = evalInt(t, pair[0])
		pairs[i].val = evalFloat(t, pair[1])
	}
	if pairs[0].idx != 1 || pairs[0].val != 5.0 {
		t.Errorf("topk[0] = (%d, %f), want (1, 5.0)", pairs[0].idx, pairs[0].val)
	}
	if pairs[1].idx != 3 || pairs[1].val != 4.0 {
		t.Errorf("topk[1] = (%d, %f), want (3, 4.0)", pairs[1].idx, pairs[1].val)
	}
	if pairs[2].idx != 2 || pairs[2].val != 3.0 {
		t.Errorf("topk[2] = (%d, %f), want (2, 3.0)", pairs[2].idx, pairs[2].val)
	}

	result = fnTopk(ctx, noopKwargs, floatList(1.0, 2.0), object.NewInteger(10))
	elems = evalList(t, result)
	if len(elems) != 2 {
		t.Errorf("topk clamped len = %d, want 2", len(elems))
	}

	assertError(t, fnTopk(ctx, noopKwargs, floatList(1.0), object.NewInteger(0)), "positive")
}

func TestClip(t *testing.T) {
	result := fnClip(ctx, noopKwargs, floatList(-2.0, 0.5, 3.0, 1.0), &object.Float{Value: -1.0}, &object.Float{Value: 2.0})
	vals := evalFloatList(t, result)
	expected := []float64{-1.0, 0.5, 2.0, 1.0}
	for i, v := range vals {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("clip[%d] = %f, want %f", i, v, expected[i])
		}
	}

	result = fnClip(ctx, noopKwargs, &object.Float{Value: 5.0}, &object.Float{Value: 0.0}, &object.Float{Value: 3.0})
	if evalFloat(t, result) != 3.0 {
		t.Errorf("clip(5, 0, 3) = %f, want 3.0", evalFloat(t, result))
	}

	assertError(t, fnClip(ctx, noopKwargs, &object.Float{Value: 1.0}, &object.Float{Value: 5.0}, &object.Float{Value: 3.0}), "lo must be <= hi")
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.5},
		{100.0, 1.0},
		{-100.0, 0.0},
	}
	for _, tt := range tests {
		result := fnSigmoid(ctx, noopKwargs, &object.Float{Value: tt.input})
		got := evalFloat(t, result)
		if math.Abs(got-tt.expected) > 1e-6 {
			t.Errorf("sigmoid(%f) = %f, want %f", tt.input, got, tt.expected)
		}
	}
}

func TestRelu(t *testing.T) {
	if evalFloat(t, fnRelu(ctx, noopKwargs, &object.Float{Value: -1.0})) != 0.0 {
		t.Error("relu(-1) != 0")
	}
	if evalFloat(t, fnRelu(ctx, noopKwargs, &object.Float{Value: 0.0})) != 0.0 {
		t.Error("relu(0) != 0")
	}
	if evalFloat(t, fnRelu(ctx, noopKwargs, &object.Float{Value: 5.0})) != 5.0 {
		t.Error("relu(5) != 5")
	}
}

func TestGelu(t *testing.T) {
	result := fnGelu(ctx, noopKwargs, &object.Float{Value: 0.0})
	if evalFloat(t, result) != 0.0 {
		t.Errorf("gelu(0) = %f, want 0", evalFloat(t, result))
	}
	result = fnGelu(ctx, noopKwargs, &object.Float{Value: 1.0})
	got := evalFloat(t, result)
	expected := 0.5 * 1.0 * (1.0 + math.Erf(1.0/math.Sqrt(2.0)))
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("gelu(1) = %f, want %f", got, expected)
	}
}

func TestSilu(t *testing.T) {
	result := fnSilu(ctx, noopKwargs, &object.Float{Value: 0.0})
	if evalFloat(t, result) != 0.0 {
		t.Errorf("silu(0) = %f, want 0", evalFloat(t, result))
	}
	x := 2.0
	result = fnSilu(ctx, noopKwargs, &object.Float{Value: x})
	expected := x * (1.0 / (1.0 + math.Exp(-x)))
	if math.Abs(evalFloat(t, result)-expected) > 1e-10 {
		t.Errorf("silu(2) = %f, want %f", evalFloat(t, result), expected)
	}
}

func TestVecAdd(t *testing.T) {
	result := fnVecAdd(ctx, noopKwargs, floatList(1.0, 2.0, 3.0), floatList(4.0, 5.0, 6.0))
	vals := evalFloatList(t, result)
	expected := []float64{5.0, 7.0, 9.0}
	for i, v := range vals {
		if v != expected[i] {
			t.Errorf("vec_add[%d] = %f, want %f", i, v, expected[i])
		}
	}
	assertError(t, fnVecAdd(ctx, noopKwargs, floatList(1.0), floatList(1.0, 2.0)), "same length")
}

func TestVecSub(t *testing.T) {
	result := fnVecSub(ctx, noopKwargs, floatList(5.0, 3.0, 1.0), floatList(1.0, 2.0, 3.0))
	vals := evalFloatList(t, result)
	expected := []float64{4.0, 1.0, -2.0}
	for i, v := range vals {
		if v != expected[i] {
			t.Errorf("vec_sub[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

func TestVecMul(t *testing.T) {
	result := fnVecMul(ctx, noopKwargs, floatList(2.0, 3.0), floatList(4.0, 5.0))
	vals := evalFloatList(t, result)
	if vals[0] != 8.0 || vals[1] != 15.0 {
		t.Errorf("vec_mul = %v, want [8, 15]", vals)
	}
}

func TestVecScale(t *testing.T) {
	result := fnVecScale(ctx, noopKwargs, floatList(1.0, 2.0, 3.0), &object.Float{Value: 2.0})
	vals := evalFloatList(t, result)
	expected := []float64{2.0, 4.0, 6.0}
	for i, v := range vals {
		if v != expected[i] {
			t.Errorf("vec_scale[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

func TestVecApply(t *testing.T) {
	input := floatList(-1.0, 0.0, 1.0, 2.0)

	result := fnVecApply(ctx, noopKwargs, input, &object.String{Value: "relu"})
	vals := evalFloatList(t, result)
	expected := []float64{0.0, 0.0, 1.0, 2.0}
	for i, v := range vals {
		if v != expected[i] {
			t.Errorf("vec_apply(relu)[%d] = %f, want %f", i, v, expected[i])
		}
	}

	result = fnVecApply(ctx, noopKwargs, input, &object.String{Value: "sigmoid"})
	vals = evalFloatList(t, result)
	for _, v := range vals {
		if v < 0 || v > 1 {
			t.Errorf("vec_apply(sigmoid) value %f out of [0,1]", v)
		}
	}

	assertError(t, fnVecApply(ctx, noopKwargs, floatList(1.0), &object.String{Value: "unknown"}), "unknown function")
}

func TestRmsNorm(t *testing.T) {
	x := floatMatrix([]float64{0.5, -0.3, 0.8}, []float64{0.1, 0.2, -0.4})
	w := floatList(1.0, 1.0, 1.0)

	result := fnRmsNorm(ctx, noopKwargs, x, w)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 2 || len(mat[0]) != 3 {
		t.Fatalf("rms_norm shape = %dx%d, want 2x3", len(mat), len(mat[0]))
	}

	row := []float64{0.5, -0.3, 0.8}
	ss := (0.25 + 0.09 + 0.64) / 3.0
	inv := 1.0 / math.Sqrt(ss+1e-5)
	for j, v := range mat[0] {
		expected := row[j] * inv * 1.0
		if math.Abs(v-expected) > 1e-10 {
			t.Errorf("rms_norm[0][%d] = %f, want %f", j, v, expected)
		}
	}

	result = fnRmsNorm(ctx, noopKwargs, x, w, &object.Float{Value: 1e-6})
	mat2 := evalFloatMatrix(t, result)
	if len(mat2) != 2 {
		t.Error("rms_norm with eps failed")
	}

	assertError(t, fnRmsNorm(ctx, noopKwargs, floatMatrix([]float64{1.0, 2.0}), floatList(1.0)), "columns")
}

func TestRope(t *testing.T) {
	x := floatMatrix([]float64{1.0, 0.0, 0.0, 1.0})
	result := fnRope(ctx, noopKwargs, x)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 4 {
		t.Fatalf("rope shape = %dx%d, want 1x4", len(mat), len(mat[0]))
	}

	dk := 4.0
	for i := 0; i < 2; i++ {
		freq := 1.0 / math.Pow(10000.0, 2.0*float64(i)/dk)
		angle := freq * 0.0
		cosA := math.Cos(angle)
		sinA := math.Sin(angle)
		expectedEven := 1.0*cosA - 0.0*sinA
		expectedOdd := 1.0*sinA + 0.0*cosA
		if i == 0 {
			if math.Abs(mat[0][0]-expectedEven) > 1e-10 {
				t.Errorf("rope[0][0] = %f, want %f", mat[0][0], expectedEven)
			}
			if math.Abs(mat[0][1]-expectedOdd) > 1e-10 {
				t.Errorf("rope[0][1] = %f, want %f", mat[0][1], expectedOdd)
			}
		}
	}

	result = fnRope(ctx, noopKwargs, x, object.NewInteger(5))
	mat = evalFloatMatrix(t, result)
	if len(mat) != 1 {
		t.Error("rope with start_pos failed")
	}

	oddDim := floatMatrix([]float64{1.0, 0.0, 1.0})
	assertError(t, fnRope(ctx, noopKwargs, oddDim), "even")
}

func TestSiluGate(t *testing.T) {
	gate := floatMatrix([]float64{1.0, -1.0}, []float64{0.0, 2.0})
	up := floatMatrix([]float64{1.0, 1.0}, []float64{1.0, 1.0})

	result := fnSiluGate(ctx, noopKwargs, gate, up)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 2 || len(mat[0]) != 2 {
		t.Fatalf("silu_gate shape = %dx%d, want 2x2", len(mat), len(mat[0]))
	}

	sig1 := 1.0 / (1.0 + math.Exp(-1.0))
	expected00 := 1.0 * sig1 * 1.0
	if math.Abs(mat[0][0]-expected00) > 1e-10 {
		t.Errorf("silu_gate[0][0] = %f, want %f", mat[0][0], expected00)
	}

	sigNeg1 := 1.0 / (1.0 + math.Exp(1.0))
	expected01 := (-1.0) * sigNeg1 * 1.0
	if math.Abs(mat[0][1]-expected01) > 1e-10 {
		t.Errorf("silu_gate[0][1] = %f, want %f", mat[0][1], expected01)
	}

	if mat[1][0] != 0.0 {
		t.Errorf("silu_gate[1][0] = %f, want 0", mat[1][0])
	}
}

func TestAttention(t *testing.T) {
	q := floatMatrix([]float64{1.0, 0.0})
	k := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})
	v := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})

	result := fnAttention(ctx, noopKwargs, q, k, v)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 2 {
		t.Fatalf("attention shape = %dx%d, want 1x2", len(mat), len(mat[0]))
	}

	if mat[0][0] < 0.5 {
		t.Errorf("attention[0][0] = %f, expected dominant weight on position 0", mat[0][0])
	}
	if mat[0][0]+mat[0][1] < 0.99 || mat[0][0]+mat[0][1] > 1.01 {
		t.Errorf("attention outputs should sum to ~1.0: got %f", mat[0][0]+mat[0][1])
	}

	result = fnAttention(ctx, noopKwargs, q, k, v, object.NewBoolean(false))
	mat = evalFloatMatrix(t, result)
	if len(mat) != 1 {
		t.Error("non-causal attention failed")
	}
}

func TestAttentionCausal(t *testing.T) {
	q := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})
	k := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})
	v := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})

	result := fnAttention(ctx, noopKwargs, q, k, v, object.NewBoolean(true))
	mat := evalFloatMatrix(t, result)

	if mat[0][0] < 0.99 {
		t.Errorf("causal attention row 0 should attend only to pos 0: got %f", mat[0][0])
	}
	if mat[1][1] < 0.49 {
		t.Errorf("causal attention row 1 should attend to both: got %f", mat[1][1])
	}
}

func TestLinear(t *testing.T) {
	x := floatMatrix([]float64{1.0, 2.0})
	weight := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})

	result := fnLinear(ctx, noopKwargs, x, weight)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 2 {
		t.Fatalf("linear shape = %dx%d, want 1x2", len(mat), len(mat[0]))
	}
	if mat[0][0] != 1.0 || mat[0][1] != 2.0 {
		t.Errorf("linear identity = %v, want [1, 2]", mat[0])
	}

	bias := floatList(10.0, 20.0)
	result = fnLinear(ctx, noopKwargs, x, weight, bias)
	mat = evalFloatMatrix(t, result)
	if mat[0][0] != 11.0 || mat[0][1] != 22.0 {
		t.Errorf("linear with bias = %v, want [11, 22]", mat[0])
	}
}

func TestLinearRow(t *testing.T) {
	x := floatMatrix([]float64{1.0, 2.0}, []float64{3.0, 4.0})
	weight := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})

	result := fnLinearRow(ctx, noopKwargs, x, weight)
	vals := evalFloatList(t, result)
	if vals[0] != 3.0 || vals[1] != 4.0 {
		t.Errorf("linear_row = %v, want [3, 4]", vals)
	}

	bias := floatList(1.0, 1.0)
	result = fnLinearRow(ctx, noopKwargs, x, weight, bias)
	vals = evalFloatList(t, result)
	if vals[0] != 4.0 || vals[1] != 5.0 {
		t.Errorf("linear_row with bias = %v, want [4, 5]", vals)
	}
}

func TestTopK(t *testing.T) {
	result := fnTopK(ctx, noopKwargs, floatList(0.1, 0.5, 0.3, 0.9, 0.7), object.NewInteger(3))
	elems := evalList(t, result)
	if len(elems) != 3 {
		t.Fatalf("top_k len = %d, want 3", len(elems))
	}

	first := evalList(t, elems[0])
	if evalInt(t, first[0]) != 3 {
		t.Errorf("top_k[0].idx = %d, want 3 (value 0.9)", evalInt(t, first[0]))
	}
	if math.Abs(evalFloat(t, first[1])-0.9) > 1e-10 {
		t.Errorf("top_k[0].val = %f, want 0.9", evalFloat(t, first[1]))
	}
}

func TestDequantizeQ8(t *testing.T) {
	data := intList(10, -5, 20, 15)
	scales := floatList(0.1, 0.2)

	result := fnDequantizeQ8(ctx, noopKwargs, data, scales, object.NewInteger(2))
	vals := evalFloatList(t, result)
	expected := []float64{1.0, -0.5, 4.0, 3.0}
	for i, v := range vals {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("dequantize_q8[%d] = %f, want %f", i, v, expected[i])
		}
	}

	assertError(t, fnDequantizeQ8(ctx, noopKwargs, intList(1), floatList(0.1), object.NewInteger(0)), "positive")
}

func TestConcatRows(t *testing.T) {
	a := floatMatrix([]float64{1.0, 2.0})
	b := floatMatrix([]float64{3.0, 4.0}, []float64{5.0, 6.0})

	result := fnConcatRows(ctx, noopKwargs, a, b)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 3 {
		t.Fatalf("concat_rows len = %d, want 3", len(mat))
	}
	if mat[0][0] != 1.0 || mat[1][0] != 3.0 || mat[2][0] != 5.0 {
		t.Errorf("concat_rows = %v, unexpected", mat)
	}

	mismatchA := floatMatrix([]float64{1.0, 2.0})
	mismatchB := floatMatrix([]float64{3.0, 4.0, 5.0})
	assertError(t, fnConcatRows(ctx, noopKwargs, mismatchA, mismatchB), "columns")
}

func TestSliceRows(t *testing.T) {
	m := floatMatrix([]float64{1.0, 2.0}, []float64{3.0, 4.0}, []float64{5.0, 6.0}, []float64{7.0, 8.0})

	result := fnSliceRows(ctx, noopKwargs, m, object.NewInteger(1), object.NewInteger(3))
	mat := evalFloatMatrix(t, result)
	if len(mat) != 2 {
		t.Fatalf("slice_rows len = %d, want 2", len(mat))
	}
	if mat[0][0] != 3.0 || mat[1][0] != 5.0 {
		t.Errorf("slice_rows = %v, unexpected", mat)
	}

	result = fnSliceRows(ctx, noopKwargs, m, object.NewInteger(10), object.NewInteger(20))
	mat = evalFloatMatrix(t, result)
	if len(mat) != 0 {
		t.Errorf("out-of-range slice should be empty")
	}
}

func TestFlatten(t *testing.T) {
	m := floatMatrix([]float64{1.0, 2.0}, []float64{3.0, 4.0})
	result := fnFlatten(ctx, noopKwargs, m)
	vals := evalFloatList(t, result)
	expected := []float64{1.0, 2.0, 3.0, 4.0}
	for i, v := range vals {
		if v != expected[i] {
			t.Errorf("flatten[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

func makeQ80Group(scaleBits uint16, values []int8) []byte {
	b := make([]byte, 34)
	b[0] = byte(scaleBits)
	b[1] = byte(scaleBits >> 8)
	for i, v := range values {
		b[2+i] = byte(v)
	}
	return b
}

func makeQ80Raw(groups ...[]byte) *object.String {
	raw := make([]byte, 0, len(groups)*34)
	for _, g := range groups {
		raw = append(raw, g...)
	}
	return &object.String{Value: string(raw)}
}

func TestFloat16ToFloat64(t *testing.T) {
	tests := []struct {
		bits     uint16
		expected float64
		nan      bool
		inf      int
		name     string
	}{
		{0x0000, 0.0, false, 0, "positive zero"},
		{0x8000, 0.0, false, 0, "negative zero"},
		{0x3C00, 1.0, false, 0, "1.0"},
		{0xBC00, -1.0, false, 0, "-1.0"},
		{0x4000, 2.0, false, 0, "2.0"},
		{0x3800, 0.5, false, 0, "0.5"},
		{0x4400, 4.0, false, 0, "4.0"},
		{0xC400, -4.0, false, 0, "-4.0"},
		{0x0001, math.Ldexp(1.0/1024.0, -14), false, 0, "subnormal"},
		{0x7C00, 0, false, 1, "positive infinity"},
		{0xFC00, 0, false, -1, "negative infinity"},
		{0x7C01, 0, true, 0, "NaN"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := float16ToFloat64(tt.bits)
			if tt.nan {
				if !math.IsNaN(got) {
					t.Errorf("float16ToFloat64(0x%04X) = %f, want NaN", tt.bits, got)
				}
			} else if tt.inf != 0 {
				if !math.IsInf(got, tt.inf) {
					t.Errorf("float16ToFloat64(0x%04X) = %f, want Inf(%d)", tt.bits, got, tt.inf)
				}
			} else {
				if math.Abs(got-tt.expected) > 1e-10 {
					t.Errorf("float16ToFloat64(0x%04X) = %f, want %f", tt.bits, got, tt.expected)
				}
			}
		})
	}
}

func TestDequantizeQ80(t *testing.T) {
	values := make([]int8, 32)
	for i := range values {
		values[i] = int8(i + 1)
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, values))

	result := fnDequantizeQ8_0(ctx, noopKwargs, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 32 {
		t.Fatalf("dequantize_q8_0 len = %d, want 32", len(vals))
	}
	for i, v := range vals {
		if math.Abs(v-float64(i+1)) > 1e-10 {
			t.Errorf("dequantize_q8_0[%d] = %f, want %d", i, v, i+1)
		}
	}

	values2 := make([]int8, 32)
	for i := range values2 {
		values2[i] = -int8(i + 1)
	}
	raw2 := makeQ80Raw(makeQ80Group(0x3C00, values), makeQ80Group(0x4000, values2))
	result2 := fnDequantizeQ8_0(ctx, noopKwargs, raw2, object.NewInteger(2))
	vals2 := evalFloatList(t, result2)
	if len(vals2) != 64 {
		t.Fatalf("dequantize_q8_0 2 groups len = %d, want 64", len(vals2))
	}
	for i := 0; i < 32; i++ {
		if math.Abs(vals2[32+i]-float64(-(i+1))*2.0) > 1e-10 {
			t.Errorf("dequantize_q8_0 group1[%d] = %f, want %f", i, vals2[32+i], float64(-(i+1))*2.0)
		}
	}
}

func TestDequantizeQ80Errors(t *testing.T) {
	assertError(t, fnDequantizeQ8_0(ctx, noopKwargs), "2 arguments")
	assertError(t, fnDequantizeQ8_0(ctx, noopKwargs, floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnDequantizeQ8_0(ctx, noopKwargs, &object.String{Value: "x"}, &object.String{Value: "x"}), "INTEGER")
	assertError(t, fnDequantizeQ8_0(ctx, noopKwargs, &object.String{Value: "x"}, object.NewInteger(0)), "positive")
	assertError(t, fnDequantizeQ8_0(ctx, noopKwargs, &object.String{Value: "short"}, object.NewInteger(1)), "too short")
}

func TestLinearQ8(t *testing.T) {
	ones := make([]float64, 32)
	for i := range ones {
		ones[i] = 1.0
	}
	x := floatMatrix(ones)
	qValues := make([]int8, 32)
	for i := range qValues {
		qValues[i] = 1
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))

	result := fnLinearQ8(ctx, noopKwargs, x, raw, object.NewInteger(1))
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 1 {
		t.Fatalf("linear_q8 shape = %dx%d, want 1x1", len(mat), len(mat[0]))
	}
	if math.Abs(mat[0][0]-32.0) > 1e-10 {
		t.Errorf("linear_q8 = %f, want 32.0", mat[0][0])
	}

	twos := make([]float64, 32)
	for i := range twos {
		twos[i] = 2.0
	}
	x2 := floatMatrix(ones, twos)
	raw2 := makeQ80Raw(makeQ80Group(0x3C00, qValues), makeQ80Group(0x4000, qValues))

	result2 := fnLinearQ8(ctx, noopKwargs, x2, raw2, object.NewInteger(1))
	mat2 := evalFloatMatrix(t, result2)
	if len(mat2) != 2 || len(mat2[0]) != 2 {
		t.Fatalf("linear_q8 2x2 shape = %dx%d, want 2x2", len(mat2), len(mat2[0]))
	}
	if math.Abs(mat2[0][0]-32.0) > 1e-10 {
		t.Errorf("linear_q8[0][0] = %f, want 32.0", mat2[0][0])
	}
	if math.Abs(mat2[1][1]-128.0) > 1e-10 {
		t.Errorf("linear_q8[1][1] = %f, want 128.0", mat2[1][1])
	}
}

func TestLinearQ8Errors(t *testing.T) {
	assertError(t, fnLinearQ8(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearQ8(ctx, noopKwargs, &object.String{Value: "x"}, &object.String{Value: "x"}, object.NewInteger(1)), "LIST or FLOAT_ARRAY")
	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), &object.String{Value: "x"}, &object.String{Value: "x"}), "INTEGER")
	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), &object.String{Value: "x"}, object.NewInteger(0)), "positive")

	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix(), makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "empty")
	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), &object.String{Value: ""}, object.NewInteger(1)), "empty")

	xNotMatrix := &object.List{Elements: []object.Object{&object.String{Value: "x"}}}
	assertError(t, fnLinearQ8(ctx, noopKwargs, xNotMatrix, makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "list of lists")
	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix([]float64{1.0}), makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "columns")

	strElems := make([]object.Object, 32)
	for i := 0; i < 31; i++ {
		strElems[i] = &object.Float{Value: 1.0}
	}
	strElems[31] = &object.String{Value: "x"}
	xWithStr := &object.List{Elements: []object.Object{&object.List{Elements: strElems}}}
	assertError(t, fnLinearQ8(ctx, noopKwargs, xWithStr, makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "INTEGER or FLOAT")
}

func TestLinearRowQ8(t *testing.T) {
	ones := make([]float64, 32)
	for i := range ones {
		ones[i] = 1.0
	}
	twos := make([]float64, 32)
	for i := range twos {
		twos[i] = 2.0
	}
	x := floatMatrix(ones, twos)
	qValues := make([]int8, 32)
	for i := range qValues {
		qValues[i] = 1
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))

	result := fnLinearRowQ8(ctx, noopKwargs, x, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 1 {
		t.Fatalf("linear_row_q8 len = %d, want 1", len(vals))
	}
	if math.Abs(vals[0]-64.0) > 1e-10 {
		t.Errorf("linear_row_q8 = %f, want 64.0", vals[0])
	}
}

func TestActivationErrors(t *testing.T) {
	for name, fn := range map[string]func(context.Context, object.Kwargs, ...object.Object) object.Object{
		"sigmoid": fnSigmoid, "relu": fnRelu, "gelu": fnGelu, "silu": fnSilu,
	} {
		t.Run(name, func(t *testing.T) {
			assertError(t, fn(ctx, noopKwargs), "1 argument")
			assertError(t, fn(ctx, noopKwargs, &object.String{Value: "x"}), "INTEGER or FLOAT")
		})
	}
}

func TestVecOpErrors(t *testing.T) {
	assertError(t, fnVecAdd(ctx, noopKwargs), "2 arguments")
	assertError(t, fnVecAdd(ctx, noopKwargs, &object.String{Value: "x"}, floatList(1.0)), "LIST")
	assertError(t, fnVecAdd(ctx, noopKwargs, floatList(1.0), &object.String{Value: "x"}), "LIST")
	assertError(t, fnVecSub(ctx, noopKwargs), "2 arguments")
	assertError(t, fnVecSub(ctx, noopKwargs, floatList(1.0), floatList(1.0, 2.0)), "same length")
	assertError(t, fnVecSub(ctx, noopKwargs, &object.String{Value: "x"}, floatList(1.0)), "LIST")
	assertError(t, fnVecMul(ctx, noopKwargs), "2 arguments")
	assertError(t, fnVecMul(ctx, noopKwargs, floatList(1.0), floatList(1.0, 2.0)), "same length")
	assertError(t, fnVecMul(ctx, noopKwargs, &object.String{Value: "x"}, floatList(1.0)), "LIST")
	assertError(t, fnVecScale(ctx, noopKwargs), "2 arguments")
	assertError(t, fnVecScale(ctx, noopKwargs, &object.String{Value: "x"}, &object.Float{Value: 1.0}), "LIST")
	assertError(t, fnVecScale(ctx, noopKwargs, floatList(1.0), &object.String{Value: "x"}), "INTEGER or FLOAT")
}

func TestVecApplyGeluSilu(t *testing.T) {
	input := floatList(0.0, 1.0, -1.0)
	result := fnVecApply(ctx, noopKwargs, input, &object.String{Value: "gelu"})
	vals := evalFloatList(t, result)
	expected1 := 0.5 * 1.0 * (1.0 + math.Erf(1.0/math.Sqrt(2.0)))
	if math.Abs(vals[1]-expected1) > 1e-10 {
		t.Errorf("vec_apply(gelu)[1] = %f, want %f", vals[1], expected1)
	}
	result = fnVecApply(ctx, noopKwargs, input, &object.String{Value: "silu"})
	vals = evalFloatList(t, result)
	expectedSilu1 := 1.0 / (1.0 + math.Exp(-1.0))
	if math.Abs(vals[1]-expectedSilu1) > 1e-10 {
		t.Errorf("vec_apply(silu)[1] = %f, want %f", vals[1], expectedSilu1)
	}
}

func TestVecApplyErrors(t *testing.T) {
	assertError(t, fnVecApply(ctx, noopKwargs), "2 arguments")
	assertError(t, fnVecApply(ctx, noopKwargs, floatList(1.0), object.NewInteger(0)), "STRING")
	assertError(t, fnVecApply(ctx, noopKwargs, &object.String{Value: "x"}, &object.String{Value: "relu"}), "LIST")
}

func TestClipErrors(t *testing.T) {
	assertError(t, fnClip(ctx, noopKwargs), "3 arguments")
	assertError(t, fnClip(ctx, noopKwargs, &object.String{Value: "x"}, &object.Float{Value: 0.0}, &object.Float{Value: 1.0}), "INTEGER, FLOAT, or LIST")
	assertError(t, fnClip(ctx, noopKwargs, floatList(1.0), &object.String{Value: "x"}, &object.Float{Value: 1.0}), "INTEGER or FLOAT")
	assertError(t, fnClip(ctx, noopKwargs, floatList(1.0), &object.Float{Value: 0.0}, &object.String{Value: "x"}), "INTEGER or FLOAT")
	strInList := &object.List{Elements: []object.Object{&object.String{Value: "x"}}}
	assertError(t, fnClip(ctx, noopKwargs, strInList, &object.Float{Value: 0.0}, &object.Float{Value: 1.0}), "INTEGER or FLOAT")
}

func TestArgminErrors(t *testing.T) {
	assertError(t, fnArgmin(ctx, noopKwargs), "1 argument")
	assertError(t, fnArgmin(ctx, noopKwargs, &object.String{Value: "x"}), "LIST")
}

func TestTopkErrors(t *testing.T) {
	assertError(t, fnTopk(ctx, noopKwargs), "2 arguments")
	assertError(t, fnTopk(ctx, noopKwargs, &object.String{Value: "x"}, object.NewInteger(1)), "LIST")
	assertError(t, fnTopk(ctx, noopKwargs, floatList(1.0), &object.String{Value: "x"}), "INTEGER")
}

func TestTopKErrors(t *testing.T) {
	assertError(t, fnTopK(ctx, noopKwargs), "2 arguments")
	assertError(t, fnTopK(ctx, noopKwargs, &object.String{Value: "x"}, object.NewInteger(1)), "LIST")
	assertError(t, fnTopK(ctx, noopKwargs, floatList(1.0), &object.String{Value: "x"}), "INTEGER")
	assertError(t, fnTopK(ctx, noopKwargs, floatList(1.0), object.NewInteger(0)), "positive")
}

func TestRmsNormErrors(t *testing.T) {
	assertError(t, fnRmsNorm(ctx, noopKwargs), "2 arguments")
	assertError(t, fnRmsNorm(ctx, noopKwargs, &object.String{Value: "x"}, floatList(1.0)), "LIST")
	assertError(t, fnRmsNorm(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnRmsNorm(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), &object.String{Value: "x"}), "INTEGER or FLOAT")
}

func TestRopeErrors(t *testing.T) {
	assertError(t, fnRope(ctx, noopKwargs), "1 argument")
	assertError(t, fnRope(ctx, noopKwargs, &object.String{Value: "x"}), "LIST")
	assertError(t, fnRope(ctx, noopKwargs, floatMatrix([]float64{1.0, 0.0}), &object.String{Value: "x"}), "INTEGER")
	result := fnRope(ctx, noopKwargs, &object.List{Elements: []object.Object{}})
	assertEmptyListOrFloatArray(t, result)
}

func TestSiluGateErrors(t *testing.T) {
	assertError(t, fnSiluGate(ctx, noopKwargs), "2 arguments")
	assertError(t, fnSiluGate(ctx, noopKwargs, &object.String{Value: "x"}, floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnSiluGate(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnSiluGate(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}, []float64{1.0})), "rows")
	assertError(t, fnSiluGate(ctx, noopKwargs, floatMatrix([]float64{1.0, 2.0}), floatMatrix([]float64{1.0})), "columns")
	empty := &object.List{Elements: []object.Object{}}
	result := fnSiluGate(ctx, noopKwargs, empty, empty)
	assertEmptyListOrFloatArray(t, result)
}

func TestAttentionErrors(t *testing.T) {
	assertError(t, fnAttention(ctx, noopKwargs), "3 arguments")
	assertError(t, fnAttention(ctx, noopKwargs, &object.String{Value: "x"}, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnAttention(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnAttention(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	empty := floatMatrix()
	assertError(t, fnAttention(ctx, noopKwargs, empty, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0})), "empty")
	assertError(t, fnAttention(ctx, noopKwargs, floatMatrix([]float64{1.0}), empty, floatMatrix([]float64{1.0})), "empty")
	assertError(t, fnAttention(ctx, noopKwargs, floatMatrix([]float64{1.0, 2.0}), floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0})), "inner dimension")
	assertError(t, fnAttention(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}, []float64{1.0})), "same number of rows")
}

func TestLinearErrors(t *testing.T) {
	assertError(t, fnLinear(ctx, noopKwargs), "2 arguments")
	assertError(t, fnLinear(ctx, noopKwargs, &object.String{Value: "x"}, floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnLinear(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnLinear(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnLinear(ctx, noopKwargs, floatMatrix(), floatMatrix([]float64{1.0})), "empty")
	assertError(t, fnLinear(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix()), "empty")
	assertError(t, fnLinear(ctx, noopKwargs, floatMatrix([]float64{1.0, 2.0}), floatMatrix([]float64{1.0})), "columns")
}

func TestLinearRowErrors(t *testing.T) {
	assertError(t, fnLinearRow(ctx, noopKwargs), "2 arguments")
	assertError(t, fnLinearRow(ctx, noopKwargs, &object.String{Value: "x"}, floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnLinearRow(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnLinearRow(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnLinearRow(ctx, noopKwargs, floatMatrix(), floatMatrix([]float64{1.0})), "empty")
	assertError(t, fnLinearRow(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix()), "empty")
	assertError(t, fnLinearRow(ctx, noopKwargs, floatMatrix([]float64{1.0, 2.0}), floatMatrix([]float64{1.0})), "columns")
}

func TestConcatRowsErrors(t *testing.T) {
	assertError(t, fnConcatRows(ctx, noopKwargs), "2 arguments")
	assertError(t, fnConcatRows(ctx, noopKwargs, &object.String{Value: "x"}, floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnConcatRows(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	result := fnConcatRows(ctx, noopKwargs, floatMatrix(), floatMatrix())
	assertEmptyListOrFloatArray(t, result)
}

func TestSliceRowsMore(t *testing.T) {
	assertError(t, fnSliceRows(ctx, noopKwargs), "3 arguments")
	assertError(t, fnSliceRows(ctx, noopKwargs, &object.String{Value: "x"}, object.NewInteger(0), object.NewInteger(1)), "LIST")
	assertError(t, fnSliceRows(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, object.NewInteger(1)), "INTEGER")
	assertError(t, fnSliceRows(ctx, noopKwargs, floatMatrix([]float64{1.0}), object.NewInteger(0), &object.String{Value: "x"}), "INTEGER")
	m := floatMatrix([]float64{1.0}, []float64{2.0}, []float64{3.0})
	result := fnSliceRows(ctx, noopKwargs, m, object.NewInteger(2), object.NewInteger(2))
	assertEmptyListOrFloatArray(t, result)
	result = fnSliceRows(ctx, noopKwargs, m, object.NewInteger(-1), object.NewInteger(1))
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || mat[0][0] != 1.0 {
		t.Errorf("slice_rows negative start = %v, want [[1.0]]", mat)
	}
}

func TestFlattenErrors(t *testing.T) {
	assertError(t, fnFlatten(ctx, noopKwargs), "1 argument")
	assertError(t, fnFlatten(ctx, noopKwargs, &object.String{Value: "x"}), "LIST")
}

func TestDequantizeQ8More(t *testing.T) {
	assertError(t, fnDequantizeQ8(ctx, noopKwargs), "3 arguments")
	assertError(t, fnDequantizeQ8(ctx, noopKwargs, &object.String{Value: "x"}, floatList(0.1), object.NewInteger(2)), "LIST")
	assertError(t, fnDequantizeQ8(ctx, noopKwargs, intList(1), &object.String{Value: "x"}, object.NewInteger(2)), "LIST")
	assertError(t, fnDequantizeQ8(ctx, noopKwargs, intList(1), floatList(0.1), &object.String{Value: "x"}), "INTEGER")
	assertError(t, fnDequantizeQ8(ctx, noopKwargs, intList(200), floatList(0.1), object.NewInteger(1)), "int8 range")
	assertError(t, fnDequantizeQ8(ctx, noopKwargs, intList(1, 2, 3, 4), floatList(0.1), object.NewInteger(2)), "scales")
	strList := &object.List{Elements: []object.Object{&object.String{Value: "x"}}}
	assertError(t, fnDequantizeQ8(ctx, noopKwargs, strList, floatList(0.1), object.NewInteger(1)), "INTEGER")
	result := fnDequantizeQ8(ctx, noopKwargs, intList(), floatList(0.1), object.NewInteger(2))
	assertEmptyListOrFloatArray(t, result)
}

func TestToFloatListErrors(t *testing.T) {
	_, errObj := toFloatList(&object.String{Value: "x"}, "test", "p")
	if errObj == nil {
		t.Error("expected error for non-list input")
	}
	strList := &object.List{Elements: []object.Object{&object.String{Value: "x"}}}
	_, errObj = toFloatList(strList, "test", "p")
	if errObj == nil {
		t.Error("expected error for non-number element")
	}
}

func TestToFloatMatrixErrors(t *testing.T) {
	_, errObj := toFloatMatrix(&object.String{Value: "x"}, "test", "p")
	if errObj == nil {
		t.Error("expected error for non-list input")
	}
	listWithStr := &object.List{Elements: []object.Object{&object.String{Value: "x"}}}
	_, errObj = toFloatMatrix(listWithStr, "test", "p")
	if errObj == nil {
		t.Error("expected error for non-list row")
	}
	nonRect := &object.List{Elements: []object.Object{
		floatList(1.0, 2.0),
		floatList(1.0),
	}}
	_, errObj = toFloatMatrix(nonRect, "test", "p")
	if errObj == nil {
		t.Error("expected error for non-rectangular matrix")
	}
	strInMatrix := &object.List{Elements: []object.Object{
		&object.List{Elements: []object.Object{&object.String{Value: "x"}}},
	}}
	_, errObj = toFloatMatrix(strInMatrix, "test", "p")
	if errObj == nil {
		t.Error("expected error for non-number element in matrix")
	}
}

func TestLinearRowQ8Errors(t *testing.T) {
	assertError(t, fnLinearRowQ8(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, &object.String{Value: "x"}, &object.String{Value: "x"}, object.NewInteger(1)), "LIST or FLOAT_ARRAY")
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), &object.String{Value: "x"}, &object.String{Value: "x"}), "INTEGER")
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), &object.String{Value: "x"}, object.NewInteger(0)), "positive")

	assertError(t, fnLinearRowQ8(ctx, noopKwargs, floatMatrix(), makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "empty")

	xNotMatrix := &object.List{Elements: []object.Object{&object.String{Value: "x"}}}
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, xNotMatrix, makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "list of lists")
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, floatMatrix([]float64{1.0}), makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "columns")

	elems := make([]object.Object, 32)
	for i := 0; i < 31; i++ {
		elems[i] = &object.Float{Value: 1.0}
	}
	elems[31] = &object.String{Value: "x"}
	xWithStr := &object.List{Elements: []object.Object{&object.List{Elements: elems}}}
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, xWithStr, makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "INTEGER or FLOAT")
}

func TestLibraryRegistration(t *testing.T) {
	if Library.Name() != "llm" {
		t.Errorf("Library.Name() = %s, want llm", Library.Name())
	}

	funcs := Library.Functions()
	funcCount := len(funcs)
	if funcCount != 39 {
		t.Errorf("Library has %d functions, want 39", funcCount)
	}

	required := []string{
		"argmax", "argmin", "topk", "clip",
		"sigmoid", "relu", "gelu", "silu",
		"vec_add", "vec_sub", "vec_mul", "vec_scale", "vec_apply",
		"rms_norm", "rope", "silu_gate", "attention", "linear", "linear_row",
		"linear_q8", "linear_row_q8", "linear_q4", "linear_row_q4",
		"top_k", "dequantize_q8", "dequantize_q8_0", "dequantize_q4_0",
		"sample", "split_heads", "merge_heads", "repeat_kv",
		"concat_rows", "slice_rows", "flatten", "reshape", "zeros", "arange",
		"quantize_q8", "quantize_q8_rows",
	}
	for _, name := range required {
		if _, ok := funcs[name]; !ok {
			t.Errorf("missing function: %s", name)
		}
	}

	consts := Library.Constants()
	if v, ok := consts["VERSION"]; !ok {
		t.Error("missing VERSION constant")
	} else {
		if v.(*object.String).Value != "1.1.0" {
			t.Errorf("VERSION = %s, want 1.1.0", v.(*object.String).Value)
		}
	}
}
