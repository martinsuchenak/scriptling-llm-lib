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
	elems := evalList(t, obj)
	vals := make([]float64, len(elems))
	for i, e := range elems {
		vals[i] = evalFloat(t, e)
	}
	return vals
}

func evalFloatMatrix(t *testing.T, obj object.Object) [][]float64 {
	t.Helper()
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
		expected := row[j] * inv * w.(*object.List).Elements[j].(*object.Float).Value
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

func TestLibraryRegistration(t *testing.T) {
	if Library.Name() != "llm" {
		t.Errorf("Library.Name() = %s, want llm", Library.Name())
	}

	funcs := Library.Functions()
	funcCount := len(funcs)
	if funcCount != 24 {
		t.Errorf("Library has %d functions, want 24", funcCount)
	}

	required := []string{
		"argmax", "argmin", "topk", "clip",
		"sigmoid", "relu", "gelu", "silu",
		"vec_add", "vec_sub", "vec_mul", "vec_scale", "vec_apply",
		"rms_norm", "rope", "silu_gate", "attention", "linear", "linear_row",
		"top_k", "dequantize_q8",
		"concat_rows", "slice_rows", "flatten",
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
		if v.(*object.String).Value != "1.0.0" {
			t.Errorf("VERSION = %s, want 1.0.0", v.(*object.String).Value)
		}
	}
}
