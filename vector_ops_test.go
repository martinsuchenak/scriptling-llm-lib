package scriptlingllmlib

import (
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

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
	assertError(t, fnVecAdd(ctx, noopKwargs), "2 arguments")
	assertError(t, fnVecAdd(ctx, noopKwargs, &object.String{Value: "x"}, floatList(1.0)), "LIST")
	assertError(t, fnVecAdd(ctx, noopKwargs, floatList(1.0), &object.String{Value: "x"}), "LIST")
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
	assertError(t, fnVecSub(ctx, noopKwargs), "2 arguments")
	assertError(t, fnVecSub(ctx, noopKwargs, floatList(1.0), floatList(1.0, 2.0)), "same length")
	assertError(t, fnVecSub(ctx, noopKwargs, &object.String{Value: "x"}, floatList(1.0)), "LIST")
}

func TestVecMul(t *testing.T) {
	result := fnVecMul(ctx, noopKwargs, floatList(2.0, 3.0), floatList(4.0, 5.0))
	vals := evalFloatList(t, result)
	if vals[0] != 8.0 || vals[1] != 15.0 {
		t.Errorf("vec_mul = %v, want [8, 15]", vals)
	}
	assertError(t, fnVecMul(ctx, noopKwargs), "2 arguments")
	assertError(t, fnVecMul(ctx, noopKwargs, floatList(1.0), floatList(1.0, 2.0)), "same length")
	assertError(t, fnVecMul(ctx, noopKwargs, &object.String{Value: "x"}, floatList(1.0)), "LIST")
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
	assertError(t, fnVecScale(ctx, noopKwargs), "2 arguments")
	assertError(t, fnVecScale(ctx, noopKwargs, &object.String{Value: "x"}, &object.Float{Value: 1.0}), "LIST")
	assertError(t, fnVecScale(ctx, noopKwargs, floatList(1.0), &object.String{Value: "x"}), "INTEGER or FLOAT")
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
	assertError(t, fnVecApply(ctx, noopKwargs), "2 arguments")
	assertError(t, fnVecApply(ctx, noopKwargs, floatList(1.0), object.NewInteger(0)), "STRING")
	assertError(t, fnVecApply(ctx, noopKwargs, &object.String{Value: "x"}, &object.String{Value: "relu"}), "LIST")
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
