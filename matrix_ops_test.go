package scriptlingllmlib

import (
	"testing"

	"github.com/paularlott/scriptling/object"
)

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
	assertError(t, fnConcatRows(ctx, noopKwargs), "2 arguments")
	assertError(t, fnConcatRows(ctx, noopKwargs, object.NewString("x"), floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnConcatRows(ctx, noopKwargs, floatMatrix([]float64{1.0}), object.NewString("x")), "LIST")
	result = fnConcatRows(ctx, noopKwargs, floatMatrix(), floatMatrix())
	assertEmptyListOrFloatArray(t, result)
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

	assertError(t, fnSliceRows(ctx, noopKwargs), "3 arguments")
	assertError(t, fnSliceRows(ctx, noopKwargs, object.NewString("x"), object.NewInteger(0), object.NewInteger(1)), "LIST")
	assertError(t, fnSliceRows(ctx, noopKwargs, floatMatrix([]float64{1.0}), object.NewString("x"), object.NewInteger(1)), "INTEGER")
	assertError(t, fnSliceRows(ctx, noopKwargs, floatMatrix([]float64{1.0}), object.NewInteger(0), object.NewString("x")), "INTEGER")

	m2 := floatMatrix([]float64{1.0}, []float64{2.0}, []float64{3.0})
	result = fnSliceRows(ctx, noopKwargs, m2, object.NewInteger(2), object.NewInteger(2))
	assertEmptyListOrFloatArray(t, result)
	result = fnSliceRows(ctx, noopKwargs, m2, object.NewInteger(-1), object.NewInteger(1))
	mat = evalFloatMatrix(t, result)
	if len(mat) != 1 || mat[0][0] != 1.0 {
		t.Errorf("slice_rows negative start = %v, want [[1.0]]", mat)
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
	assertError(t, fnFlatten(ctx, noopKwargs), "1 argument")
	assertError(t, fnFlatten(ctx, noopKwargs, object.NewString("x")), "LIST")
}

func TestReshape(t *testing.T) {
	data := floatList(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
	result := fnReshape(ctx, noopKwargs, data, object.NewInteger(2), object.NewInteger(3))
	mat := evalFloatMatrix(t, result)
	if len(mat) != 2 || len(mat[0]) != 3 {
		t.Fatalf("reshape = %dx%d, want 2x3", len(mat), len(mat[0]))
	}
	if mat[0][0] != 1.0 || mat[1][2] != 6.0 {
		t.Errorf("reshape values = %v", mat)
	}

	assertError(t, fnReshape(ctx, noopKwargs, data, object.NewInteger(2), object.NewInteger(2)), "must equal")
	assertError(t, fnReshape(ctx, noopKwargs), "3 arguments")
	assertError(t, fnReshape(ctx, noopKwargs, object.NewString("x"), object.NewInteger(1), object.NewInteger(1)), "LIST")
}

func TestZeros(t *testing.T) {
	result := fnZeros(ctx, noopKwargs, object.NewInteger(4))
	vals := evalFloatList(t, result)
	if len(vals) != 4 {
		t.Fatalf("zeros(4) len = %d, want 4", len(vals))
	}
	for _, v := range vals {
		if v != 0 {
			t.Errorf("zeros should be all zero, got %f", v)
		}
	}

	result2 := fnZeros(ctx, noopKwargs, object.NewInteger(2), object.NewInteger(3))
	mat := evalFloatMatrix(t, result2)
	if len(mat) != 2 || len(mat[0]) != 3 {
		t.Fatalf("zeros(2,3) = %dx%d, want 2x3", len(mat), len(mat[0]))
	}

	assertError(t, fnZeros(ctx, noopKwargs), "1 argument")
	assertError(t, fnZeros(ctx, noopKwargs, object.NewString("x")), "INTEGER")
}

func TestArange(t *testing.T) {
	result := fnArange(ctx, noopKwargs, object.NewInteger(5))
	vals := evalFloatList(t, result)
	expected := []float64{0, 1, 2, 3, 4}
	for i, v := range vals {
		if v != expected[i] {
			t.Errorf("arange(5)[%d] = %f, want %f", i, v, expected[i])
		}
	}

	result = fnArange(ctx, noopKwargs, object.NewFloat(1.0), object.NewFloat(4.0))
	vals = evalFloatList(t, result)
	if len(vals) != 3 || vals[0] != 1.0 || vals[2] != 3.0 {
		t.Errorf("arange(1,4) = %v", vals)
	}

	result = fnArange(ctx, noopKwargs, object.NewFloat(0.0), object.NewFloat(1.0), object.NewFloat(0.25))
	vals = evalFloatList(t, result)
	if len(vals) != 4 {
		t.Errorf("arange(0,1,0.25) len = %d, want 4", len(vals))
	}

	assertError(t, fnArange(ctx, noopKwargs, object.NewFloat(0.0), object.NewFloat(1.0), object.NewFloat(0.0)), "step must not be zero")
	assertError(t, fnArange(ctx, noopKwargs), "1 argument")
}
