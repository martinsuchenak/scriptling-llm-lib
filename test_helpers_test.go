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
		elems[i] = object.NewFloat(v)
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
	return object.NewString(string(raw))
}

func approxEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-10
}

var ctx = context.Background()
var noopKwargs = object.Kwargs{}
