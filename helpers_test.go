package scriptlingllmlib

import (
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestToFloatListErrors(t *testing.T) {
	_, errObj := toFloatList(object.NewString("x"), "test", "p")
	if errObj == nil {
		t.Error("expected error for non-list input")
	}
	strList := &object.List{Elements: []object.Object{object.NewString("x")}}
	_, errObj = toFloatList(strList, "test", "p")
	if errObj == nil {
		t.Error("expected error for non-number element")
	}
}

func TestToFloatMatrixErrors(t *testing.T) {
	_, errObj := toFloatMatrix(object.NewString("x"), "test", "p")
	if errObj == nil {
		t.Error("expected error for non-list input")
	}
	listWithStr := &object.List{Elements: []object.Object{object.NewString("x")}}
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
		&object.List{Elements: []object.Object{object.NewString("x")}},
	}}
	_, errObj = toFloatMatrix(strInMatrix, "test", "p")
	if errObj == nil {
		t.Error("expected error for non-number element in matrix")
	}
}

func TestToFloatListValid(t *testing.T) {
	vals, errObj := toFloatList(floatList(1.0, 2.0, 3.0), "test", "p")
	if errObj != nil {
		t.Fatalf("unexpected error: %v", errObj)
	}
	if len(vals) != 3 || vals[0] != 1.0 || vals[2] != 3.0 {
		t.Errorf("toFloatList = %v", vals)
	}
}

func TestToFloatMatrixValid(t *testing.T) {
	mat, errObj := toFloatMatrix(floatMatrix([]float64{1.0, 2.0}, []float64{3.0, 4.0}), "test", "p")
	if errObj != nil {
		t.Fatalf("unexpected error: %v", errObj)
	}
	if len(mat) != 2 || mat[0][0] != 1.0 || mat[1][1] != 4.0 {
		t.Errorf("toFloatMatrix = %v", mat)
	}
}
