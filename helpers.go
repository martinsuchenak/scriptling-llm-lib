package scriptlingllmlib

import (
	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

// toFloatList converts a Scriptling LIST of numbers into a Go []float64.
// Returns an error object if conversion fails.
func toFloatList(obj object.Object, fnName, paramName string) ([]float64, object.Object) {
	list, ok := obj.(*object.List)
	if !ok {
		return nil, errors.NewTypeError("LIST", obj.Type().String())
	}
	vals := make([]float64, len(list.Elements))
	for i, el := range list.Elements {
		f, err := el.AsFloat()
		if err != nil {
			return nil, errors.NewTypeError("INTEGER or FLOAT", el.Type().String())
		}
		vals[i] = f
	}
	return vals, nil
}

// toFloatMatrix converts a Scriptling LIST of LIST of numbers into a Go [][]float64.
// Validates that the matrix is rectangular. Returns an error object if
// conversion fails or rows have inconsistent lengths.
func toFloatMatrix(obj object.Object, fnName, paramName string) ([][]float64, object.Object) {
	list, ok := obj.(*object.List)
	if !ok {
		return nil, errors.NewTypeError("LIST", obj.Type().String())
	}
	rows := make([][]float64, len(list.Elements))
	width := -1
	for i, rowObj := range list.Elements {
		row, ok := rowObj.(*object.List)
		if !ok {
			return nil, errors.NewError("%s: %s must be a list of lists", fnName, paramName)
		}
		if width == -1 {
			width = len(row.Elements)
		} else if len(row.Elements) != width {
			return nil, errors.NewError("%s: %s must be a rectangular matrix", fnName, paramName)
		}
		rows[i] = make([]float64, len(row.Elements))
		for j, el := range row.Elements {
			f, err := el.AsFloat()
			if err != nil {
				return nil, errors.NewTypeError("INTEGER or FLOAT", el.Type().String())
			}
			rows[i][j] = f
		}
	}
	return rows, nil
}

// floatListToObject converts a Go []float64 into a Scriptling LIST of FLOAT objects.
func floatListToObject(vals []float64) object.Object {
	elems := make([]object.Object, len(vals))
	for i, v := range vals {
		elems[i] = &object.Float{Value: v}
	}
	return &object.List{Elements: elems}
}

// floatMatrixToObject converts a Go [][]float64 into a Scriptling LIST of LIST of FLOAT objects.
func floatMatrixToObject(m [][]float64) object.Object {
	rows := make([]object.Object, len(m))
	for i, r := range m {
		rows[i] = floatListToObject(r)
	}
	return &object.List{Elements: rows}
}
