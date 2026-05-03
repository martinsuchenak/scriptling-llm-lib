package scriptlingllmlib

import (
	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func toFloatList(obj object.Object, fnName, paramName string) ([]float64, object.Object) {
	if fa, ok := obj.(*object.FloatArray); ok && !fa.Is2D() {
		result := make([]float64, len(fa.Data))
		copy(result, fa.Data)
		return result, nil
	}
	list, ok := obj.(*object.List)
	if !ok {
		return nil, errors.NewTypeError("LIST or FLOAT_ARRAY", obj.Type().String())
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

func toFloatMatrix(obj object.Object, fnName, paramName string) ([][]float64, object.Object) {
	if fa, ok := obj.(*object.FloatArray); ok && fa.Is2D() {
		rows := fa.Rows()
		result := make([][]float64, rows)
		for i := 0; i < rows; i++ {
			row := fa.Row(i)
			rowCopy := make([]float64, len(row))
			copy(rowCopy, row)
			result[i] = rowCopy
		}
		return result, nil
	}
	if fa, ok := obj.(*object.FloatArray); ok && !fa.Is2D() {
		return nil, errors.NewError("%s: %s must be a 2D matrix, got 1D FloatArray", fnName, paramName)
	}
	list, ok := obj.(*object.List)
	if !ok {
		return nil, errors.NewTypeError("LIST or FLOAT_ARRAY", obj.Type().String())
	}
	rows := make([][]float64, len(list.Elements))
	width := -1
	for i, rowObj := range list.Elements {
		if innerFA, ok := rowObj.(*object.FloatArray); ok && !innerFA.Is2D() {
			if width == -1 {
				width = len(innerFA.Data)
			} else if len(innerFA.Data) != width {
				return nil, errors.NewError("%s: %s must be a rectangular matrix", fnName, paramName)
			}
			rowCopy := make([]float64, len(innerFA.Data))
			copy(rowCopy, innerFA.Data)
			rows[i] = rowCopy
			continue
		}
		row, ok := rowObj.(*object.List)
		if !ok {
			return nil, errors.NewError("%s: %s must be a list of lists or FloatArray", fnName, paramName)
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
