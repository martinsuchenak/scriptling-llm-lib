package scriptlingllmlib

import (
	"context"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

// fnConcatRows implements llm.concat_rows: concatenate two matrices along the row axis.
// Both matrices must have the same number of columns.
func fnConcatRows(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	a, errObj := toFloatMatrix(args[0], "concat_rows", "a")
	if errObj != nil {
		return errObj
	}
	b, errObj := toFloatMatrix(args[1], "concat_rows", "b")
	if errObj != nil {
		return errObj
	}
	if len(a) == 0 && len(b) == 0 {
		return &object.List{Elements: []object.Object{}}
	}
	if len(a) > 0 && len(b) > 0 && len(a[0]) != len(b[0]) {
		return errors.NewError("concat_rows: matrices must have the same number of columns")
	}
	result := make([][]float64, 0, len(a)+len(b))
	result = append(result, a...)
	result = append(result, b...)
	return floatMatrixToObject(result)
}

// fnSliceRows implements llm.slice_rows: extract rows [start, end) from a matrix.
// Start and end are clamped to valid bounds. Returns empty list if start >= end.
func fnSliceRows(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 3); err != nil {
		return err
	}
	rows, errObj := toFloatMatrix(args[0], "slice_rows", "x")
	if errObj != nil {
		return errObj
	}
	start, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	end, err := args[2].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[2].Type().String())
	}
	n := int64(len(rows))
	if start < 0 {
		start = 0
	}
	if end > n {
		end = n
	}
	if start >= end {
		return &object.List{Elements: []object.Object{}}
	}
	return floatMatrixToObject(rows[start:end])
}

// fnFlatten implements llm.flatten: flatten a 2D matrix into a 1D list in row-major order.
func fnFlatten(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 1); err != nil {
		return err
	}
	rows, errObj := toFloatMatrix(args[0], "flatten", "x")
	if errObj != nil {
		return errObj
	}
	total := 0
	for _, r := range rows {
		total += len(r)
	}
	result := make([]float64, 0, total)
	for _, r := range rows {
		result = append(result, r...)
	}
	return floatListToObject(result)
}
