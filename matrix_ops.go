package scriptlingllmlib

import (
	"context"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

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
		return object.NewFloatArray2D(nil, 0, 0)
	}
	if len(a) > 0 && len(b) > 0 && len(a[0]) != len(b[0]) {
		return errors.NewError("concat_rows: matrices must have the same number of columns")
	}

	cols := 0
	if len(a) > 0 {
		cols = len(a[0])
	} else if len(b) > 0 {
		cols = len(b[0])
	}
	rows := len(a) + len(b)
	data := make([]float64, 0, rows*cols)
	for _, r := range a {
		data = append(data, r...)
	}
	for _, r := range b {
		data = append(data, r...)
	}
	return object.NewFloatArray2D(data, rows, cols)
}

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
	if start >= end || len(rows) == 0 {
		return object.NewFloatArray2D(nil, 0, 0)
	}
	cols := len(rows[0])
	resultRows := int(end - start)
	data := make([]float64, 0, resultRows*cols)
	for i := int(start); i < int(end); i++ {
		data = append(data, rows[i]...)
	}
	return object.NewFloatArray2D(data, resultRows, cols)
}

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
	return object.NewFloatArray1D(result)
}

func fnReshape(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 3); err != nil {
		return err
	}
	data, errObj := toFloatList(args[0], "reshape", "data")
	if errObj != nil {
		return errObj
	}
	rows, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	cols, err := args[2].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[2].Type().String())
	}
	if int(rows)*int(cols) != len(data) {
		return errors.NewError("reshape: data length (%d) must equal rows*cols (%d*%d=%d)", len(data), rows, cols, int(rows)*int(cols))
	}
	return object.NewFloatArray2D(data, int(rows), int(cols))
}

func fnZeros(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 1, 2); err != nil {
		return err
	}
	if len(args) == 1 {
		n, err := args[0].AsInt()
		if err != nil {
			return errors.NewTypeError("INTEGER", args[0].Type().String())
		}
		return object.NewFloatArray1D(make([]float64, n))
	}
	rows, err := args[0].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[0].Type().String())
	}
	cols, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	return object.NewFloatArray2D(make([]float64, int(rows)*int(cols)), int(rows), int(cols))
}

func fnArange(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 1, 3); err != nil {
		return err
	}
	var start, stop, step float64
	switch len(args) {
	case 1:
		start = 0
		s, err := args[0].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[0].Type().String())
		}
		stop = s
		step = 1
	case 2:
		s, err := args[0].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[0].Type().String())
		}
		start = s
		s2, err := args[1].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[1].Type().String())
		}
		stop = s2
		step = 1
	case 3:
		s, err := args[0].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[0].Type().String())
		}
		start = s
		s2, err := args[1].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[1].Type().String())
		}
		stop = s2
		s3, err := args[2].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[2].Type().String())
		}
		step = s3
	}
	if step == 0 {
		return errors.NewError("arange: step must not be zero")
	}
	n := int((stop - start) / step)
	if n < 0 {
		n = 0
	}
	data := make([]float64, n)
	for i := 0; i < n; i++ {
		data[i] = start + float64(i)*step
	}
	return object.NewFloatArray1D(data)
}
