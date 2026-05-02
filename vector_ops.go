package scriptlingllmlib

import (
	"context"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func fnVecAdd(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	a, errObj := toFloatList(args[0], "vec_add", "a")
	if errObj != nil {
		return errObj
	}
	b, errObj := toFloatList(args[1], "vec_add", "b")
	if errObj != nil {
		return errObj
	}
	if len(a) != len(b) {
		return errors.NewError("vec_add: vectors must have the same length")
	}
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return object.NewFloatArray1D(result)
}

func fnVecSub(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	a, errObj := toFloatList(args[0], "vec_sub", "a")
	if errObj != nil {
		return errObj
	}
	b, errObj := toFloatList(args[1], "vec_sub", "b")
	if errObj != nil {
		return errObj
	}
	if len(a) != len(b) {
		return errors.NewError("vec_sub: vectors must have the same length")
	}
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return object.NewFloatArray1D(result)
}

func fnVecMul(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	a, errObj := toFloatList(args[0], "vec_mul", "a")
	if errObj != nil {
		return errObj
	}
	b, errObj := toFloatList(args[1], "vec_mul", "b")
	if errObj != nil {
		return errObj
	}
	if len(a) != len(b) {
		return errors.NewError("vec_mul: vectors must have the same length")
	}
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return object.NewFloatArray1D(result)
}

func fnVecScale(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	a, errObj := toFloatList(args[0], "vec_scale", "a")
	if errObj != nil {
		return errObj
	}
	s, err := args[1].AsFloat()
	if err != nil {
		return errors.NewTypeError("INTEGER or FLOAT", args[1].Type().String())
	}
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] * s
	}
	return object.NewFloatArray1D(result)
}

func fnVecApply(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	vals, errObj := toFloatList(args[0], "vec_apply", "x")
	if errObj != nil {
		return errObj
	}
	fnName, err := args[1].AsString()
	if err != nil {
		return errors.NewTypeError("STRING", args[1].Type().String())
	}

	var fn func(float64) float64
	switch fnName {
	case "sigmoid":
		fn = sigmoid
	case "relu":
		fn = relu
	case "gelu":
		fn = gelu
	case "silu":
		fn = silu
	default:
		return errors.NewError("vec_apply: unknown function '%s', must be one of: sigmoid, relu, gelu, silu", fnName)
	}

	result := make([]float64, len(vals))
	for i, v := range vals {
		result[i] = fn(v)
	}
	return object.NewFloatArray1D(result)
}
