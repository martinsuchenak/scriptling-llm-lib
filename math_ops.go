package scriptlingllmlib

import (
	"context"
	"math"
	"sort"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func fnArgmax(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 1); err != nil {
		return err
	}
	vals, errObj := toFloatList(args[0], "argmax", "x")
	if errObj != nil {
		return errObj
	}
	if len(vals) == 0 {
		return errors.NewError("argmax: input list cannot be empty")
	}
	bestIdx := 0
	bestVal := vals[0]
	for i := 1; i < len(vals); i++ {
		if vals[i] > bestVal {
			bestVal = vals[i]
			bestIdx = i
		}
	}
	return object.NewInteger(int64(bestIdx))
}

func fnArgmin(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 1); err != nil {
		return err
	}
	vals, errObj := toFloatList(args[0], "argmin", "x")
	if errObj != nil {
		return errObj
	}
	if len(vals) == 0 {
		return errors.NewError("argmin: input list cannot be empty")
	}
	bestIdx := 0
	bestVal := vals[0]
	for i := 1; i < len(vals); i++ {
		if vals[i] < bestVal {
			bestVal = vals[i]
			bestIdx = i
		}
	}
	return object.NewInteger(int64(bestIdx))
}

type indexedVal struct {
	index int
	value float64
}

func fnTopk(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	vals, errObj := toFloatList(args[0], "topk", "x")
	if errObj != nil {
		return errObj
	}
	k, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	n := len(vals)
	if k <= 0 {
		return errors.NewError("topk: k must be positive")
	}
	if int(k) > n {
		k = int64(n)
	}

	indexed := make([]indexedVal, n)
	for i, v := range vals {
		indexed[i] = indexedVal{index: i, value: v}
	}
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].value > indexed[j].value
	})

	result := make([]object.Object, k)
	for i := int64(0); i < k; i++ {
		result[i] = &object.List{Elements: []object.Object{
			object.NewInteger(int64(indexed[i].index)),
			&object.Float{Value: indexed[i].value},
		}}
	}
	return &object.List{Elements: result}
}

func fnClip(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 3); err != nil {
		return err
	}

	if list, ok := args[0].(*object.List); ok {
		lo, err := args[1].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[1].Type().String())
		}
		hi, err := args[2].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[2].Type().String())
		}
		if lo > hi {
			return errors.NewError("clip: lo must be <= hi")
		}
		result := make([]float64, len(list.Elements))
		for i, el := range list.Elements {
			v, err := el.AsFloat()
			if err != nil {
				return errors.NewTypeError("INTEGER or FLOAT", el.Type().String())
			}
			result[i] = math.Max(lo, math.Min(hi, v))
		}
		return object.NewFloatArray1D(result)
	}

	if fa, ok := args[0].(*object.FloatArray); ok && !fa.Is2D() {
		lo, err := args[1].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[1].Type().String())
		}
		hi, err := args[2].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[2].Type().String())
		}
		if lo > hi {
			return errors.NewError("clip: lo must be <= hi")
		}
		result := make([]float64, len(fa.Data))
		for i, v := range fa.Data {
			result[i] = math.Max(lo, math.Min(hi, v))
		}
		return object.NewFloatArray1D(result)
	}

	x, err := args[0].AsFloat()
	if err != nil {
		return errors.NewTypeError("INTEGER, FLOAT, or LIST", args[0].Type().String())
	}
	lo, err := args[1].AsFloat()
	if err != nil {
		return errors.NewTypeError("INTEGER or FLOAT", args[1].Type().String())
	}
	hi, err := args[2].AsFloat()
	if err != nil {
		return errors.NewTypeError("INTEGER or FLOAT", args[2].Type().String())
	}
	if lo > hi {
		return errors.NewError("clip: lo must be <= hi")
	}
	return &object.Float{Value: math.Max(lo, math.Min(hi, x))}
}
