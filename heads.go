package scriptlingllmlib

import (
	"context"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func fnSplitHeads(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	xData, xRows, xCols, ok := object.GetFloatMatrix(args[0])
	if !ok {
		mat, errObj := toFloatMatrix(args[0], "split_heads", "x")
		if errObj != nil {
			return errObj
		}
		if len(mat) == 0 {
			return errors.NewError("split_heads: x cannot be empty")
		}
		xData = make([]float64, 0, len(mat)*len(mat[0]))
		for _, r := range mat {
			xData = append(xData, r...)
		}
		xRows = len(mat)
		xCols = len(mat[0])
	}
	nHeads64, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	nHeads := int(nHeads64)
	if nHeads <= 0 {
		return errors.NewError("split_heads: n_heads must be positive")
	}
	if xCols%nHeads != 0 {
		return errors.NewError("split_heads: columns (%d) must be divisible by n_heads (%d)", xCols, nHeads)
	}
	dK := xCols / nHeads

	heads := make([]object.Object, nHeads)
	for h := 0; h < nHeads; h++ {
		headData := make([]float64, xRows*dK)
		for r := 0; r < xRows; r++ {
			srcOff := r*xCols + h*dK
			dstOff := r * dK
			copy(headData[dstOff:dstOff+dK], xData[srcOff:srcOff+dK])
		}
		heads[h] = object.NewFloatArray2D(headData, xRows, dK)
	}
	return &object.List{Elements: heads}
}

func fnMergeHeads(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 1); err != nil {
		return err
	}
	list, ok := args[0].(*object.List)
	if !ok {
		return errors.NewTypeError("LIST", args[0].Type().String())
	}
	nHeads := len(list.Elements)
	if nHeads == 0 {
		return errors.NewError("merge_heads: heads list cannot be empty")
	}

	firstData, firstRows, firstCols, ok := object.GetFloatMatrix(list.Elements[0])
	if !ok {
		mat, errObj := toFloatMatrix(list.Elements[0], "merge_heads", "head_0")
		if errObj != nil {
			return errObj
		}
		firstRows = len(mat)
		if firstRows == 0 {
			return errors.NewError("merge_heads: head matrices cannot be empty")
		}
		firstCols = len(mat[0])
		firstData = make([]float64, 0, firstRows*firstCols)
		for _, r := range mat {
			firstData = append(firstData, r...)
		}
	}
	seqLen := firstRows
	dK := firstCols
	totalCols := nHeads * dK

	allData := make([][]float64, nHeads)
	allData[0] = firstData
	for h := 1; h < nHeads; h++ {
		hd, hr, hc, ok := object.GetFloatMatrix(list.Elements[h])
		if !ok {
			mat, errObj := toFloatMatrix(list.Elements[h], "merge_heads", "head")
			if errObj != nil {
				return errObj
			}
			if len(mat) != seqLen || len(mat[0]) != dK {
				return errors.NewError("merge_heads: all heads must have the same shape")
			}
			hd = make([]float64, 0, seqLen*dK)
			for _, r := range mat {
				hd = append(hd, r...)
			}
			hr = seqLen
			hc = dK
		}
		if hr != seqLen || hc != dK {
			return errors.NewError("merge_heads: all heads must have the same shape")
		}
		allData[h] = hd
	}

	result := make([]float64, seqLen*totalCols)
	for s := 0; s < seqLen; s++ {
		for h := 0; h < nHeads; h++ {
			srcOff := s * dK
			dstOff := s*totalCols + h*dK
			copy(result[dstOff:dstOff+dK], allData[h][srcOff:srcOff+dK])
		}
	}
	return object.NewFloatArray2D(result, seqLen, totalCols)
}

func fnRepeatKV(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}
	headList, ok := args[0].(*object.List)
	if !ok {
		return errors.NewTypeError("LIST", args[0].Type().String())
	}
	nRep64, err := args[1].AsInt()
	if err != nil {
		return errors.NewTypeError("INTEGER", args[1].Type().String())
	}
	nRep := int(nRep64)
	if nRep <= 0 {
		return errors.NewError("repeat_kv: n_rep must be positive")
	}

	result := make([]object.Object, 0, len(headList.Elements)*nRep)
	for _, h := range headList.Elements {
		for i := 0; i < nRep; i++ {
			result = append(result, h)
		}
	}
	return &object.List{Elements: result}
}
