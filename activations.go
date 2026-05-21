package scriptlingllmlib

import (
	"context"
	"math"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

// sigmoid computes the logistic sigmoid: 1 / (1 + exp(-x)).
// Returns values in the range (0, 1).
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// relu computes the Rectified Linear Unit: max(0, x).
func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// gelu computes the Gaussian Error Linear Unit: 0.5 * x * (1 + erf(x / sqrt(2))).
// Used in BERT, GPT-2, T5.
func gelu(x float64) float64 {
	return 0.5 * x * (1.0 + math.Erf(x/math.Sqrt(2.0)))
}

// silu computes the Sigmoid Linear Unit (Swish): x * sigmoid(x).
// Used in LLaMA, Gemma, Mistral.
func silu(x float64) float64 {
	return x * sigmoid(x)
}

// fnSigmoid implements llm.sigmoid.
func fnSigmoid(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 1); err != nil {
		return err
	}
	x, err := args[0].AsFloat()
	if err != nil {
		return errors.NewTypeError("INTEGER or FLOAT", args[0].Type().String())
	}
	return object.NewFloat(sigmoid(x))
}

// fnRelu implements llm.relu.
func fnRelu(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 1); err != nil {
		return err
	}
	x, err := args[0].AsFloat()
	if err != nil {
		return errors.NewTypeError("INTEGER or FLOAT", args[0].Type().String())
	}
	return object.NewFloat(relu(x))
}

// fnGelu implements llm.gelu.
func fnGelu(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 1); err != nil {
		return err
	}
	x, err := args[0].AsFloat()
	if err != nil {
		return errors.NewTypeError("INTEGER or FLOAT", args[0].Type().String())
	}
	return object.NewFloat(gelu(x))
}

// fnSilu implements llm.silu.
func fnSilu(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 1); err != nil {
		return err
	}
	x, err := args[0].AsFloat()
	if err != nil {
		return errors.NewTypeError("INTEGER or FLOAT", args[0].Type().String())
	}
	return object.NewFloat(silu(x))
}
