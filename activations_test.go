package scriptlingllmlib

import (
	"context"
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.5},
		{100.0, 1.0},
		{-100.0, 0.0},
	}
	for _, tt := range tests {
		result := fnSigmoid(ctx, noopKwargs, &object.Float{Value: tt.input})
		got := evalFloat(t, result)
		if math.Abs(got-tt.expected) > 1e-6 {
			t.Errorf("sigmoid(%f) = %f, want %f", tt.input, got, tt.expected)
		}
	}
}

func TestRelu(t *testing.T) {
	if evalFloat(t, fnRelu(ctx, noopKwargs, &object.Float{Value: -1.0})) != 0.0 {
		t.Error("relu(-1) != 0")
	}
	if evalFloat(t, fnRelu(ctx, noopKwargs, &object.Float{Value: 0.0})) != 0.0 {
		t.Error("relu(0) != 0")
	}
	if evalFloat(t, fnRelu(ctx, noopKwargs, &object.Float{Value: 5.0})) != 5.0 {
		t.Error("relu(5) != 5")
	}
}

func TestGelu(t *testing.T) {
	result := fnGelu(ctx, noopKwargs, &object.Float{Value: 0.0})
	if evalFloat(t, result) != 0.0 {
		t.Errorf("gelu(0) = %f, want 0", evalFloat(t, result))
	}
	result = fnGelu(ctx, noopKwargs, &object.Float{Value: 1.0})
	got := evalFloat(t, result)
	expected := 0.5 * 1.0 * (1.0 + math.Erf(1.0/math.Sqrt(2.0)))
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("gelu(1) = %f, want %f", got, expected)
	}
}

func TestSilu(t *testing.T) {
	result := fnSilu(ctx, noopKwargs, &object.Float{Value: 0.0})
	if evalFloat(t, result) != 0.0 {
		t.Errorf("silu(0) = %f, want 0", evalFloat(t, result))
	}
	x := 2.0
	result = fnSilu(ctx, noopKwargs, &object.Float{Value: x})
	expected := x * (1.0 / (1.0 + math.Exp(-x)))
	if math.Abs(evalFloat(t, result)-expected) > 1e-10 {
		t.Errorf("silu(2) = %f, want %f", evalFloat(t, result), expected)
	}
}

func TestActivationErrors(t *testing.T) {
	for name, fn := range map[string]func(context.Context, object.Kwargs, ...object.Object) object.Object{
		"sigmoid": fnSigmoid, "relu": fnRelu, "gelu": fnGelu, "silu": fnSilu,
	} {
		t.Run(name, func(t *testing.T) {
			assertError(t, fn(ctx, noopKwargs), "1 argument")
			assertError(t, fn(ctx, noopKwargs, &object.String{Value: "x"}), "INTEGER or FLOAT")
		})
	}
}

func TestActivationPureFunctions(t *testing.T) {
	if !approxEqual(sigmoid(0), 0.5) {
		t.Errorf("sigmoid(0) = %f", sigmoid(0))
	}
	if relu(-5) != 0 {
		t.Errorf("relu(-5) = %f", relu(-5))
	}
	if relu(3) != 3 {
		t.Errorf("relu(3) = %f", relu(3))
	}
	if gelu(0) != 0 {
		t.Errorf("gelu(0) = %f", gelu(0))
	}
	if silu(0) != 0 {
		t.Errorf("silu(0) = %f", silu(0))
	}
	sig := sigmoid(1)
	if sig <= 0 || sig >= 1 {
		t.Errorf("sigmoid(1) = %f, want (0,1)", sig)
	}
	s := silu(2)
	expected := 2.0 * sigmoid(2.0)
	if !approxEqual(s, expected) {
		t.Errorf("silu(2) = %f, want %f", s, expected)
	}
}
