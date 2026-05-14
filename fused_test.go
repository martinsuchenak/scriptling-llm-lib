package scriptlingllmlib

import (
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestOutputLogitsFloat(t *testing.T) {
	x := object.NewFloatArray2D([]float64{0.5, -0.3, 0.8, 1.0, 0.0, 1.0}, 2, 3)
	normW := floatList(1.0, 1.0, 1.0)
	w := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, 2, 3)

	result := fnOutputLogits(ctx, noopKwargs, x, normW, w)
	logits := evalFloatList(t, result)
	if len(logits) != 2 {
		t.Fatalf("output_logits len = %d, want 2", len(logits))
	}

	lastRow := []float64{1.0, 0.0, 1.0}
	ss := (1.0 + 0.0 + 1.0) / 3.0
	inv := 1.0 / math.Sqrt(ss+1e-5)
	normed0 := lastRow[0] * inv * 1.0

	if math.Abs(logits[0]-normed0) > 1e-10 {
		t.Errorf("output_logits[0] = %f, want %f", logits[0], normed0)
	}
	normed1 := lastRow[1] * inv * 1.0
	if math.Abs(logits[1]-normed1) > 1e-10 {
		t.Errorf("output_logits[1] = %f, want %f", logits[1], normed1)
	}
}

func TestOutputLogitsQ8(t *testing.T) {
	ones := make([]float64, 32)
	for i := range ones {
		ones[i] = 1.0
	}
	x := object.NewFloatArray2D(ones, 1, 32)
	normW := floatList(ones...)

	qValues := make([]int8, 32)
	for i := range qValues {
		qValues[i] = 1
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))

	result := fnOutputLogits(ctx, noopKwargs, x, normW, raw)
	logits := evalFloatList(t, result)
	if len(logits) != 1 {
		t.Fatalf("output_logits Q8 len = %d, want 1", len(logits))
	}
	if math.IsNaN(logits[0]) || math.IsInf(logits[0], 0) {
		t.Errorf("output_logits Q8 = %f, expected finite", logits[0])
	}
}

func TestOutputLogitsErrors(t *testing.T) {
	x := object.NewFloatArray2D([]float64{1.0, 1.0}, 1, 2)
	normW := floatList(1.0, 1.0)
	w := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})

	assertError(t, fnOutputLogits(ctx, noopKwargs), "3 arguments")
	assertError(t, fnOutputLogits(ctx, noopKwargs, object.NewString("x"), normW, w), "FLOAT_MATRIX")
	assertError(t, fnOutputLogits(ctx, noopKwargs, x, object.NewString("x"), w), "LIST")

	emptyMat := object.NewFloatArray2D([]float64{}, 0, 0)
	assertError(t, fnOutputLogits(ctx, noopKwargs, emptyMat, normW, w), "empty")

	assertError(t, fnOutputLogits(ctx, noopKwargs, x, normW, object.NewInteger(42)), "unsupported weight type")
}

func TestMathSqrt(t *testing.T) {
	if math_sqrt(4.0) != 2.0 {
		t.Errorf("math_sqrt(4) = %f, want 2", math_sqrt(4.0))
	}
	if math.Abs(math_sqrt(2.0)-1.4142135623730951) > 1e-10 {
		t.Errorf("math_sqrt(2) = %f", math_sqrt(2.0))
	}
}
