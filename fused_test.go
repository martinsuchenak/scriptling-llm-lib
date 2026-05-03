package scriptlingllmlib

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func makeQ4KBlock(d, dmin float64, scales, qs []byte) []byte {
	block := make([]byte, 144)
	binary.LittleEndian.PutUint16(block[0:2], float64ToFloat16(d))
	binary.LittleEndian.PutUint16(block[2:4], float64ToFloat16(dmin))
	copy(block[4:16], scales)
	copy(block[16:144], qs)
	return block
}

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
	assertError(t, fnOutputLogits(ctx, noopKwargs, &object.String{Value: "x"}, normW, w), "FLOAT_MATRIX")
	assertError(t, fnOutputLogits(ctx, noopKwargs, x, &object.String{Value: "x"}, w), "LIST")

	emptyMat := object.NewFloatArray2D([]float64{}, 0, 0)
	assertError(t, fnOutputLogits(ctx, noopKwargs, emptyMat, normW, w), "empty")

	assertError(t, fnOutputLogits(ctx, noopKwargs, x, normW, &object.Integer{Value: 42}), "unsupported weight type")
}

func TestQ4KDotBlockEquivalence(t *testing.T) {
	scales := make([]byte, 12)
	for i := range scales {
		scales[i] = 1
	}
	qs := make([]byte, 128)
	for i := range qs {
		qs[i] = byte(i % 256)
	}

	block := makeQ4KBlock(2.0, 0.5, scales, qs)

	x := make([]float64, 256)
	for i := range x {
		x[i] = float64(i%10) * 0.1
	}

	slow := q4kDotBlock(block, 0, x, 0)
	fast := q4kDotBlockFast(block, 0, x, 0)

	if math.IsNaN(slow) || math.IsInf(slow, 0) {
		t.Fatalf("slow result = %f", slow)
	}
	if math.Abs(slow-fast) > 1e-6 {
		t.Errorf("q4kDotBlockFast = %f, q4kDotBlock = %f, diff = %f", fast, slow, math.Abs(fast-slow))
	}
}

func TestMathSqrt(t *testing.T) {
	if math_sqrt(4.0) != 2.0 {
		t.Errorf("math_sqrt(4) = %f, want 2", math_sqrt(4.0))
	}
	if math.Abs(math_sqrt(2.0)-1.4142135623730951) > 1e-10 {
		t.Errorf("math_sqrt(2) = %f", math_sqrt(2.0))
	}
}
