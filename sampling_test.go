package scriptlingllmlib

import (
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestSampleGreedy(t *testing.T) {
	result := fnSample(ctx, noopKwargs, floatList(0.1, 0.9, 0.3), object.NewString("greedy"))
	idx := evalInt(t, result)
	if idx != 1 {
		t.Errorf("greedy sample = %d, want 1", idx)
	}
}

func TestSampleGreedyDefault(t *testing.T) {
	result := fnSample(ctx, noopKwargs, floatList(0.1, 0.9, 0.3))
	idx := evalInt(t, result)
	if idx != 1 {
		t.Errorf("default greedy = %d, want 1", idx)
	}
}

func TestSampleErrors(t *testing.T) {
	assertError(t, fnSample(ctx, noopKwargs, floatList()), "empty")
	assertError(t, fnSample(ctx, noopKwargs), "1 argument")
	assertError(t, fnSample(ctx, noopKwargs, floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnSample(ctx, noopKwargs, floatList(1.0), object.NewString("unknown_strategy")), "unknown strategy")
}

func TestSampleTemperatureReturnsValidIndex(t *testing.T) {
	result := fnSample(ctx, noopKwargs, floatList(1.0, 2.0, 3.0), object.NewString("temperature"), object.NewFloat(1.0))
	idx := evalInt(t, result)
	if idx < 0 || idx > 2 {
		t.Errorf("temperature sample index = %d, out of range [0,2]", idx)
	}
}

func TestSampleTemperatureArgError(t *testing.T) {
	assertError(t, fnSample(ctx, noopKwargs, floatList(1.0, 2.0), object.NewString("temperature"), object.NewFloat(0.0)), "positive")
}

func TestApplyRepeatPenalty(t *testing.T) {
	logits := []float64{10.0, -5.0, 3.0}
	recent := []int64{0, 2}
	applyRepeatPenalty(logits, recent, 2.0, 64)

	if logits[0] != 5.0 {
		t.Errorf("positive logit should be divided: got %f, want 5.0", logits[0])
	}
	if logits[1] != -5.0 {
		t.Errorf("untouched logit: got %f, want -5.0", logits[1])
	}
	if logits[2] != 1.5 {
		t.Errorf("positive logit should be divided: got %f, want 1.5", logits[2])
	}
}

func TestApplyRepeatPenaltyNegative(t *testing.T) {
	logits := []float64{-10.0, 5.0}
	recent := []int64{0}
	applyRepeatPenalty(logits, recent, 2.0, 64)

	if logits[0] != -20.0 {
		t.Errorf("negative logit should be multiplied: got %f, want -20.0", logits[0])
	}
	if logits[1] != 5.0 {
		t.Errorf("untouched: got %f, want 5.0", logits[1])
	}
}

func TestApplyRepeatPenaltyOutOfBounds(t *testing.T) {
	logits := []float64{1.0, 2.0}
	recent := []int64{5, -1}
	applyRepeatPenalty(logits, recent, 2.0, 64)

	if logits[0] != 1.0 || logits[1] != 2.0 {
		t.Errorf("out-of-bounds tokens should be skipped: %v", logits)
	}
}

func TestApplyRepeatPenaltyDedup(t *testing.T) {
	logits := []float64{10.0, 5.0}
	recent := []int64{0, 0, 0}
	applyRepeatPenalty(logits, recent, 2.0, 64)

	if logits[0] != 5.0 {
		t.Errorf("dedup: should only penalize once, got %f", logits[0])
	}
}

func TestSoftmax(t *testing.T) {
	probs := softmax([]float64{1.0, 2.0, 3.0}, 1.0)

	var sum float64
	for _, p := range probs {
		sum += p
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("softmax sum = %f, want ~1.0", sum)
	}
	if probs[2] <= probs[1] || probs[1] <= probs[0] {
		t.Errorf("softmax should be monotonically increasing for increasing input: %v", probs)
	}
}

func TestSoftmaxInPlace(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0}
	result := softmaxInPlace(logits)

	var sum float64
	for _, p := range result {
		sum += p
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("softmaxInPlace sum = %f, want ~1.0", sum)
	}
}

func TestPartialNthElement(t *testing.T) {
	data := make([]indexedFloat, 5)
	vals := []float64{3.0, 1.0, 5.0, 2.0, 4.0}
	for i, v := range vals {
		data[i] = indexedFloat{index: i, value: v}
	}
	partialNthElement(data, 3)

	for i := 0; i < 3; i++ {
		for j := 3; j < 5; j++ {
			if data[i].value < data[j].value {
				t.Errorf("partialNthElement failed: top[%d]=%f < rest[%d]=%f", i, data[i].value, j, data[j].value)
			}
		}
	}
}

func TestSampleTopKStrategy(t *testing.T) {
	result := fnSample(ctx, noopKwargs, floatList(0.1, 0.9, 0.3), object.NewString("top_k"), object.NewFloat(1.0), object.NewInteger(2))
	idx := evalInt(t, result)
	if idx < 0 || idx > 2 {
		t.Errorf("top_k idx = %d, out of range [0,2]", idx)
	}
}

func TestSampleTopKStrategyDefaultK(t *testing.T) {
	result := fnSample(ctx, noopKwargs, floatList(0.1, 0.9, 0.3), object.NewString("top_k"), object.NewFloat(1.0))
	idx := evalInt(t, result)
	if idx < 0 || idx > 2 {
		t.Errorf("top_k default idx = %d, out of range [0,2]", idx)
	}
}

func TestSampleTopPStrategy(t *testing.T) {
	result := fnSample(ctx, noopKwargs, floatList(0.1, 0.9, 0.3), object.NewString("top_p"), object.NewFloat(1.0))
	idx := evalInt(t, result)
	if idx < 0 || idx > 2 {
		t.Errorf("top_p idx = %d, out of range [0,2]", idx)
	}
}

func TestSampleTemperatureKwarg(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"temperature": object.NewFloat(0.5),
	})
	result := fnSample(ctx, kwargs, floatList(0.1, 0.9, 0.3), object.NewString("temperature"))
	idx := evalInt(t, result)
	if idx < 0 || idx > 2 {
		t.Errorf("temperature kwarg idx = %d, out of range [0,2]", idx)
	}
}

func TestSampleTemperatureKwargTypeError(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"temperature": object.NewString("bad"),
	})
	assertError(t, fnSample(ctx, kwargs, floatList(1.0), object.NewString("temperature")), "INTEGER or FLOAT")
}

func TestSampleTemperatureKwargZero(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"temperature": object.NewFloat(0.0),
	})
	assertError(t, fnSample(ctx, kwargs, floatList(1.0), object.NewString("temperature")), "positive")
}

func TestSampleTopKKwarg(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"top_k": object.NewInteger(2),
	})
	result := fnSample(ctx, kwargs, floatList(0.1, 0.9, 0.3), object.NewString("top_k"), object.NewFloat(1.0))
	idx := evalInt(t, result)
	if idx < 0 || idx > 2 {
		t.Errorf("top_k kwarg idx = %d, out of range [0,2]", idx)
	}
}

func TestSampleTopKKwargTypeError(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"top_k": object.NewString("bad"),
	})
	assertError(t, fnSample(ctx, kwargs, floatList(1.0), object.NewString("top_k"), object.NewFloat(1.0)), "INTEGER")
}

func TestSampleTopPKwarg(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"top_p": object.NewFloat(0.9),
	})
	result := fnSample(ctx, kwargs, floatList(0.1, 0.9, 0.3), object.NewString("top_p"), object.NewFloat(1.0))
	idx := evalInt(t, result)
	if idx < 0 || idx > 2 {
		t.Errorf("top_p kwarg idx = %d, out of range [0,2]", idx)
	}
}

func TestSampleTopPInvalidRangeZero(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"top_p": object.NewFloat(0.0),
	})
	assertError(t, fnSample(ctx, kwargs, floatList(1.0), object.NewString("top_p")), "top_p must be in")
}

func TestSampleTopPInvalidRangeOverOne(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"top_p": object.NewFloat(1.5),
	})
	assertError(t, fnSample(ctx, kwargs, floatList(1.0), object.NewString("top_p")), "top_p must be in")
}

func TestSampleTopPTypeErrorKwarg(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"top_p": object.NewString("bad"),
	})
	assertError(t, fnSample(ctx, kwargs, floatList(1.0), object.NewString("top_p")), "INTEGER or FLOAT")
}

func TestSampleRepeatPenaltyKwarg(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"repeat_penalty": object.NewFloat(1.5),
		"recent_tokens":  intList(0, 1),
	})
	result := fnSample(ctx, kwargs, floatList(10.0, 5.0, 3.0), object.NewString("greedy"))
	idx := evalInt(t, result)
	if idx < 0 || idx > 2 {
		t.Errorf("repeat penalty idx = %d, out of range [0,2]", idx)
	}
}

func TestSampleRepeatPenaltyBelowOne(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"repeat_penalty": object.NewFloat(0.5),
	})
	assertError(t, fnSample(ctx, kwargs, floatList(1.0), object.NewString("greedy")), "repeat_penalty must be >= 1.0")
}

func TestSampleRepeatPenaltyKwargTypeError(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"repeat_penalty": object.NewString("bad"),
	})
	assertError(t, fnSample(ctx, kwargs, floatList(1.0), object.NewString("greedy")), "INTEGER or FLOAT")
}

func TestSampleRecentTokensNonInteger(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"recent_tokens":  &object.List{Elements: []object.Object{object.NewString("x")}},
		"repeat_penalty": object.NewFloat(1.5),
	})
	assertError(t, fnSample(ctx, kwargs, floatList(1.0), object.NewString("greedy")), "INTEGER")
}

func TestSampleRepeatLastNKwarg(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"repeat_last_n": object.NewInteger(32),
	})
	result := fnSample(ctx, kwargs, floatList(0.1, 0.9, 0.3), object.NewString("greedy"))
	idx := evalInt(t, result)
	if idx != 1 {
		t.Errorf("greedy with repeat_last_n = %d, want 1", idx)
	}
}

func TestSampleRepeatLastNKwargTypeError(t *testing.T) {
	kwargs := object.NewKwargs(map[string]object.Object{
		"repeat_last_n": object.NewString("bad"),
	})
	assertError(t, fnSample(ctx, kwargs, floatList(1.0), object.NewString("greedy")), "INTEGER")
}

func TestSampleTopKPositionalArgTypeError(t *testing.T) {
	assertError(t, fnSample(ctx, noopKwargs, floatList(0.1, 0.9, 0.3), object.NewString("top_k"), object.NewFloat(1.0), object.NewString("bad")), "INTEGER")
}

func TestPartialNthElementSingle(t *testing.T) {
	data := []indexedFloat{{index: 0, value: 1.0}}
	partialNthElement(data, 1)
	if data[0].value != 1.0 {
		t.Errorf("single element should be unchanged")
	}
}
