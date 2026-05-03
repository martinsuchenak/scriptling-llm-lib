package scriptlingllmlib

import (
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestArgmax(t *testing.T) {
	result := fnArgmax(ctx, noopKwargs, floatList(1.0, 3.0, 2.0))
	if evalInt(t, result) != 1 {
		t.Errorf("argmax([1,3,2]) = %d, want 1", evalInt(t, result))
	}

	result = fnArgmax(ctx, noopKwargs, floatList(5.0))
	if evalInt(t, result) != 0 {
		t.Errorf("argmax([5]) = %d, want 0", evalInt(t, result))
	}

	assertError(t, fnArgmax(ctx, noopKwargs), "1 argument")
	assertError(t, fnArgmax(ctx, noopKwargs, floatList()), "empty")
	assertError(t, fnArgmax(ctx, noopKwargs, &object.String{Value: "x"}), "LIST")
}

func TestArgmin(t *testing.T) {
	result := fnArgmin(ctx, noopKwargs, floatList(3.0, 1.0, 2.0))
	if evalInt(t, result) != 1 {
		t.Errorf("argmin([3,1,2]) = %d, want 1", evalInt(t, result))
	}

	result = fnArgmin(ctx, noopKwargs, floatList(-1.0, 0.0, 1.0))
	if evalInt(t, result) != 0 {
		t.Errorf("argmin([-1,0,1]) = %d, want 0", evalInt(t, result))
	}

	assertError(t, fnArgmin(ctx, noopKwargs, floatList()), "empty")
	assertError(t, fnArgmin(ctx, noopKwargs), "1 argument")
	assertError(t, fnArgmin(ctx, noopKwargs, &object.String{Value: "x"}), "LIST")
}

func TestTopk(t *testing.T) {
	result := fnTopk(ctx, noopKwargs, floatList(1.0, 5.0, 3.0, 4.0, 2.0), object.NewInteger(3))
	elems := evalList(t, result)
	if len(elems) != 3 {
		t.Fatalf("topk len = %d, want 3", len(elems))
	}

	pairs := make([]struct {
		idx int64
		val float64
	}, 3)
	for i, e := range elems {
		pair := evalList(t, e)
		pairs[i].idx = evalInt(t, pair[0])
		pairs[i].val = evalFloat(t, pair[1])
	}
	if pairs[0].idx != 1 || pairs[0].val != 5.0 {
		t.Errorf("topk[0] = (%d, %f), want (1, 5.0)", pairs[0].idx, pairs[0].val)
	}
	if pairs[1].idx != 3 || pairs[1].val != 4.0 {
		t.Errorf("topk[1] = (%d, %f), want (3, 4.0)", pairs[1].idx, pairs[1].val)
	}
	if pairs[2].idx != 2 || pairs[2].val != 3.0 {
		t.Errorf("topk[2] = (%d, %f), want (2, 3.0)", pairs[2].idx, pairs[2].val)
	}

	result = fnTopk(ctx, noopKwargs, floatList(1.0, 2.0), object.NewInteger(10))
	elems = evalList(t, result)
	if len(elems) != 2 {
		t.Errorf("topk clamped len = %d, want 2", len(elems))
	}

	assertError(t, fnTopk(ctx, noopKwargs, floatList(1.0), object.NewInteger(0)), "positive")
	assertError(t, fnTopk(ctx, noopKwargs), "2 arguments")
	assertError(t, fnTopk(ctx, noopKwargs, &object.String{Value: "x"}, object.NewInteger(1)), "LIST")
	assertError(t, fnTopk(ctx, noopKwargs, floatList(1.0), &object.String{Value: "x"}), "INTEGER")
}

func TestClip(t *testing.T) {
	result := fnClip(ctx, noopKwargs, floatList(-2.0, 0.5, 3.0, 1.0), &object.Float{Value: -1.0}, &object.Float{Value: 2.0})
	vals := evalFloatList(t, result)
	expected := []float64{-1.0, 0.5, 2.0, 1.0}
	for i, v := range vals {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("clip[%d] = %f, want %f", i, v, expected[i])
		}
	}

	result = fnClip(ctx, noopKwargs, &object.Float{Value: 5.0}, &object.Float{Value: 0.0}, &object.Float{Value: 3.0})
	if evalFloat(t, result) != 3.0 {
		t.Errorf("clip(5, 0, 3) = %f, want 3.0", evalFloat(t, result))
	}

	assertError(t, fnClip(ctx, noopKwargs, &object.Float{Value: 1.0}, &object.Float{Value: 5.0}, &object.Float{Value: 3.0}), "lo must be <= hi")
	assertError(t, fnClip(ctx, noopKwargs), "3 arguments")
	assertError(t, fnClip(ctx, noopKwargs, &object.String{Value: "x"}, &object.Float{Value: 0.0}, &object.Float{Value: 1.0}), "INTEGER, FLOAT, or LIST")
	assertError(t, fnClip(ctx, noopKwargs, floatList(1.0), &object.String{Value: "x"}, &object.Float{Value: 1.0}), "INTEGER or FLOAT")
	assertError(t, fnClip(ctx, noopKwargs, floatList(1.0), &object.Float{Value: 0.0}, &object.String{Value: "x"}), "INTEGER or FLOAT")
	strInList := &object.List{Elements: []object.Object{&object.String{Value: "x"}}}
	assertError(t, fnClip(ctx, noopKwargs, strInList, &object.Float{Value: 0.0}, &object.Float{Value: 1.0}), "INTEGER or FLOAT")
}
