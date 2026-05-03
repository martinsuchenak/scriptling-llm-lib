package scriptlingllmlib

import (
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestSplitHeads(t *testing.T) {
	x := floatMatrix([]float64{1.0, 2.0, 3.0, 4.0})
	result := fnSplitHeads(ctx, noopKwargs, x, object.NewInteger(2))
	elems := evalList(t, result)
	if len(elems) != 2 {
		t.Fatalf("split_heads count = %d, want 2", len(elems))
	}

	head0 := evalFloatMatrix(t, elems[0])
	if len(head0) != 1 || len(head0[0]) != 2 {
		t.Fatalf("head0 shape = %dx%d, want 1x2", len(head0), len(head0[0]))
	}
	if head0[0][0] != 1.0 || head0[0][1] != 2.0 {
		t.Errorf("head0 = %v, want [[1,2]]", head0)
	}

	head1 := evalFloatMatrix(t, elems[1])
	if head1[0][0] != 3.0 || head1[0][1] != 4.0 {
		t.Errorf("head1 = %v, want [[3,4]]", head1)
	}

	assertError(t, fnSplitHeads(ctx, noopKwargs, x), "2 arguments")
	assertError(t, fnSplitHeads(ctx, noopKwargs, x, &object.String{Value: "x"}), "INTEGER")
	assertError(t, fnSplitHeads(ctx, noopKwargs, x, object.NewInteger(0)), "positive")
	assertError(t, fnSplitHeads(ctx, noopKwargs, x, object.NewInteger(3)), "divisible")
}

func TestMergeHeads(t *testing.T) {
	h0 := floatMatrix([]float64{1.0, 2.0})
	h1 := floatMatrix([]float64{3.0, 4.0})
	heads := &object.List{Elements: []object.Object{h0, h1}}

	result := fnMergeHeads(ctx, noopKwargs, heads)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 4 {
		t.Fatalf("merge_heads shape = %dx%d, want 1x4", len(mat), len(mat[0]))
	}
	if mat[0][0] != 1.0 || mat[0][1] != 2.0 || mat[0][2] != 3.0 || mat[0][3] != 4.0 {
		t.Errorf("merge_heads = %v", mat[0])
	}

	assertError(t, fnMergeHeads(ctx, noopKwargs), "1 argument")
	assertError(t, fnMergeHeads(ctx, noopKwargs, &object.String{Value: "x"}), "LIST")
	emptyList := &object.List{Elements: []object.Object{}}
	assertError(t, fnMergeHeads(ctx, noopKwargs, emptyList), "empty")
}

func TestRepeatKV(t *testing.T) {
	h0 := floatMatrix([]float64{1.0})
	h1 := floatMatrix([]float64{2.0})
	heads := &object.List{Elements: []object.Object{h0, h1}}

	result := fnRepeatKV(ctx, noopKwargs, heads, object.NewInteger(3))
	elems := evalList(t, result)
	if len(elems) != 6 {
		t.Fatalf("repeat_kv count = %d, want 6", len(elems))
	}

	assertError(t, fnRepeatKV(ctx, noopKwargs), "2 arguments")
	assertError(t, fnRepeatKV(ctx, noopKwargs, &object.String{Value: "x"}, object.NewInteger(2)), "LIST")
	assertError(t, fnRepeatKV(ctx, noopKwargs, heads, &object.String{Value: "x"}), "INTEGER")
	assertError(t, fnRepeatKV(ctx, noopKwargs, heads, object.NewInteger(0)), "positive")
}
