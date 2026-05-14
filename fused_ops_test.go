package scriptlingllmlib

import (
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestFusedQKVFloat(t *testing.T) {
	x := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)
	wQ := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)
	wK := object.NewFloatArray2D([]float64{0.0, 1.0, 1.0, 0.0}, 2, 2)
	wV := object.NewFloatArray2D([]float64{1.0, 1.0, 1.0, 1.0}, 2, 2)

	result := fnFusedQKV(ctx, noopKwargs, x, wQ, wK, wV)
	elems := evalList(t, result)
	if len(elems) != 3 {
		t.Fatalf("fused_qkv returned %d elements, want 3", len(elems))
	}

	qMat := evalFloatMatrix(t, elems[0])
	if len(qMat) != 2 || len(qMat[0]) != 2 {
		t.Fatalf("q shape = %dx%d, want 2x2", len(qMat), len(qMat[0]))
	}
	if qMat[0][0] != 1.0 || qMat[0][1] != 0.0 {
		t.Errorf("q[0] = %v, want [1, 0]", qMat[0])
	}
	if qMat[1][0] != 0.0 || qMat[1][1] != 1.0 {
		t.Errorf("q[1] = %v, want [0, 1]", qMat[1])
	}

	kMat := evalFloatMatrix(t, elems[1])
	if kMat[0][0] != 0.0 || kMat[0][1] != 1.0 {
		t.Errorf("k[0] = %v, want [0, 1]", kMat[0])
	}
}

func TestFusedQKVWithQ8(t *testing.T) {
	ones := make([]float64, 32)
	for i := range ones {
		ones[i] = 1.0
	}
	x := object.NewFloatArray2D(ones, 1, 32)

	qValues := make([]int8, 32)
	for i := range qValues {
		qValues[i] = 1
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))

	result := fnFusedQKV(ctx, noopKwargs, x, raw, raw, raw)
	elems := evalList(t, result)
	if len(elems) != 3 {
		t.Fatalf("fused_qkv Q8 returned %d elements, want 3", len(elems))
	}

	for i, name := range []string{"q", "k", "v"} {
		mat := evalFloatMatrix(t, elems[i])
		if len(mat) != 1 || len(mat[0]) != 1 {
			t.Fatalf("%s shape = %dx%d, want 1x1", name, len(mat), len(mat[0]))
		}
		if math.IsNaN(mat[0][0]) || math.IsInf(mat[0][0], 0) {
			t.Errorf("%s[0][0] = %f, expected finite", name, mat[0][0])
		}
	}
}

func TestFusedQKVErrors(t *testing.T) {
	assertError(t, fnFusedQKV(ctx, noopKwargs), "4 arguments")
	assertError(t, fnFusedQKV(ctx, noopKwargs, object.NewString("x")), "4 arguments")

	x := object.NewFloatArray2D([]float64{1.0}, 1, 1)
	assertError(t, fnFusedQKV(ctx, noopKwargs, object.NewInteger(1), x, x, x), "FLOAT_MATRIX")

	emptyMat := object.NewFloatArray2D([]float64{}, 0, 0)
	assertError(t, fnFusedQKV(ctx, noopKwargs, emptyMat, x, x, x), "empty")
}

func TestFusedFFNFloat(t *testing.T) {
	x := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)

	wGate := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)
	wUp := object.NewFloatArray2D([]float64{1.0, 1.0, 1.0, 1.0}, 2, 2)
	wDown := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)

	result := fnFusedFFN(ctx, noopKwargs, x, wGate, wUp, wDown)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 2 || len(mat[0]) != 2 {
		t.Fatalf("fused_ffn shape = %dx%d, want 2x2", len(mat), len(mat[0]))
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.IsNaN(mat[i][j]) || math.IsInf(mat[i][j], 0) {
				t.Errorf("fused_ffn[%d][%d] = %f, expected finite", i, j, mat[i][j])
			}
		}
	}
}

func TestFusedFFNErrors(t *testing.T) {
	assertError(t, fnFusedFFN(ctx, noopKwargs), "4 arguments")

	x := object.NewFloatArray2D([]float64{1.0}, 1, 1)
	assertError(t, fnFusedFFN(ctx, noopKwargs, object.NewInteger(1), x, x, x), "FLOAT_MATRIX")

	emptyMat := object.NewFloatArray2D([]float64{}, 0, 0)
	assertError(t, fnFusedFFN(ctx, noopKwargs, emptyMat, x, x, x), "empty")
}

func TestFusedRopeBatch(t *testing.T) {
	h0 := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 1, 4)
	h1 := object.NewFloatArray2D([]float64{0.0, 1.0, 1.0, 0.0}, 1, 4)
	headList := &object.List{Elements: []object.Object{h0, h1}}

	kwargs := object.NewKwargs(map[string]object.Object{
		"rope_dim": object.NewInteger(4),
	})
	result := fnFusedRopeBatch(ctx, kwargs, headList, object.NewInteger(0))

	elems := evalList(t, result)
	if len(elems) != 2 {
		t.Fatalf("fused_rope_batch returned %d heads, want 2", len(elems))
	}

	mat0 := evalFloatMatrix(t, elems[0])
	if len(mat0) != 1 || len(mat0[0]) != 4 {
		t.Fatalf("head0 shape = %dx%d, want 1x4", len(mat0), len(mat0[0]))
	}

	ropeDim := 4
	freqBase := 10000.0
	freq := 1.0 / math.Pow(freqBase, 0.0/float64(ropeDim))
	angle := freq * 0.0
	cosA := math.Cos(angle)
	sinA := math.Sin(angle)
	expected0 := 1.0*cosA - 0.0*sinA
	if math.Abs(mat0[0][0]-expected0) > 1e-10 {
		t.Errorf("rope head0[0][0] = %f, want %f", mat0[0][0], expected0)
	}
}

func TestFusedRopeBatchErrors(t *testing.T) {
	assertError(t, fnFusedRopeBatch(ctx, noopKwargs), "2 arguments")
	assertError(t, fnFusedRopeBatch(ctx, noopKwargs, object.NewInteger(1), object.NewInteger(0)), "LIST")
	assertError(t, fnFusedRopeBatch(ctx, noopKwargs, &object.List{Elements: []object.Object{object.NewInteger(1)}}, object.NewInteger(0)), "not a matrix")
}

func TestFusedAttention(t *testing.T) {
	q0 := object.NewFloatArray2D([]float64{1.0, 0.0}, 1, 2)
	q1 := object.NewFloatArray2D([]float64{0.0, 1.0}, 1, 2)
	k0 := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)
	k1 := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)
	v0 := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)
	v1 := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)

	qList := &object.List{Elements: []object.Object{q0, q1}}
	kList := &object.List{Elements: []object.Object{k0, k1}}
	vList := &object.List{Elements: []object.Object{v0, v1}}

	result := fnFusedAttention(ctx, noopKwargs, qList, kList, vList, object.NewBoolean(false))
	elems := evalList(t, result)
	if len(elems) != 2 {
		t.Fatalf("fused_attention returned %d heads, want 2", len(elems))
	}

	mat0 := evalFloatMatrix(t, elems[0])
	if len(mat0) != 1 || len(mat0[0]) != 2 {
		t.Fatalf("head0 shape = %dx%d, want 1x2", len(mat0), len(mat0[0]))
	}

	total := mat0[0][0] + mat0[0][1]
	if total < 0.99 || total > 1.01 {
		t.Errorf("attention outputs should sum to ~1.0: got %f", total)
	}
}

func TestFusedAttentionCausal(t *testing.T) {
	q0 := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)
	k0 := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)
	v0 := object.NewFloatArray2D([]float64{1.0, 0.0, 0.0, 1.0}, 2, 2)

	qList := &object.List{Elements: []object.Object{q0}}
	kList := &object.List{Elements: []object.Object{k0}}
	vList := &object.List{Elements: []object.Object{v0}}

	result := fnFusedAttention(ctx, noopKwargs, qList, kList, vList, object.NewBoolean(true))
	elems := evalList(t, result)
	mat := evalFloatMatrix(t, elems[0])

	if mat[0][0] < 0.99 {
		t.Errorf("causal row 0 should attend only to pos 0: got %f", mat[0][0])
	}
}

func TestFusedAttentionErrors(t *testing.T) {
	assertError(t, fnFusedAttention(ctx, noopKwargs), "3 arguments")
	assertError(t, fnFusedAttention(ctx, noopKwargs, object.NewInteger(1)), "3 arguments")
	assertError(t, fnFusedAttention(ctx, noopKwargs, object.NewInteger(1), &object.List{Elements: nil}, &object.List{Elements: nil}), "LIST")
	assertError(t, fnFusedAttention(ctx, noopKwargs, &object.List{Elements: nil}, object.NewInteger(1), &object.List{Elements: nil}), "LIST")
	assertError(t, fnFusedAttention(ctx, noopKwargs, &object.List{Elements: nil}, &object.List{Elements: nil}, object.NewInteger(1)), "LIST")
	assertError(t, fnFusedAttention(ctx, noopKwargs, &object.List{Elements: []object.Object{}}, &object.List{Elements: nil}, &object.List{Elements: nil}), "empty")
}
