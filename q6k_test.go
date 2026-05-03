package scriptlingllmlib

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func makeQ6KBlock(d float64, ql, qh, scales []byte) []byte {
	block := make([]byte, 210)
	copy(block[0:128], ql)
	copy(block[128:192], qh)
	copy(block[192:208], scales)
	binary.LittleEndian.PutUint16(block[208:210], float64ToFloat16(d))
	return block
}

func TestQ6KDotBlock(t *testing.T) {
	ql := make([]byte, 128)
	qh := make([]byte, 64)
	scales := make([]byte, 16)
	for i := range scales {
		scales[i] = 1
	}
	block := makeQ6KBlock(1.0, ql, qh, scales)

	x := make([]float64, 256)
	for i := range x {
		x[i] = 0.0
	}

	result := q6kDotBlock(block, 0, x, 0)
	if result != 0.0 {
		t.Errorf("q6kDotBlock with zero input = %f, want 0.0", result)
	}
}

func TestQ6KDotBlockNonZero(t *testing.T) {
	ql := make([]byte, 128)
	qh := make([]byte, 64)
	scales := make([]byte, 16)
	for i := range scales {
		scales[i] = 1
	}

	for i := range ql {
		ql[i] = 0x20
	}
	for i := range qh {
		qh[i] = 0
	}

	block := makeQ6KBlock(1.0, ql, qh, scales)

	x := make([]float64, 256)
	for i := range x {
		x[i] = 1.0
	}

	result := q6kDotBlock(block, 0, x, 0)
	if math.IsNaN(result) || math.IsInf(result, 0) {
		t.Fatalf("q6kDotBlock = %f, expected finite", result)
	}
}

func TestLinearQ6K(t *testing.T) {
	ql := make([]byte, 128)
	qh := make([]byte, 64)
	scales := make([]byte, 16)
	for i := range scales {
		scales[i] = 1
	}
	block := makeQ6KBlock(1.0, ql, qh, scales)

	raw := &object.String{Value: string(block)}
	ones := make([]float64, 256)
	for i := range ones {
		ones[i] = 1.0
	}
	x := object.NewFloatArray2D(ones, 1, 256)

	result := fnLinearQ6K(ctx, noopKwargs, x, raw, object.NewInteger(1))
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 1 {
		t.Fatalf("linear_q6_k shape = %dx%d, want 1x1", len(mat), len(mat[0]))
	}
	if math.IsNaN(mat[0][0]) || math.IsInf(mat[0][0], 0) {
		t.Errorf("linear_q6_k = %f, expected finite", mat[0][0])
	}
}

func TestLinearRowQ6K(t *testing.T) {
	ql := make([]byte, 128)
	qh := make([]byte, 64)
	scales := make([]byte, 16)
	for i := range scales {
		scales[i] = 1
	}
	block := makeQ6KBlock(1.0, ql, qh, scales)

	raw := &object.String{Value: string(block)}

	row0 := make([]float64, 256)
	row1 := make([]float64, 256)
	for i := range row1 {
		row1[i] = 1.0
	}
	x := object.NewFloatArray2D(append(row0, row1...), 2, 256)

	result := fnLinearRowQ6K(ctx, noopKwargs, x, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 1 {
		t.Fatalf("linear_row_q6_k len = %d, want 1", len(vals))
	}
	if math.IsNaN(vals[0]) || math.IsInf(vals[0], 0) {
		t.Errorf("linear_row_q6_k = %f, expected finite", vals[0])
	}
}

func TestDequantizeQ6K(t *testing.T) {
	ql := make([]byte, 128)
	qh := make([]byte, 64)
	scales := make([]byte, 16)
	block := makeQ6KBlock(1.0, ql, qh, scales)
	raw := &object.String{Value: string(block)}

	result := fnDequantizeQ6K(ctx, noopKwargs, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 256 {
		t.Fatalf("dequantize_q6_k len = %d, want 256", len(vals))
	}
	for i, v := range vals {
		if v != 0.0 {
			t.Errorf("dequantize_q6_k[%d] = %f, want 0.0", i, v)
		}
	}
}

func TestDequantizeQ6KNonZero(t *testing.T) {
	ql := make([]byte, 128)
	qh := make([]byte, 64)
	scales := make([]byte, 16)
	for i := range scales {
		scales[i] = 10
	}
	for i := range ql {
		ql[i] = 0x20
	}
	block := makeQ6KBlock(2.0, ql, qh, scales)
	raw := &object.String{Value: string(block)}

	result := fnDequantizeQ6K(ctx, noopKwargs, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 256 {
		t.Fatalf("dequantize_q6_k len = %d, want 256", len(vals))
	}
	for i, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("dequantize_q6_k[%d] = %f, expected finite", i, v)
		}
	}
}

func TestDequantizeQ6KBlock(t *testing.T) {
	ql := make([]byte, 128)
	qh := make([]byte, 64)
	scales := make([]byte, 16)
	scales[0] = 5
	scales[1] = 5
	scales[2] = 5
	scales[3] = 5
	scales[4] = 5
	scales[5] = 5
	scales[6] = 5
	scales[7] = 5

	ql[0] = 0x0F
	qh[0] = 0x00

	block := makeQ6KBlock(1.0, ql, qh, scales)
	out := make([]float64, 256)
	dequantizeQ6KBlock(block, 0, out, 0)

	qVal := int(int8(0x0F&0xF)) - 32
	expected := 1.0 * float64(int8(5)) * float64(qVal)
	if math.Abs(out[0]-expected) > 1e-10 {
		t.Errorf("dequantizeQ6KBlock[0] = %f, want %f", out[0], expected)
	}
}

func TestLinearQ6KErrors(t *testing.T) {
	assertError(t, fnLinearQ6K(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearQ6K(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearQ6K(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, floatList(1.0)), "INTEGER")
}

func TestLinearRowQ6KErrors(t *testing.T) {
	assertError(t, fnLinearRowQ6K(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearRowQ6K(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearRowQ6K(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, floatList(1.0)), "INTEGER")
}

func TestDequantizeQ6KErrors(t *testing.T) {
	assertError(t, fnDequantizeQ6K(ctx, noopKwargs), "2 arguments")
	assertError(t, fnDequantizeQ6K(ctx, noopKwargs, floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnDequantizeQ6K(ctx, noopKwargs, &object.String{Value: "x"}, floatList(1.0)), "INTEGER")
}

func TestLinearQ6KDimensionMismatch(t *testing.T) {
	ql := make([]byte, 128)
	qh := make([]byte, 64)
	scales := make([]byte, 16)
	block := makeQ6KBlock(1.0, ql, qh, scales)
	raw := &object.String{Value: string(block)}

	x := object.NewFloatArray2D([]float64{1.0, 1.0}, 1, 2)
	assertError(t, fnLinearQ6K(ctx, noopKwargs, x, raw, object.NewInteger(1)), "columns")
}

func TestQ6KLinearViaDict(t *testing.T) {
	ql := make([]byte, 128)
	qh := make([]byte, 64)
	scales := make([]byte, 16)
	for i := range scales {
		scales[i] = 1
	}
	block := makeQ6KBlock(1.0, ql, qh, scales)
	rawStr := string(block)

	d := object.NewStringDict(map[string]object.Object{
		"q6k":            object.NewInteger(1),
		"raw":            &object.String{Value: rawStr},
		"blocks_per_row": object.NewInteger(1),
	})

	ones := make([]float64, 256)
	for i := range ones {
		ones[i] = 1.0
	}
	x := object.NewFloatArray2D(ones, 1, 256)

	result := fnFusedQKV(ctx, noopKwargs, x, d, d, d)
	elems := evalList(t, result)
	if len(elems) != 3 {
		t.Fatalf("fused_qkv Q6_K dict returned %d elements, want 3", len(elems))
	}
}
