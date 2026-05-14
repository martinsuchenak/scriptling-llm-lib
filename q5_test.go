package scriptlingllmlib

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func makeQ5Group(scale float64, qh uint32, qs []byte) []byte {
	group := make([]byte, 22)
	binary.LittleEndian.PutUint16(group[0:2], float64ToFloat16(scale))
	binary.LittleEndian.PutUint32(group[2:6], qh)
	copy(group[6:22], qs)
	return group
}

func TestQ5DotGroupXEquivalence(t *testing.T) {
	qs := make([]byte, 16)
	for i := range qs {
		qs[i] = byte(i)
	}
	var qh uint32 = 0x00010000

	group := makeQ5Group(0.5, qh, qs)

	xData := make([]float64, 32)
	for i := range xData {
		xData[i] = float64(i) * 0.1
	}

	resultX := q5DotGroupX(group, 0, xData, 0)
	resultSlow := q5DotGroup(group, 0, xData, 0)

	if math.IsNaN(resultX) || math.IsInf(resultX, 0) {
		t.Fatalf("q5DotGroupX = %f, expected finite", resultX)
	}
	if math.Abs(resultX-resultSlow) > 1e-10 {
		t.Errorf("q5DotGroupX = %f, q5DotGroup = %f, diff = %e", resultX, resultSlow, math.Abs(resultX-resultSlow))
	}
}

func TestLinearQ5Fast(t *testing.T) {
	qs := make([]byte, 16)
	for i := range qs {
		qs[i] = 0x00
	}
	group := makeQ5Group(1.0, 0, qs)

	raw := object.NewString(string(group))
	ones := make([]float64, 32)
	for i := range ones {
		ones[i] = 1.0
	}
	x := object.NewFloatArray2D(ones, 1, 32)

	result := fnLinearQ5Fast(ctx, noopKwargs, x, raw, object.NewInteger(1))
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 1 {
		t.Fatalf("linear_q5 shape = %dx%d, want 1x1", len(mat), len(mat[0]))
	}
	if math.IsNaN(mat[0][0]) || math.IsInf(mat[0][0], 0) {
		t.Errorf("linear_q5 = %f, expected finite", mat[0][0])
	}
}

func TestLinearRowQ5Fast(t *testing.T) {
	qs := make([]byte, 16)
	for i := range qs {
		qs[i] = 0x00
	}
	group := makeQ5Group(1.0, 0, qs)

	raw := object.NewString(string(group))

	row0 := make([]float64, 32)
	row1 := make([]float64, 32)
	for i := range row1 {
		row1[i] = 1.0
	}
	x := object.NewFloatArray2D(append(row0, row1...), 2, 32)

	result := fnLinearRowQ5Fast(ctx, noopKwargs, x, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 1 {
		t.Fatalf("linear_row_q5 len = %d, want 1", len(vals))
	}
	if math.IsNaN(vals[0]) || math.IsInf(vals[0], 0) {
		t.Errorf("linear_row_q5 = %f, expected finite", vals[0])
	}
}

func TestDequantizeQ5_0(t *testing.T) {
	qs := make([]byte, 16)
	for i := range qs {
		qs[i] = 0x00
	}
	group := makeQ5Group(1.0, 0, qs)
	raw := object.NewString(string(group))

	result := fnDequantizeQ5_0(ctx, noopKwargs, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 32 {
		t.Fatalf("dequantize_q5_0 len = %d, want 32", len(vals))
	}
	for i, v := range vals {
		if v != -16.0 {
			t.Errorf("dequantize_q5_0[%d] = %f, want -16.0 (zero nibbles + zero high bits)", i, v)
		}
	}
}

func TestDequantizeQ5_0WithHighBits(t *testing.T) {
	qs := make([]byte, 16)
	for i := range qs {
		qs[i] = 0xFF
	}
	var qh uint32 = 0xFFFFFFFF
	group := makeQ5Group(2.0, qh, qs)
	raw := object.NewString(string(group))

	result := fnDequantizeQ5_0(ctx, noopKwargs, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 32 {
		t.Fatalf("dequantize_q5_0 len = %d, want 32", len(vals))
	}
	for i, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("dequantize_q5_0[%d] = %f, expected finite", i, v)
		}
	}
}

func TestLinearQ5FastErrors(t *testing.T) {
	assertError(t, fnLinearQ5Fast(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearQ5Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearQ5Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), object.NewString("x"), floatList(1.0)), "INTEGER")
	assertError(t, fnLinearQ5Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), object.NewString("x"), object.NewInteger(0)), "positive")
}

func TestLinearRowQ5FastErrors(t *testing.T) {
	assertError(t, fnLinearRowQ5Fast(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearRowQ5Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearRowQ5Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), object.NewString("x"), floatList(1.0)), "INTEGER")
	assertError(t, fnLinearRowQ5Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), object.NewString("x"), object.NewInteger(0)), "positive")
}

func TestDequantizeQ5_0Errors(t *testing.T) {
	assertError(t, fnDequantizeQ5_0(ctx, noopKwargs), "2 arguments")
	assertError(t, fnDequantizeQ5_0(ctx, noopKwargs, floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnDequantizeQ5_0(ctx, noopKwargs, object.NewString("x"), floatList(1.0)), "INTEGER")
}

func TestLinearQ5FastDimensionMismatch(t *testing.T) {
	qs := make([]byte, 16)
	group := makeQ5Group(1.0, 0, qs)
	raw := object.NewString(string(group))

	x := object.NewFloatArray2D([]float64{1.0, 1.0}, 1, 2)
	assertError(t, fnLinearQ5Fast(ctx, noopKwargs, x, raw, object.NewInteger(1)), "columns")
}

func TestLinearQ5ViaDict(t *testing.T) {
	qs := make([]byte, 16)
	for i := range qs {
		qs[i] = 0x00
	}
	group := makeQ5Group(1.0, 0, qs)
	rawStr := string(group)

	d := object.NewStringDict(map[string]object.Object{
		"q5":             object.NewInteger(1),
		"raw":            object.NewString(rawStr),
		"groups_per_row": object.NewInteger(1),
	})

	ones := make([]float64, 32)
	for i := range ones {
		ones[i] = 1.0
	}
	x := object.NewFloatArray2D(ones, 1, 32)

	result := fnFusedQKV(ctx, noopKwargs, x, d, d, d)
	elems := evalList(t, result)
	if len(elems) != 3 {
		t.Fatalf("fused_qkv Q5 dict returned %d elements, want 3", len(elems))
	}
}
