package scriptlingllmlib

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestGetScaleMinK4(t *testing.T) {
	scales := make([]byte, 12)
	scales[0] = 5
	scales[1] = 10
	scales[2] = 15
	scales[3] = 20
	scales[4] = 1
	scales[5] = 2
	scales[6] = 3
	scales[7] = 4
	scales[8] = 0x12
	scales[9] = 0x34
	scales[10] = 0x56
	scales[11] = 0x78

	sc, m := getScaleMinK4(0, scales)
	if sc != 5 || m != 1 {
		t.Errorf("getScaleMinK4(0) = (%d, %d), want (5, 1)", sc, m)
	}

	sc, m = getScaleMinK4(1, scales)
	if sc != 10 || m != 2 {
		t.Errorf("getScaleMinK4(1) = (%d, %d), want (10, 2)", sc, m)
	}

	sc, m = getScaleMinK4(2, scales)
	if sc != 15 || m != 3 {
		t.Errorf("getScaleMinK4(2) = (%d, %d), want (15, 3)", sc, m)
	}

	sc, m = getScaleMinK4(3, scales)
	if sc != 20 || m != 4 {
		t.Errorf("getScaleMinK4(3) = (%d, %d), want (20, 4)", sc, m)
	}

	sc, m = getScaleMinK4(4, scales)
	expectedSc := uint8(0x12&0xF) | uint8((scales[0]>>6)<<4)
	expectedM := uint8(0x12>>4) | uint8((scales[4]>>6)<<4)
	if sc != expectedSc || m != expectedM {
		t.Errorf("getScaleMinK4(4) = (%d, %d), want (%d, %d)", sc, m, expectedSc, expectedM)
	}
}

func TestLinearQ4K(t *testing.T) {
	scales := make([]byte, 12)
	for i := 0; i < 4; i++ {
		scales[i] = 1
	}
	qs := make([]byte, 128)
	for i := range qs {
		qs[i] = 0x11
	}
	block := makeQ4KBlock(1.0, 0.0, scales, qs)

	gpr := 1
	raw := &object.String{Value: string(block)}

	ones := make([]float64, 256)
	for i := range ones {
		ones[i] = 1.0
	}
	x := object.NewFloatArray2D(ones, 1, 256)

	result := fnLinearQ4K(ctx, noopKwargs, x, raw, object.NewInteger(int64(gpr)))
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 1 {
		t.Fatalf("linear_q4_k shape = %dx%d, want 1x1", len(mat), len(mat[0]))
	}
	if math.IsNaN(mat[0][0]) || math.IsInf(mat[0][0], 0) {
		t.Errorf("linear_q4_k = %f, expected finite", mat[0][0])
	}
}

func TestLinearRowQ4K(t *testing.T) {
	scales := make([]byte, 12)
	for i := 0; i < 4; i++ {
		scales[i] = 1
	}
	qs := make([]byte, 128)
	for i := range qs {
		qs[i] = 0x11
	}
	block := makeQ4KBlock(1.0, 0.0, scales, qs)

	raw := &object.String{Value: string(block)}

	row0 := make([]float64, 256)
	row1 := make([]float64, 256)
	for i := range row1 {
		row1[i] = 1.0
	}
	x := object.NewFloatArray2D(append(row0, row1...), 2, 256)

	result := fnLinearRowQ4K(ctx, noopKwargs, x, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 1 {
		t.Fatalf("linear_row_q4_k len = %d, want 1", len(vals))
	}
	if math.IsNaN(vals[0]) || math.IsInf(vals[0], 0) {
		t.Errorf("linear_row_q4_k = %f, expected finite", vals[0])
	}
}

func TestDequantizeQ4K(t *testing.T) {
	scales := make([]byte, 12)
	for i := 0; i < 4; i++ {
		scales[i] = 1
	}
	qs := make([]byte, 128)
	for i := range qs {
		qs[i] = 0
	}
	block := makeQ4KBlock(1.0, 0.0, scales, qs)
	raw := &object.String{Value: string(block)}

	result := fnDequantizeQ4K(ctx, noopKwargs, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 256 {
		t.Fatalf("dequantize_q4_k len = %d, want 256", len(vals))
	}
	for i, v := range vals {
		if v != 0.0 {
			t.Errorf("dequantize_q4_k[%d] = %f, want 0.0 (zero nibbles)", i, v)
		}
	}
}

func TestDequantizeQ4KRoundtrip(t *testing.T) {
	scales := make([]byte, 12)
	for i := 0; i < 4; i++ {
		scales[i] = 10
	}
	qs := make([]byte, 128)
	for i := range qs {
		qs[i] = 0x55
	}
	block := makeQ4KBlock(2.0, 0.0, scales, qs)
	raw := &object.String{Value: string(block)}

	result := fnDequantizeQ4K(ctx, noopKwargs, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)

	for i, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("dequantize_q4_k[%d] = %f, expected finite", i, v)
		}
	}
}

func TestLinearQ4KErrors(t *testing.T) {
	assertError(t, fnLinearQ4K(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearQ4K(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearQ4K(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, floatList(1.0)), "INTEGER")
}

func TestLinearRowQ4KErrors(t *testing.T) {
	assertError(t, fnLinearRowQ4K(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearRowQ4K(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearRowQ4K(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, floatList(1.0)), "INTEGER")
}

func TestDequantizeQ4KErrors(t *testing.T) {
	assertError(t, fnDequantizeQ4K(ctx, noopKwargs), "2 arguments")
	assertError(t, fnDequantizeQ4K(ctx, noopKwargs, floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnDequantizeQ4K(ctx, noopKwargs, &object.String{Value: "x"}, floatList(1.0)), "INTEGER")
}

func TestQ4KDotBlockSlowEquivalence(t *testing.T) {
	scales := make([]byte, 12)
	for i := 0; i < 4; i++ {
		scales[i] = 5
	}
	qs := make([]byte, 128)
	for i := range qs {
		qs[i] = byte(i)
	}
	block := makeQ4KBlock(1.5, 0.25, scales, qs)

	x := make([]float64, 256)
	for i := range x {
		x[i] = float64(i%10) * 0.1
	}

	regular := q4kDotBlock(block, 0, x, 0)
	slow := q4kDotBlockSlow(block, 0, x, 0)

	if math.Abs(regular-slow) > 1e-10 {
		t.Errorf("q4kDotBlockSlow = %f, q4kDotBlock = %f", slow, regular)
	}
}

func TestQ4KLinearViaDict(t *testing.T) {
	scales := make([]byte, 12)
	for i := 0; i < 4; i++ {
		scales[i] = 1
	}
	qs := make([]byte, 128)
	block := makeQ4KBlock(1.0, 0.0, scales, qs)

	d := object.NewStringDict(map[string]object.Object{
		"q4k":           object.NewInteger(1),
		"raw":           &object.String{Value: string(block)},
		"blocks_per_row": object.NewInteger(1),
	})

	ones := make([]float64, 256)
	for i := range ones {
		ones[i] = 1.0
	}
	x := object.NewFloatArray2D(ones, 1, 256)

	wQ := d
	wK := d
	wV := d

	result := fnFusedQKV(ctx, noopKwargs, x, wQ, wK, wV)
	elems := evalList(t, result)
	if len(elems) != 3 {
		t.Fatalf("fused_qkv via dict returned %d elements, want 3", len(elems))
	}
	for i, name := range []string{"q", "k", "v"} {
		mat := evalFloatMatrix(t, elems[i])
		if len(mat) != 1 || len(mat[0]) != 1 {
			t.Fatalf("%s shape = %dx%d, want 1x1", name, len(mat), len(mat[0]))
		}
		if math.IsNaN(mat[0][0]) {
			t.Errorf("%s is NaN", name)
		}
	}
}

func TestQ4KLinearViaDictDirect(t *testing.T) {
	scales := make([]byte, 12)
	for i := 0; i < 4; i++ {
		scales[i] = 1
	}
	qs := make([]byte, 128)
	for i := range qs {
		qs[i] = 0x11
	}
	block := makeQ4KBlock(1.0, 0.0, scales, qs)

	ones := make([]float64, 256)
	for i := range ones {
		ones[i] = 1.0
	}
	x := object.NewFloatArray2D(ones, 1, 256)

	rawStr := &object.String{Value: string(block)}

	result := fnLinearQ4K(ctx, noopKwargs, x, rawStr, object.NewInteger(1))
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 1 {
		t.Fatalf("shape = %dx%d, want 1x1", len(mat), len(mat[0]))
	}

	rowBytes := 1 * 144
	outFeatures := len(block) / rowBytes
	if outFeatures != 1 {
		t.Fatalf("outFeatures = %d, want 1", outFeatures)
	}
}

func TestQ4KBlockLayout(t *testing.T) {
	block := make([]byte, 144)
	binary.LittleEndian.PutUint16(block[0:2], float64ToFloat16(1.0))
	binary.LittleEndian.PutUint16(block[2:4], float64ToFloat16(0.0))

	scales := block[4:16]
	for i := 0; i < 4; i++ {
		scales[i] = 1
	}
	scales[4] = 0
	scales[5] = 0
	scales[6] = 0
	scales[7] = 0
	for i := 8; i < 12; i++ {
		scales[i] = 0
	}

	qs := block[16:144]
	for i := range qs {
		qs[i] = 0x88
	}

	x := make([]float64, 256)
	for i := range x {
		x[i] = 1.0
	}

	result := q4kDotBlock(block, 0, x, 0)
	if math.IsNaN(result) || math.IsInf(result, 0) {
		t.Fatalf("q4kDotBlock = %f, expected finite", result)
	}
}
