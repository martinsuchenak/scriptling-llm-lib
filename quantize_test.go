package scriptlingllmlib

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestLinearQ8Scriptling(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Q8 linear test in short mode")
	}

	nCols := 32
	nRows := 2

	weightData := make([]float64, nRows*nCols)
	for i := range weightData {
		weightData[i] = 0.01
	}
	raw := quantizeQ8RowsPure(weightData, nRows, nCols)
	groupsPerRow := nCols / 32

	x := floatMatrix([]float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0})

	w := &object.String{Value: string(raw)}
	gpr := object.NewInteger(int64(groupsPerRow))

	result := fnLinearQ8(ctx, noopKwargs, x, w, gpr)
	mat := evalFloatMatrix(t, result)

	if len(mat) != 1 || len(mat[0]) != nRows {
		t.Fatalf("shape = %dx%d, want 1x%d", len(mat), len(mat[0]), nRows)
	}

	for _, v := range mat[0] {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("linear_q8 result = %f, expected finite", v)
		}
	}
}

func TestLinearQ4Scriptling(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Q4 linear test in short mode")
	}

	nCols := 32
	nRows := 2

	q4Raw := make([]byte, nRows*18)
	for i := range q4Raw {
		q4Raw[i] = 0
	}
	scaleF16 := float64ToFloat16(0.01)
	binary.LittleEndian.PutUint16(q4Raw[0:2], scaleF16)
	binary.LittleEndian.PutUint16(q4Raw[18:20], scaleF16)

	x := floatMatrix([]float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0})

	w := &object.String{Value: string(q4Raw)}
	gpr := object.NewInteger(int64(nCols / 32))

	result := fnLinearQ4(ctx, noopKwargs, x, w, gpr)
	mat := evalFloatMatrix(t, result)

	if len(mat) != 1 || len(mat[0]) != nRows {
		t.Fatalf("shape = %dx%d, want 1x%d", len(mat), len(mat[0]), nRows)
	}

	for _, v := range mat[0] {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("linear_q4 result = %f, expected finite", v)
		}
	}
}

func TestQuantizeQ8(t *testing.T) {
	nCols := 32
	nRows := 2
	flatData := make([]float64, nRows*nCols)
	for i := range flatData {
		flatData[i] = float64(i%10) * 0.1
	}

	result := fnQuantizeQ8(ctx, noopKwargs, floatList(flatData...), object.NewInteger(int64(nRows)), object.NewInteger(int64(nCols)))
	raw, ok := result.(*object.String)
	if !ok {
		t.Fatalf("quantize_q8 returned %s, want STRING", result.Type().String())
	}

	rawBytes := []byte(raw.Value)
	expectedLen := nRows * (nCols / 32) * 34
	if len(rawBytes) != expectedLen {
		t.Fatalf("quantize_q8 raw len = %d, want %d", len(rawBytes), expectedLen)
	}
}

func TestQuantizeQ8Roundtrip(t *testing.T) {
	nCols := 32
	nRows := 1
	values := make([]float64, nCols)
	for i := range values {
		values[i] = 1.0
	}

	result := fnQuantizeQ8(ctx, noopKwargs, floatList(values...), object.NewInteger(int64(nRows)), object.NewInteger(int64(nCols)))
	raw, ok := result.(*object.String)
	if !ok {
		t.Fatalf("quantize_q8 returned %s", result.Type().String())
	}

	dequantResult := fnDequantizeQ8_0(ctx, noopKwargs, raw, object.NewInteger(int64(nCols/32)))
	dequantVals := evalFloatList(t, dequantResult)
	if len(dequantVals) != nCols {
		t.Fatalf("dequant len = %d, want %d", len(dequantVals), nCols)
	}
	for i, v := range dequantVals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("dequant[%d] = %f, expected finite", i, v)
		}
		if math.Abs(v-1.0) > 0.02 {
			t.Errorf("roundtrip[%d] = %f, want ~1.0", i, v)
		}
	}
}

func TestQuantizeQ8Errors(t *testing.T) {
	assertError(t, fnQuantizeQ8(ctx, noopKwargs), "3 arguments")
	assertError(t, fnQuantizeQ8(ctx, noopKwargs, &object.String{Value: "x"}, object.NewInteger(1), object.NewInteger(32)), "LIST")
	assertError(t, fnQuantizeQ8(ctx, noopKwargs, floatList(make([]float64, 32)...), &object.String{Value: "x"}, object.NewInteger(32)), "INTEGER")
	assertError(t, fnQuantizeQ8(ctx, noopKwargs, floatList(make([]float64, 32)...), object.NewInteger(1), object.NewInteger(31)), "divisible by 32")
	assertError(t, fnQuantizeQ8(ctx, noopKwargs, floatList(make([]float64, 10)...), object.NewInteger(1), object.NewInteger(32)), "must equal rows*cols")
}

func TestQuantizeQ8Rows(t *testing.T) {
	nCols := 32
	matrix := floatMatrix(make([]float64, nCols), make([]float64, nCols))
	for i := 0; i < nCols; i++ {
		matrix.(*object.List).Elements[0].(*object.List).Elements[i] = &object.Float{Value: 1.0}
		matrix.(*object.List).Elements[1].(*object.List).Elements[i] = &object.Float{Value: -1.0}
	}

	result := fnQuantizeQ8Rows(ctx, noopKwargs, matrix, object.NewInteger(int64(nCols)))
	raw, ok := result.(*object.String)
	if !ok {
		t.Fatalf("quantize_q8_rows returned %s", result.Type().String())
	}

	rawBytes := []byte(raw.Value)
	expectedLen := 2 * (nCols / 32) * 34
	if len(rawBytes) != expectedLen {
		t.Fatalf("quantize_q8_rows raw len = %d, want %d", len(rawBytes), expectedLen)
	}
}

func TestQuantizeQ8RowsErrors(t *testing.T) {
	assertError(t, fnQuantizeQ8Rows(ctx, noopKwargs), "2 arguments")
	assertError(t, fnQuantizeQ8Rows(ctx, noopKwargs, &object.String{Value: "x"}, object.NewInteger(32)), "LIST")
	assertError(t, fnQuantizeQ8Rows(ctx, noopKwargs, floatList(1.0), object.NewInteger(32)), "2D matrix")
	empty := &object.List{Elements: []object.Object{}}
	assertError(t, fnQuantizeQ8Rows(ctx, noopKwargs, empty, object.NewInteger(32)), "empty")
	assertError(t, fnQuantizeQ8Rows(ctx, noopKwargs, floatMatrix(make([]float64, 32)), object.NewInteger(31)), "divisible by 32")
}

func TestFloat64ToFloat16(t *testing.T) {
	tests := []struct {
		input    float64
		expected uint16
		name     string
	}{
		{0.0, 0x0000, "zero"},
		{1.0, 0x3C00, "one"},
		{-1.0, 0xBC00, "negative one"},
		{2.0, 0x4000, "two"},
		{0.5, 0x3800, "half"},
		{4.0, 0x4400, "four"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := float64ToFloat16(tt.input)
			if got != tt.expected {
				t.Errorf("float64ToFloat16(%f) = 0x%04X, want 0x%04X", tt.input, got, tt.expected)
			}
		})
	}
}

func TestFloat64ToFloat16Special(t *testing.T) {
	got := float64ToFloat16(math.NaN())
	if got != 0 {
		t.Errorf("float64ToFloat16(NaN) = 0x%04X, want 0", got)
	}

	got = float64ToFloat16(math.Inf(1))
	if got != 0 {
		t.Errorf("float64ToFloat16(+Inf) = 0x%04X, want 0", got)
	}

	got = float64ToFloat16(math.Inf(-1))
	if got != 0 {
		t.Errorf("float64ToFloat16(-Inf) = 0x%04X, want 0", got)
	}
}
