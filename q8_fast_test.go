package scriptlingllmlib

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestQ8DotGroupXEquivalence(t *testing.T) {
	group := make([]byte, 34)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	for i := 2; i < 34; i++ {
		group[i] = byte(i)
	}

	xData := make([]float64, 32)
	for i := range xData {
		xData[i] = float64(i) * 0.1
	}

	resultX := q8DotGroupX(group, 0, xData, 0)
	resultSlow := q8DotGroup(group, 0, xData, 0)

	if math.IsNaN(resultX) || math.IsInf(resultX, 0) {
		t.Fatalf("q8DotGroupX = %f, expected finite", resultX)
	}
	if math.Abs(resultX-resultSlow) > 1e-10 {
		t.Errorf("q8DotGroupX = %f, q8DotGroup = %f", resultX, resultSlow)
	}
}

func TestQ4DotGroupXEquivalence(t *testing.T) {
	group := make([]byte, 18)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	for i := 2; i < 18; i++ {
		group[i] = byte(i * 17)
	}

	xData := make([]float64, 32)
	for i := range xData {
		xData[i] = float64(i) * 0.1
	}

	resultX := q4DotGroupX(group, 0, xData, 0)
	resultSlow := q4DotGroup(group, 0, xData, 0)

	if math.IsNaN(resultX) || math.IsInf(resultX, 0) {
		t.Fatalf("q4DotGroupX = %f, expected finite", resultX)
	}
	if math.Abs(resultX-resultSlow) > 1e-10 {
		t.Errorf("q4DotGroupX = %f, q4DotGroup = %f", resultX, resultSlow)
	}
}

func TestQ4DotGroupX(t *testing.T) {
	group := make([]byte, 18)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00) // scale = 1.0
	for i := 2; i < 18; i++ {
		group[i] = 0x88 // all nibbles = 8
	}

	xData := make([]float64, 32)
	for i := range xData {
		xData[i] = 1.0
	}

	result := q4DotGroupX(group, 0, xData, 0)
	if math.IsNaN(result) || math.IsInf(result, 0) {
		t.Fatalf("q4DotGroupX = %f, expected finite", result)
	}

	// nibble=8, int8(8-8)=0, so result should be 0
	if math.Abs(result) > 1e-10 {
		t.Errorf("q4DotGroupX with nibble 8 = %f, want ~0", result)
	}
}

func TestQ4DotGroupXNonZero(t *testing.T) {
	group := make([]byte, 18)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00) // scale = 1.0
	for i := 2; i < 18; i++ {
		group[i] = 0x99 // all nibbles = 9
	}

	xData := make([]float64, 32)
	for i := range xData {
		xData[i] = 1.0
	}

	result := q4DotGroupX(group, 0, xData, 0)
	if math.IsNaN(result) || math.IsInf(result, 0) {
		t.Fatalf("q4DotGroupX = %f, expected finite", result)
	}
	// nibble=9, int8(9-8)=1, scale=1.0, 32 values * 1.0 * 1.0 = 32.0
	if math.Abs(result-32.0) > 1e-10 {
		t.Errorf("q4DotGroupX with nibble 9 = %f, want 32.0", result)
	}
}

func TestQ41DotGroupX(t *testing.T) {
	group := make([]byte, 20)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00) // d = 1.0
	binary.LittleEndian.PutUint16(group[2:4], 0x0000) // m = 0.0
	for i := 4; i < 20; i++ {
		group[i] = 0x55
	}

	xData := make([]float64, 32)
	for i := range xData {
		xData[i] = 1.0
	}

	result := q4_1DotGroupX(group, 0, xData, 0)
	if math.IsNaN(result) || math.IsInf(result, 0) {
		t.Fatalf("q4_1DotGroupX = %f, expected finite", result)
	}
}

func TestQ41DotGroupXWithMin(t *testing.T) {
	group := make([]byte, 20)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00) // d = 1.0
	binary.LittleEndian.PutUint16(group[2:4], 0x3C00) // m = 1.0
	for i := 4; i < 20; i++ {
		group[i] = 0x00 // all nibbles = 0
	}

	xData := make([]float64, 32)
	for i := range xData {
		xData[i] = 2.0
	}

	result := q4_1DotGroupX(group, 0, xData, 0)
	if math.IsNaN(result) || math.IsInf(result, 0) {
		t.Fatalf("q4_1DotGroupX = %f, expected finite", result)
	}
	// d=1, qSum=0 (all nibbles 0), m=1, xSum=64.0 (32*2.0)
	// result = 1*0 + 1*64 = 64.0
	if math.Abs(result-64.0) > 1e-10 {
		t.Errorf("q4_1DotGroupX = %f, want 64.0", result)
	}
}

func TestLinearQ8Fast(t *testing.T) {
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

	result := fnLinearQ8Fast(ctx, noopKwargs, x, raw, object.NewInteger(1))
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 1 {
		t.Fatalf("linear_q8 shape = %dx%d, want 1x1", len(mat), len(mat[0]))
	}
	if math.Abs(mat[0][0]-32.0) > 1e-10 {
		t.Errorf("linear_q8 = %f, want 32.0", mat[0][0])
	}
}

func TestLinearRowQ8Fast(t *testing.T) {
	row0 := make([]float64, 32)
	row1 := make([]float64, 32)
	for i := range row1 {
		row1[i] = 2.0
	}
	x := object.NewFloatArray2D(append(row0, row1...), 2, 32)

	qValues := make([]int8, 32)
	for i := range qValues {
		qValues[i] = 1
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))

	result := fnLinearRowQ8Fast(ctx, noopKwargs, x, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 1 {
		t.Fatalf("linear_row_q8 len = %d, want 1", len(vals))
	}
	if math.Abs(vals[0]-64.0) > 1e-10 {
		t.Errorf("linear_row_q8 = %f, want 64.0", vals[0])
	}
}

func TestLinearQ4Fast(t *testing.T) {
	ones := make([]float64, 32)
	for i := range ones {
		ones[i] = 1.0
	}
	x := object.NewFloatArray2D(ones, 1, 32)

	group := make([]byte, 18)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00) // scale = 1.0
	for i := 2; i < 18; i++ {
		group[i] = 0x99 // all nibbles = 9
	}
	raw := &object.String{Value: string(group)}

	result := fnLinearQ4Fast(ctx, noopKwargs, x, raw, object.NewInteger(1))
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 1 {
		t.Fatalf("linear_q4 shape = %dx%d, want 1x1", len(mat), len(mat[0]))
	}
	// nibble=9, int8(9-8)=1, scale=1.0, 32 values * 1.0 = 32.0
	if math.Abs(mat[0][0]-32.0) > 1e-6 {
		t.Errorf("linear_q4 = %f, want ~32.0", mat[0][0])
	}
}

func TestLinearRowQ4Fast(t *testing.T) {
	row0 := make([]float64, 32)
	row1 := make([]float64, 32)
	for i := range row1 {
		row1[i] = 1.0
	}
	x := object.NewFloatArray2D(append(row0, row1...), 2, 32)

	group := make([]byte, 18)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	for i := 2; i < 18; i++ {
		group[i] = 0x99
	}
	raw := &object.String{Value: string(group)}

	result := fnLinearRowQ4Fast(ctx, noopKwargs, x, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 1 {
		t.Fatalf("linear_row_q4 len = %d, want 1", len(vals))
	}
}

func TestLinearQ8FastErrors(t *testing.T) {
	assertError(t, fnLinearQ8Fast(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearQ8Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearQ8Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, floatList(1.0)), "INTEGER")
	assertError(t, fnLinearQ8Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, object.NewInteger(0)), "positive")
}

func TestLinearRowQ8FastErrors(t *testing.T) {
	assertError(t, fnLinearRowQ8Fast(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearRowQ8Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearRowQ8Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, floatList(1.0)), "INTEGER")
}

func TestLinearQ4FastErrors(t *testing.T) {
	assertError(t, fnLinearQ4Fast(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearQ4Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearQ4Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, floatList(1.0)), "INTEGER")
	assertError(t, fnLinearQ4Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, object.NewInteger(0)), "positive")
}

func TestLinearRowQ4FastErrors(t *testing.T) {
	assertError(t, fnLinearRowQ4Fast(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearRowQ4Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearRowQ4Fast(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, floatList(1.0)), "INTEGER")
}

func TestLinearQ8FastDimensionMismatch(t *testing.T) {
	qValues := make([]int8, 32)
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))
	x := object.NewFloatArray2D([]float64{1.0, 1.0}, 1, 2)
	assertError(t, fnLinearQ8Fast(ctx, noopKwargs, x, raw, object.NewInteger(1)), "columns")
}

func TestLinearQ4FastDimensionMismatch(t *testing.T) {
	group := make([]byte, 18)
	raw := &object.String{Value: string(group)}
	x := object.NewFloatArray2D([]float64{1.0, 1.0}, 1, 2)
	assertError(t, fnLinearQ4Fast(ctx, noopKwargs, x, raw, object.NewInteger(1)), "columns")
}

func TestLinearQ8FastEmptyInput(t *testing.T) {
	qValues := make([]int8, 32)
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))
	emptyMat := object.NewFloatArray2D([]float64{}, 0, 0)
	assertError(t, fnLinearQ8Fast(ctx, noopKwargs, emptyMat, raw, object.NewInteger(1)), "empty")
}

func TestLinearQ4FastEmptyInput(t *testing.T) {
	group := make([]byte, 18)
	raw := &object.String{Value: string(group)}
	emptyMat := object.NewFloatArray2D([]float64{}, 0, 0)
	assertError(t, fnLinearQ4Fast(ctx, noopKwargs, emptyMat, raw, object.NewInteger(1)), "empty")
}

func TestReadF16(t *testing.T) {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint16(b[0:2], 0x3C00) // 1.0
	binary.LittleEndian.PutUint16(b[2:4], 0x4000) // 2.0

	v1 := readF16(b, 0)
	v2 := readF16(b, 2)
	if math.Abs(v1-1.0) > 1e-10 {
		t.Errorf("readF16(0) = %f, want 1.0", v1)
	}
	if math.Abs(v2-2.0) > 1e-10 {
		t.Errorf("readF16(2) = %f, want 2.0", v2)
	}
}
