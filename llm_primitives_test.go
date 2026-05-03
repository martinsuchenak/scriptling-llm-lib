package scriptlingllmlib

import (
	"math"
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestRmsNorm(t *testing.T) {
	x := floatMatrix([]float64{0.5, -0.3, 0.8}, []float64{0.1, 0.2, -0.4})
	w := floatList(1.0, 1.0, 1.0)

	result := fnRmsNorm(ctx, noopKwargs, x, w)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 2 || len(mat[0]) != 3 {
		t.Fatalf("rms_norm shape = %dx%d, want 2x3", len(mat), len(mat[0]))
	}

	row := []float64{0.5, -0.3, 0.8}
	ss := (0.25 + 0.09 + 0.64) / 3.0
	inv := 1.0 / math.Sqrt(ss+1e-5)
	for j, v := range mat[0] {
		expected := row[j] * inv * 1.0
		if math.Abs(v-expected) > 1e-10 {
			t.Errorf("rms_norm[0][%d] = %f, want %f", j, v, expected)
		}
	}

	result = fnRmsNorm(ctx, noopKwargs, x, w, &object.Float{Value: 1e-6})
	mat2 := evalFloatMatrix(t, result)
	if len(mat2) != 2 {
		t.Error("rms_norm with eps failed")
	}

	assertError(t, fnRmsNorm(ctx, noopKwargs, floatMatrix([]float64{1.0, 2.0}), floatList(1.0)), "columns")
	assertError(t, fnRmsNorm(ctx, noopKwargs), "2 arguments")
	assertError(t, fnRmsNorm(ctx, noopKwargs, &object.String{Value: "x"}, floatList(1.0)), "LIST")
	assertError(t, fnRmsNorm(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnRmsNorm(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatList(1.0), &object.String{Value: "x"}), "INTEGER or FLOAT")
}

func TestRope(t *testing.T) {
	x := floatMatrix([]float64{1.0, 0.0, 0.0, 1.0})
	result := fnRope(ctx, noopKwargs, x)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 4 {
		t.Fatalf("rope shape = %dx%d, want 1x4", len(mat), len(mat[0]))
	}

	dk := 4.0
	for i := 0; i < 2; i++ {
		freq := 1.0 / math.Pow(10000.0, 2.0*float64(i)/dk)
		angle := freq * 0.0
		cosA := math.Cos(angle)
		sinA := math.Sin(angle)
		expectedEven := 1.0*cosA - 0.0*sinA
		if i == 0 {
			if math.Abs(mat[0][0]-expectedEven) > 1e-10 {
				t.Errorf("rope[0][0] = %f, want %f", mat[0][0], expectedEven)
			}
		}
	}

	result = fnRope(ctx, noopKwargs, x, object.NewInteger(5))
	mat = evalFloatMatrix(t, result)
	if len(mat) != 1 {
		t.Error("rope with start_pos failed")
	}

	oddDim := floatMatrix([]float64{1.0, 0.0, 1.0})
	assertError(t, fnRope(ctx, noopKwargs, oddDim), "even")
	assertError(t, fnRope(ctx, noopKwargs), "1 argument")
	assertError(t, fnRope(ctx, noopKwargs, &object.String{Value: "x"}), "LIST")
	assertError(t, fnRope(ctx, noopKwargs, floatMatrix([]float64{1.0, 0.0}), &object.String{Value: "x"}), "INTEGER")
	result = fnRope(ctx, noopKwargs, &object.List{Elements: []object.Object{}})
	assertEmptyListOrFloatArray(t, result)
}

func TestSiluGate(t *testing.T) {
	gate := floatMatrix([]float64{1.0, -1.0}, []float64{0.0, 2.0})
	up := floatMatrix([]float64{1.0, 1.0}, []float64{1.0, 1.0})

	result := fnSiluGate(ctx, noopKwargs, gate, up)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 2 || len(mat[0]) != 2 {
		t.Fatalf("silu_gate shape = %dx%d, want 2x2", len(mat), len(mat[0]))
	}

	sig1 := 1.0 / (1.0 + math.Exp(-1.0))
	expected00 := 1.0 * sig1 * 1.0
	if math.Abs(mat[0][0]-expected00) > 1e-10 {
		t.Errorf("silu_gate[0][0] = %f, want %f", mat[0][0], expected00)
	}

	sigNeg1 := 1.0 / (1.0 + math.Exp(1.0))
	expected01 := (-1.0) * sigNeg1 * 1.0
	if math.Abs(mat[0][1]-expected01) > 1e-10 {
		t.Errorf("silu_gate[0][1] = %f, want %f", mat[0][1], expected01)
	}

	if mat[1][0] != 0.0 {
		t.Errorf("silu_gate[1][0] = %f, want 0", mat[1][0])
	}

	assertError(t, fnSiluGate(ctx, noopKwargs), "2 arguments")
	assertError(t, fnSiluGate(ctx, noopKwargs, &object.String{Value: "x"}, floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnSiluGate(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnSiluGate(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}, []float64{1.0})), "rows")
	assertError(t, fnSiluGate(ctx, noopKwargs, floatMatrix([]float64{1.0, 2.0}), floatMatrix([]float64{1.0})), "columns")
	empty := &object.List{Elements: []object.Object{}}
	result = fnSiluGate(ctx, noopKwargs, empty, empty)
	assertEmptyListOrFloatArray(t, result)
}

func TestAttention(t *testing.T) {
	q := floatMatrix([]float64{1.0, 0.0})
	k := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})
	v := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})

	result := fnAttention(ctx, noopKwargs, q, k, v)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 2 {
		t.Fatalf("attention shape = %dx%d, want 1x2", len(mat), len(mat[0]))
	}

	if mat[0][0] < 0.5 {
		t.Errorf("attention[0][0] = %f, expected dominant weight on position 0", mat[0][0])
	}
	if mat[0][0]+mat[0][1] < 0.99 || mat[0][0]+mat[0][1] > 1.01 {
		t.Errorf("attention outputs should sum to ~1.0: got %f", mat[0][0]+mat[0][1])
	}

	result = fnAttention(ctx, noopKwargs, q, k, v, object.NewBoolean(false))
	mat = evalFloatMatrix(t, result)
	if len(mat) != 1 {
		t.Error("non-causal attention failed")
	}
}

func TestAttentionCausal(t *testing.T) {
	q := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})
	k := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})
	v := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})

	result := fnAttention(ctx, noopKwargs, q, k, v, object.NewBoolean(true))
	mat := evalFloatMatrix(t, result)

	if mat[0][0] < 0.99 {
		t.Errorf("causal attention row 0 should attend only to pos 0: got %f", mat[0][0])
	}
	if mat[1][1] < 0.49 {
		t.Errorf("causal attention row 1 should attend to both: got %f", mat[1][1])
	}
}

func TestAttentionErrors(t *testing.T) {
	assertError(t, fnAttention(ctx, noopKwargs), "3 arguments")
	assertError(t, fnAttention(ctx, noopKwargs, &object.String{Value: "x"}, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnAttention(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}, floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnAttention(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	empty := floatMatrix()
	assertError(t, fnAttention(ctx, noopKwargs, empty, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0})), "empty")
	assertError(t, fnAttention(ctx, noopKwargs, floatMatrix([]float64{1.0}), empty, floatMatrix([]float64{1.0})), "empty")
	assertError(t, fnAttention(ctx, noopKwargs, floatMatrix([]float64{1.0, 2.0}), floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0})), "inner dimension")
	assertError(t, fnAttention(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}, []float64{1.0})), "same number of rows")
}

func TestLinear(t *testing.T) {
	x := floatMatrix([]float64{1.0, 2.0})
	weight := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})

	result := fnLinear(ctx, noopKwargs, x, weight)
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 2 {
		t.Fatalf("linear shape = %dx%d, want 1x2", len(mat), len(mat[0]))
	}
	if mat[0][0] != 1.0 || mat[0][1] != 2.0 {
		t.Errorf("linear identity = %v, want [1, 2]", mat[0])
	}

	bias := floatList(10.0, 20.0)
	result = fnLinear(ctx, noopKwargs, x, weight, bias)
	mat = evalFloatMatrix(t, result)
	if mat[0][0] != 11.0 || mat[0][1] != 22.0 {
		t.Errorf("linear with bias = %v, want [11, 22]", mat[0])
	}
}

func TestLinearRow(t *testing.T) {
	x := floatMatrix([]float64{1.0, 2.0}, []float64{3.0, 4.0})
	weight := floatMatrix([]float64{1.0, 0.0}, []float64{0.0, 1.0})

	result := fnLinearRow(ctx, noopKwargs, x, weight)
	vals := evalFloatList(t, result)
	if vals[0] != 3.0 || vals[1] != 4.0 {
		t.Errorf("linear_row = %v, want [3, 4]", vals)
	}

	bias := floatList(1.0, 1.0)
	result = fnLinearRow(ctx, noopKwargs, x, weight, bias)
	vals = evalFloatList(t, result)
	if vals[0] != 4.0 || vals[1] != 5.0 {
		t.Errorf("linear_row with bias = %v, want [4, 5]", vals)
	}
}

func TestLinearErrors(t *testing.T) {
	assertError(t, fnLinear(ctx, noopKwargs), "2 arguments")
	assertError(t, fnLinear(ctx, noopKwargs, &object.String{Value: "x"}, floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnLinear(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnLinear(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnLinear(ctx, noopKwargs, floatMatrix(), floatMatrix([]float64{1.0})), "empty")
	assertError(t, fnLinear(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix()), "empty")
	assertError(t, fnLinear(ctx, noopKwargs, floatMatrix([]float64{1.0, 2.0}), floatMatrix([]float64{1.0})), "columns")
}

func TestLinearRowErrors(t *testing.T) {
	assertError(t, fnLinearRow(ctx, noopKwargs), "2 arguments")
	assertError(t, fnLinearRow(ctx, noopKwargs, &object.String{Value: "x"}, floatMatrix([]float64{1.0})), "LIST")
	assertError(t, fnLinearRow(ctx, noopKwargs, floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnLinearRow(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix([]float64{1.0}), &object.String{Value: "x"}), "LIST")
	assertError(t, fnLinearRow(ctx, noopKwargs, floatMatrix(), floatMatrix([]float64{1.0})), "empty")
	assertError(t, fnLinearRow(ctx, noopKwargs, floatMatrix([]float64{1.0}), floatMatrix()), "empty")
	assertError(t, fnLinearRow(ctx, noopKwargs, floatMatrix([]float64{1.0, 2.0}), floatMatrix([]float64{1.0})), "columns")
}

func TestTopK(t *testing.T) {
	result := fnTopK(ctx, noopKwargs, floatList(0.1, 0.5, 0.3, 0.9, 0.7), object.NewInteger(3))
	elems := evalList(t, result)
	if len(elems) != 3 {
		t.Fatalf("top_k len = %d, want 3", len(elems))
	}

	first := evalList(t, elems[0])
	if evalInt(t, first[0]) != 3 {
		t.Errorf("top_k[0].idx = %d, want 3 (value 0.9)", evalInt(t, first[0]))
	}
	if math.Abs(evalFloat(t, first[1])-0.9) > 1e-10 {
		t.Errorf("top_k[0].val = %f, want 0.9", evalFloat(t, first[1]))
	}

	assertError(t, fnTopK(ctx, noopKwargs), "2 arguments")
	assertError(t, fnTopK(ctx, noopKwargs, &object.String{Value: "x"}, object.NewInteger(1)), "LIST")
	assertError(t, fnTopK(ctx, noopKwargs, floatList(1.0), &object.String{Value: "x"}), "INTEGER")
	assertError(t, fnTopK(ctx, noopKwargs, floatList(1.0), object.NewInteger(0)), "positive")
}

func TestDequantizeQ8(t *testing.T) {
	data := intList(10, -5, 20, 15)
	scales := floatList(0.1, 0.2)

	result := fnDequantizeQ8(ctx, noopKwargs, data, scales, object.NewInteger(2))
	vals := evalFloatList(t, result)
	expected := []float64{1.0, -0.5, 4.0, 3.0}
	for i, v := range vals {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("dequantize_q8[%d] = %f, want %f", i, v, expected[i])
		}
	}

	assertError(t, fnDequantizeQ8(ctx, noopKwargs, intList(1), floatList(0.1), object.NewInteger(0)), "positive")
	assertError(t, fnDequantizeQ8(ctx, noopKwargs), "3 arguments")
	assertError(t, fnDequantizeQ8(ctx, noopKwargs, &object.String{Value: "x"}, floatList(0.1), object.NewInteger(2)), "LIST")
	assertError(t, fnDequantizeQ8(ctx, noopKwargs, intList(200), floatList(0.1), object.NewInteger(1)), "int8 range")
	result = fnDequantizeQ8(ctx, noopKwargs, intList(), floatList(0.1), object.NewInteger(2))
	assertEmptyListOrFloatArray(t, result)
}

func TestFloat16ToFloat64(t *testing.T) {
	tests := []struct {
		bits     uint16
		expected float64
		nan      bool
		inf      int
		name     string
	}{
		{0x0000, 0.0, false, 0, "positive zero"},
		{0x8000, 0.0, false, 0, "negative zero"},
		{0x3C00, 1.0, false, 0, "1.0"},
		{0xBC00, -1.0, false, 0, "-1.0"},
		{0x4000, 2.0, false, 0, "2.0"},
		{0x3800, 0.5, false, 0, "0.5"},
		{0x4400, 4.0, false, 0, "4.0"},
		{0xC400, -4.0, false, 0, "-4.0"},
		{0x0001, math.Ldexp(1.0/1024.0, -14), false, 0, "subnormal"},
		{0x7C00, 0, false, 1, "positive infinity"},
		{0xFC00, 0, false, -1, "negative infinity"},
		{0x7C01, 0, true, 0, "NaN"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := float16ToFloat64(tt.bits)
			if tt.nan {
				if !math.IsNaN(got) {
					t.Errorf("float16ToFloat64(0x%04X) = %f, want NaN", tt.bits, got)
				}
			} else if tt.inf != 0 {
				if !math.IsInf(got, tt.inf) {
					t.Errorf("float16ToFloat64(0x%04X) = %f, want Inf(%d)", tt.bits, got, tt.inf)
				}
			} else {
				if math.Abs(got-tt.expected) > 1e-10 {
					t.Errorf("float16ToFloat64(0x%04X) = %f, want %f", tt.bits, got, tt.expected)
				}
			}
		})
	}
}

func TestDequantizeQ80(t *testing.T) {
	values := make([]int8, 32)
	for i := range values {
		values[i] = int8(i + 1)
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, values))

	result := fnDequantizeQ8_0(ctx, noopKwargs, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 32 {
		t.Fatalf("dequantize_q8_0 len = %d, want 32", len(vals))
	}
	for i, v := range vals {
		if math.Abs(v-float64(i+1)) > 1e-10 {
			t.Errorf("dequantize_q8_0[%d] = %f, want %d", i, v, i+1)
		}
	}

	values2 := make([]int8, 32)
	for i := range values2 {
		values2[i] = -int8(i + 1)
	}
	raw2 := makeQ80Raw(makeQ80Group(0x3C00, values), makeQ80Group(0x4000, values2))
	result2 := fnDequantizeQ8_0(ctx, noopKwargs, raw2, object.NewInteger(2))
	vals2 := evalFloatList(t, result2)
	if len(vals2) != 64 {
		t.Fatalf("dequantize_q8_0 2 groups len = %d, want 64", len(vals2))
	}
	for i := 0; i < 32; i++ {
		if math.Abs(vals2[32+i]-float64(-(i+1))*2.0) > 1e-10 {
			t.Errorf("dequantize_q8_0 group1[%d] = %f, want %f", i, vals2[32+i], float64(-(i+1))*2.0)
		}
	}
}

func TestDequantizeQ80Errors(t *testing.T) {
	assertError(t, fnDequantizeQ8_0(ctx, noopKwargs), "2 arguments")
	assertError(t, fnDequantizeQ8_0(ctx, noopKwargs, floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnDequantizeQ8_0(ctx, noopKwargs, &object.String{Value: "x"}, &object.String{Value: "x"}), "INTEGER")
	assertError(t, fnDequantizeQ8_0(ctx, noopKwargs, &object.String{Value: "x"}, object.NewInteger(0)), "positive")
	assertError(t, fnDequantizeQ8_0(ctx, noopKwargs, &object.String{Value: "short"}, object.NewInteger(1)), "too short")
}

func TestLinearQ8(t *testing.T) {
	ones := make([]float64, 32)
	for i := range ones {
		ones[i] = 1.0
	}
	x := floatMatrix(ones)
	qValues := make([]int8, 32)
	for i := range qValues {
		qValues[i] = 1
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))

	result := fnLinearQ8(ctx, noopKwargs, x, raw, object.NewInteger(1))
	mat := evalFloatMatrix(t, result)
	if len(mat) != 1 || len(mat[0]) != 1 {
		t.Fatalf("linear_q8 shape = %dx%d, want 1x1", len(mat), len(mat[0]))
	}
	if math.Abs(mat[0][0]-32.0) > 1e-10 {
		t.Errorf("linear_q8 = %f, want 32.0", mat[0][0])
	}
}

func TestLinearQ8Errors(t *testing.T) {
	assertError(t, fnLinearQ8(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearQ8(ctx, noopKwargs, &object.String{Value: "x"}, &object.String{Value: "x"}, object.NewInteger(1)), "LIST or FLOAT_ARRAY")
	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), &object.String{Value: "x"}, &object.String{Value: "x"}), "INTEGER")
	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), &object.String{Value: "x"}, object.NewInteger(0)), "positive")
	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix(), makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "empty")
	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), &object.String{Value: ""}, object.NewInteger(1)), "empty")
	assertError(t, fnLinearQ8(ctx, noopKwargs, floatMatrix([]float64{1.0}), makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "columns")
}

func TestLinearRowQ8(t *testing.T) {
	ones := make([]float64, 32)
	for i := range ones {
		ones[i] = 1.0
	}
	twos := make([]float64, 32)
	for i := range twos {
		twos[i] = 2.0
	}
	x := floatMatrix(ones, twos)
	qValues := make([]int8, 32)
	for i := range qValues {
		qValues[i] = 1
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))

	result := fnLinearRowQ8(ctx, noopKwargs, x, raw, object.NewInteger(1))
	vals := evalFloatList(t, result)
	if len(vals) != 1 {
		t.Fatalf("linear_row_q8 len = %d, want 1", len(vals))
	}
	if math.Abs(vals[0]-64.0) > 1e-10 {
		t.Errorf("linear_row_q8 = %f, want 64.0", vals[0])
	}
}

func TestLinearRowQ8Errors(t *testing.T) {
	assertError(t, fnLinearRowQ8(ctx, noopKwargs), "3 arguments")
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, &object.String{Value: "x"}, &object.String{Value: "x"}, object.NewInteger(1)), "LIST or FLOAT_ARRAY")
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), floatList(1.0), object.NewInteger(1)), "STRING")
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), &object.String{Value: "x"}, &object.String{Value: "x"}), "INTEGER")
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, floatMatrix(make([]float64, 32)), &object.String{Value: "x"}, object.NewInteger(0)), "positive")
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, floatMatrix(), makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "empty")
	assertError(t, fnLinearRowQ8(ctx, noopKwargs, floatMatrix([]float64{1.0}), makeQ80Raw(make([]byte, 34)), object.NewInteger(1)), "columns")
}
