package scriptlingllmlib

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

const float32Tol = 1e-5

func approxEqualF32(a, b float32) bool {
	return math.Abs(float64(a-b)) < float64(float32Tol)
}

func TestRmsNormFlatF32(t *testing.T) {
	xData := []float32{0.5, -0.3, 0.8}
	w := []float32{1.0, 1.0, 1.0}
	out := make([]float32, 3)

	rmsNormFlatF32(xData, w, 1e-5, 1, 3, out)

	ss := (0.25 + 0.09 + 0.64) / 3.0
	inv := float32(1.0 / math.Sqrt(ss+1e-5))
	for j := 0; j < 3; j++ {
		expected := xData[j] * inv * w[j]
		if !approxEqualF32(out[j], expected) {
			t.Errorf("rmsNormFlatF32[%d] = %f, want %f", j, out[j], expected)
		}
	}
}

func TestRmsNormFlatF32MultiRow(t *testing.T) {
	xData := []float32{3.0, 4.0, 6.0, 8.0}
	w := []float32{1.0, 1.0}
	out := make([]float32, 4)

	rmsNormFlatF32(xData, w, 1e-5, 2, 2, out)

	for r := 0; r < 2; r++ {
		off := r * 2
		ss := float64(xData[off]*xData[off]+xData[off+1]*xData[off+1]) / 2.0
		inv := float32(1.0 / math.Sqrt(ss+1e-5))
		for j := 0; j < 2; j++ {
			expected := xData[off+j] * inv * w[j]
			if !approxEqualF32(out[off+j], expected) {
				t.Errorf("rmsNormFlatF32 row%d[%d] = %f, want %f", r, j, out[off+j], expected)
			}
		}
	}
}

func TestApplyRopeInPlaceF32(t *testing.T) {
	data := []float32{1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5}
	dK := 4
	halfDim := 2
	freqs := make([]float32, halfDim)
	freqs[0] = float32(1.0 / math.Pow(10000.0, 0.0/float64(dK)))
	freqs[1] = float32(1.0 / math.Pow(10000.0, 2.0/float64(dK)))

	origData := make([]float32, len(data))
	copy(origData, data)

	applyRopeInPlaceF32(data, 2, dK, 5, freqs, halfDim)

	for r := 0; r < 2; r++ {
		pos := 5 + r
		off := r * dK
		for i := 0; i < halfDim; i++ {
			angle := float64(freqs[i]) * float64(pos)
			cosA := float32(math.Cos(angle))
			sinA := float32(math.Sin(angle))

			origOff := off + 2*i
			x0 := data[origOff]
			x1 := data[origOff+1]

			expected0 := origData[origOff]*cosA - origData[origOff+1]*sinA
			expected1 := origData[origOff]*sinA + origData[origOff+1]*cosA
			if !approxEqualF32(x0, expected0) {
				t.Errorf("data[%d] = %f, want %f", origOff, x0, expected0)
			}
			if !approxEqualF32(x1, expected1) {
				t.Errorf("data[%d] = %f, want %f", origOff+1, x1, expected1)
			}
		}
	}
}

func TestFusedAttentionHeadF32(t *testing.T) {
	qData := []float32{1.0, 0.0}
	kData := []float32{1.0, 0.0, 0.0, 1.0}
	vData := []float32{1.0, 0.0, 0.0, 1.0}

	out := make([]float32, 2)
	fusedAttentionHeadF32(qData, kData, vData, 1, 2, 2, false, 0, out, 0)

	total := out[0] + out[1]
	if total < 0.99 || total > 1.01 {
		t.Errorf("attention output sum = %f, want ~1.0", total)
	}
	if out[0] < 0.5 {
		t.Errorf("attention[0] = %f, expected dominant", out[0])
	}
}

func TestFusedAttentionHeadF32Causal(t *testing.T) {
	qData := []float32{1.0, 0.0, 0.0, 1.0}
	kData := []float32{1.0, 0.0, 0.0, 1.0}
	vData := []float32{1.0, 0.0, 0.0, 1.0}

	out := make([]float32, 4)
	fusedAttentionHeadF32(qData, kData, vData, 2, 2, 2, true, 0, out, 0)

	if out[0] < 0.99 {
		t.Errorf("causal row0 should attend only to pos0: got %f", out[0])
	}
}

func TestSplitHeadsDataF32(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	heads := splitHeadsDataF32(data, 2, 3, 3)

	if len(heads) != 3 {
		t.Fatalf("splitHeadsDataF32 count = %d, want 3", len(heads))
	}

	dK := 1
	for h := 0; h < 3; h++ {
		if len(heads[h]) != 2*dK {
			t.Fatalf("head[%d] len = %d, want %d", h, len(heads[h]), 2*dK)
		}
		for r := 0; r < 2; r++ {
			expected := data[r*3+h*dK]
			if heads[h][r*dK] != expected {
				t.Errorf("head[%d][%d] = %f, want %f", h, r*dK, heads[h][r*dK], expected)
			}
		}
	}
}

func TestMergeHeadsDataF32(t *testing.T) {
	dK := 2
	seqLen := 2
	nHeads := 2
	data := make([]float32, nHeads*seqLen*dK)
	for i := range data {
		data[i] = float32(i)
	}

	result := mergeHeadsDataF32(data, seqLen, nHeads, dK)
	if len(result) != seqLen*nHeads*dK {
		t.Fatalf("mergeHeadsDataF32 len = %d, want %d", len(result), seqLen*nHeads*dK)
	}

	for s := 0; s < seqLen; s++ {
		for h := 0; h < nHeads; h++ {
			for d := 0; d < dK; d++ {
				srcIdx := h*seqLen*dK + s*dK + d
				dstIdx := s*nHeads*dK + h*dK + d
				if result[dstIdx] != data[srcIdx] {
					t.Errorf("mergeHeadsDataF32[%d] = %f, want %f (from data[%d])", dstIdx, result[dstIdx], data[srcIdx], srcIdx)
				}
			}
		}
	}
}

func TestRepeatKVDataF32(t *testing.T) {
	heads := [][]float32{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	expanded := repeatKVDataF32(heads, 3)
	if len(expanded) != 6 {
		t.Fatalf("repeatKVDataF32 count = %d, want 6", len(expanded))
	}

	for i := 0; i < 3; i++ {
		if expanded[i][0] != 1.0 || expanded[i][1] != 2.0 {
			t.Errorf("expanded[%d] = %v, want [1, 2]", i, expanded[i])
		}
	}
	for i := 3; i < 6; i++ {
		if expanded[i][0] != 3.0 || expanded[i][1] != 4.0 {
			t.Errorf("expanded[%d] = %v, want [3, 4]", i, expanded[i])
		}
	}
}

func TestQ8DotGroupXF32(t *testing.T) {
	group := make([]byte, 34)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	for i := 2; i < 34; i++ {
		group[i] = byte(i)
	}

	xData := make([]float32, 32)
	for i := range xData {
		xData[i] = float32(i) * 0.1
	}

	resultF32 := q8DotGroupXF32(group, 0, xData, 0)
	resultF64 := q8DotGroupX(group, 0, f32ToF64(xData), 0)

	if math.IsNaN(float64(resultF32)) || math.IsInf(float64(resultF32), 0) {
		t.Fatalf("q8DotGroupXF32 = %f, expected finite", resultF32)
	}
	if math.Abs(float64(resultF32)-resultF64) > 0.01 {
		t.Errorf("q8DotGroupXF32 = %f, q8DotGroupX (f64) = %f", resultF32, resultF64)
	}
}

func TestQ4DotGroupXF32(t *testing.T) {
	group := make([]byte, 18)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	for i := 2; i < 18; i++ {
		group[i] = byte(i * 17)
	}

	xData := make([]float32, 32)
	for i := range xData {
		xData[i] = float32(i) * 0.1
	}

	resultF32 := q4DotGroupXF32(group, 0, xData, 0)
	resultF64 := q4DotGroupX(group, 0, f32ToF64(xData), 0)

	if math.IsNaN(float64(resultF32)) || math.IsInf(float64(resultF32), 0) {
		t.Fatalf("q4DotGroupXF32 = %f, expected finite", resultF32)
	}
	if math.Abs(float64(resultF32)-resultF64) > 0.01 {
		t.Errorf("q4DotGroupXF32 = %f, q4DotGroupX (f64) = %f", resultF32, resultF64)
	}
}

func TestQ4DotGroupXF32KnownValue(t *testing.T) {
	group := make([]byte, 18)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	for i := 2; i < 18; i++ {
		group[i] = 0x99
	}

	xData := make([]float32, 32)
	for i := range xData {
		xData[i] = 1.0
	}

	result := q4DotGroupXF32(group, 0, xData, 0)
	if math.Abs(float64(result)-32.0) > 0.01 {
		t.Errorf("q4DotGroupXF32 with nibble 9 = %f, want ~32.0", result)
	}
}

func TestQ41DotGroupXF32(t *testing.T) {
	group := make([]byte, 20)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	binary.LittleEndian.PutUint16(group[2:4], 0x0000)
	for i := 4; i < 20; i++ {
		group[i] = 0x55
	}

	xData := make([]float32, 32)
	for i := range xData {
		xData[i] = 1.0
	}

	resultF32 := q4_1DotGroupXF32(group, 0, xData, 0)
	resultF64 := q4_1DotGroupX(group, 0, f32ToF64(xData), 0)

	if math.IsNaN(float64(resultF32)) || math.IsInf(float64(resultF32), 0) {
		t.Fatalf("q4_1DotGroupXF32 = %f, expected finite", resultF32)
	}
	if math.Abs(float64(resultF32)-resultF64) > 0.01 {
		t.Errorf("q4_1DotGroupXF32 = %f, q4_1DotGroupX (f64) = %f", resultF32, resultF64)
	}
}

func TestQ5DotGroupXF32(t *testing.T) {
	group := make([]byte, 22)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	binary.LittleEndian.PutUint32(group[2:6], 0)
	for i := 6; i < 22; i++ {
		group[i] = byte(i)
	}

	xData := make([]float32, 32)
	for i := range xData {
		xData[i] = float32(i) * 0.1
	}

	resultF32 := q5DotGroupXF32(group, 0, xData, 0)
	resultF64 := q5DotGroupX(group, 0, f32ToF64(xData), 0)

	if math.IsNaN(float64(resultF32)) || math.IsInf(float64(resultF32), 0) {
		t.Fatalf("q5DotGroupXF32 = %f, expected finite", resultF32)
	}
	if math.Abs(float64(resultF32)-resultF64) > 0.01 {
		t.Errorf("q5DotGroupXF32 = %f, q5DotGroupX (f64) = %f", resultF32, resultF64)
	}
}

func TestModelMatmulQuantF32Q8(t *testing.T) {
	qValues := make([]int8, 32)
	for i := range qValues {
		qValues[i] = 1
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))

	w := &QuantWeight{QType: "q8", Raw: []byte(raw.StringValue()), Groups: 1, Rows: 1, Cols: 32}
	xData := make([]float32, 32)
	for i := range xData {
		xData[i] = 1.0
	}

	result, rows, cols := modelMatmulQuantF32(w, xData, 1, 32)
	if rows != 1 || cols != 1 {
		t.Fatalf("shape = %dx%d, want 1x1", rows, cols)
	}
	if math.Abs(float64(result[0])-32.0) > 0.01 {
		t.Errorf("matmul q8 f32 = %f, want ~32.0", result[0])
	}
}

func TestModelMatmulRowQuantF32Q8(t *testing.T) {
	qValues := make([]int8, 32)
	for i := range qValues {
		qValues[i] = 1
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))

	w := &QuantWeight{QType: "q8", Raw: []byte(raw.StringValue()), Groups: 1, Rows: 1, Cols: 32}
	normed := make([]float32, 32)
	for i := range normed {
		normed[i] = 1.0
	}

	result := modelMatmulRowQuantF32(w, normed)
	if len(result) != 1 {
		t.Fatalf("len = %d, want 1", len(result))
	}
	if math.Abs(float64(result[0])-32.0) > 0.01 {
		t.Errorf("row quant q8 f32 = %f, want ~32.0", result[0])
	}
}

func TestModelMatmulFloatF32(t *testing.T) {
	wFlat := []float32{1.0, 2.0, 3.0, 4.0}
	xData := []float32{1.0, 0.0, 0.0, 1.0}

	result, rows, cols := modelMatmulFloatF32(wFlat, xData, 2, 2)
	if rows != 2 || cols != 2 {
		t.Fatalf("shape = %dx%d, want 2x2", rows, cols)
	}
	if !approxEqualF32(result[0], 1.0) {
		t.Errorf("result[0] = %f, want 1.0", result[0])
	}
	if !approxEqualF32(result[1], 3.0) {
		t.Errorf("result[1] = %f, want 3.0", result[1])
	}
	if !approxEqualF32(result[2], 2.0) {
		t.Errorf("result[2] = %f, want 2.0", result[2])
	}
	if !approxEqualF32(result[3], 4.0) {
		t.Errorf("result[3] = %f, want 4.0", result[3])
	}
}

func TestModelMatmulRowFloatF32(t *testing.T) {
	wFlat := []float32{1.0, 2.0, 3.0, 4.0}
	normed := []float32{1.0, 1.0}

	result := modelMatmulRowFloatF32(wFlat, normed)
	if len(result) != 2 {
		t.Fatalf("len = %d, want 2", len(result))
	}
	if !approxEqualF32(result[0], 3.0) {
		t.Errorf("result[0] = %f, want 3.0", result[0])
	}
	if !approxEqualF32(result[1], 7.0) {
		t.Errorf("result[1] = %f, want 7.0", result[1])
	}
}

func TestF64ToF32(t *testing.T) {
	in := []float64{1.0, 2.5, -3.7}
	out := f64ToF32(in)
	for i, v := range out {
		if !approxEqualF32(v, float32(in[i])) {
			t.Errorf("f64ToF32[%d] = %f, want %f", i, v, float32(in[i]))
		}
	}
}

func TestF32ToF64(t *testing.T) {
	in := []float32{1.0, 2.5, -3.7}
	out := f32ToF64(in)
	for i, v := range out {
		if math.Abs(v-float64(in[i])) > 1e-10 {
			t.Errorf("f32ToF64[%d] = %f, want %f", i, v, float64(in[i]))
		}
	}
}

func TestFlattenF64ToF32(t *testing.T) {
	m := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	flat := flattenF64ToF32(m)

	if len(flat) != 4 {
		t.Fatalf("len = %d, want 4", len(flat))
	}
	expected := []float32{1.0, 2.0, 3.0, 4.0}
	for i, v := range flat {
		if !approxEqualF32(v, expected[i]) {
			t.Errorf("flat[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

func TestCopyKVCachesF32(t *testing.T) {
	original := []KVCacheF32{
		{
			K: [][]float32{{1.0, 2.0, 3.0}},
			V: [][]float32{{4.0, 5.0, 6.0}},
		},
	}

	copied := copyKVCachesF32(original)
	if len(copied) != 1 {
		t.Fatalf("copyKVCachesF32 len = %d, want 1", len(copied))
	}

	copied[0].K[0][0] = 999.0
	if original[0].K[0][0] == 999.0 {
		t.Error("copyKVCachesF32 did not deep copy - modifying copy affected original")
	}
}

func TestBufPool(t *testing.T) {
	b1 := bp.get(100)
	if len(b1) != 100 {
		t.Fatalf("get(100) len = %d, want 100", len(b1))
	}
	for i := range b1 {
		b1[i] = float32(i)
	}
	bp.put(b1)

	b2 := bp.get(50)
	if len(b2) != 50 {
		t.Fatalf("get(50) len = %d, want 50", len(b2))
	}
	for _, v := range b2 {
		if v != 0 {
			t.Error("reused buffer should be zeroed")
		}
	}
}

func TestBuildInferenceModelF32(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model build test in short mode")
	}
	if raceEnabled {
		t.Skip("skipping model build test under race detector")
	}

	paths := []string{
		"models/SmolLM2-135M-Instruct-Q8_0.gguf",
		"models/SmolLM2-1.7B-Instruct-Q8_0.gguf",
	}
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			if _, err := os.Stat(path); os.IsNotExist(err) {
				t.Skip("model file not found")
			}

			gguf, err := LoadGGUF(path)
			if err != nil {
				t.Fatalf("LoadGGUF failed: %v", err)
			}
			gguf.Metadata["_path"] = path

			model, err := buildInferenceModelF32(gguf, path)
			if err != nil {
				t.Fatalf("buildInferenceModelF32 failed: %v", err)
			}

			cfg := model.Config
			t.Logf("arch=%s vocab=%d d_model=%d d_ff=%d n_layers=%d n_heads=%d n_kv_heads=%d d_k=%d output_type=%T",
				model.Arch, cfg.VocabSize, cfg.DModel, cfg.DFF, cfg.NLayers,
				model.nHeads, model.nKVHeads, model.dK, model.OutputW)
		})
	}
}

func TestInferenceModelF32Forward(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping forward test in short mode")
	}
	if raceEnabled {
		t.Skip("skipping forward test under race detector")
	}

	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("model file not found")
	}

	gguf, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF failed: %v", err)
	}
	gguf.Metadata["_path"] = path

	model, err := buildInferenceModelF32(gguf, path)
	if err != nil {
		t.Fatalf("buildInferenceModelF32 failed: %v", err)
	}

	model.initKVCaches()

	ids := model.Tokenizer.Encode("Hello")
	if len(ids) == 0 {
		t.Fatal("tokenizer returned empty ids")
	}

	logits := model.Forward(ids, 0)
	if len(logits) != model.Config.VocabSize {
		t.Fatalf("logits length = %d, want VocabSize %d", len(logits), model.Config.VocabSize)
	}

	hasNaN := false
	hasInf := false
	for _, v := range logits {
		if math.IsNaN(float64(v)) {
			hasNaN = true
		}
		if math.IsInf(float64(v), 0) {
			hasInf = true
		}
	}
	if hasNaN {
		t.Error("logits contain NaN")
	}
	if hasInf {
		t.Error("logits contain Inf")
	}

	maxIdx := 0
	for i, v := range logits {
		if v > logits[maxIdx] {
			maxIdx = i
		}
	}
	t.Logf("Top token after 'Hello' (f32): %d (%q)", maxIdx, model.Tokenizer.IDToToken[maxIdx])
}

func TestInferenceModelF32Generate(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping generate test in short mode")
	}
	if raceEnabled {
		t.Skip("skipping generate test under race detector")
	}

	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("model file not found")
	}

	gguf, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF failed: %v", err)
	}
	gguf.Metadata["_path"] = path

	model, err := buildInferenceModelF32(gguf, path)
	if err != nil {
		t.Fatalf("buildInferenceModelF32 failed: %v", err)
	}

	model.initKVCaches()

	output, nGen, nPrompt, _ := model.Generate("Hello", 3, "greedy", 0, 0, 0, 0, 0, "", "", 0)

	if nPrompt == 0 {
		t.Error("prompt tokens should be > 0")
	}
	if nGen == 0 {
		t.Error("generated tokens should be > 0")
	}
	if output == "" {
		t.Error("output should not be empty")
	}

	t.Logf("Generated f32 (%d prompt + %d gen): %q", nPrompt, nGen, output)
}

func TestModelMatmulQuantF32Q4(t *testing.T) {
	group := make([]byte, 18)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	for i := 2; i < 18; i++ {
		group[i] = 0x99
	}

	w := &QuantWeight{QType: "q4", Raw: group, Groups: 1, Rows: 1, Cols: 32}
	xData := make([]float32, 32)
	for i := range xData {
		xData[i] = 1.0
	}

	result, rows, cols := modelMatmulQuantF32(w, xData, 1, 32)
	if rows != 1 || cols != 1 {
		t.Fatalf("shape = %dx%d, want 1x1", rows, cols)
	}
	if math.Abs(float64(result[0])-32.0) > 0.01 {
		t.Errorf("matmul q4 f32 = %f, want ~32.0", result[0])
	}
}

func TestModelMatmulQuantF32Q41(t *testing.T) {
	group := make([]byte, 20)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	binary.LittleEndian.PutUint16(group[2:4], 0x0000)
	for i := 4; i < 20; i++ {
		group[i] = 0x55
	}

	w := &QuantWeight{QType: "q4_1", Raw: group, Groups: 1, Rows: 1, Cols: 32}
	xData := make([]float32, 32)
	for i := range xData {
		xData[i] = 1.0
	}

	result, rows, cols := modelMatmulQuantF32(w, xData, 1, 32)
	if rows != 1 || cols != 1 {
		t.Fatalf("shape = %dx%d, want 1x1", rows, cols)
	}
	if math.IsNaN(float64(result[0])) || math.IsInf(float64(result[0]), 0) {
		t.Errorf("q4_1 result = %f, expected finite", result[0])
	}
}

func TestModelMatmulQuantF32Q5(t *testing.T) {
	group := make([]byte, 22)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	binary.LittleEndian.PutUint32(group[2:6], 0)
	for i := 6; i < 22; i++ {
		group[i] = byte(i)
	}

	w := &QuantWeight{QType: "q5", Raw: group, Groups: 1, Rows: 1, Cols: 32}
	xData := make([]float32, 32)
	for i := range xData {
		xData[i] = float32(i) * 0.1
	}

	result, rows, cols := modelMatmulQuantF32(w, xData, 1, 32)
	if rows != 1 || cols != 1 {
		t.Fatalf("shape = %dx%d, want 1x1", rows, cols)
	}
	if math.IsNaN(float64(result[0])) || math.IsInf(float64(result[0]), 0) {
		t.Errorf("q5 result = %f, expected finite", result[0])
	}
}

func TestModelMatmulQuantF32Unknown(t *testing.T) {
	w := &QuantWeight{QType: "unknown", Raw: nil, Groups: 0, Rows: 0, Cols: 0}
	result, rows, cols := modelMatmulQuantF32(w, []float32{1.0}, 1, 1)
	if result != nil || rows != 0 || cols != 0 {
		t.Errorf("unknown type should return nil, got %v %d %d", result, rows, cols)
	}
}

func TestModelMatmulRowQuantIntoF32Q8(t *testing.T) {
	qValues := make([]int8, 32)
	for i := range qValues {
		qValues[i] = 1
	}
	raw := makeQ80Raw(makeQ80Group(0x3C00, qValues))

	w := &QuantWeight{QType: "q8", Raw: []byte(raw.StringValue()), Groups: 1, Rows: 1, Cols: 32}
	normed := make([]float32, 32)
	for i := range normed {
		normed[i] = 1.0
	}
	dst := make([]float32, 1)

	result := modelMatmulRowQuantIntoF32(w, normed, dst)
	if math.Abs(float64(result[0])-32.0) > 0.01 {
		t.Errorf("q8 into = %f, want ~32.0", result[0])
	}
}

func TestModelMatmulRowQuantIntoF32Q4(t *testing.T) {
	group := make([]byte, 18)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	for i := 2; i < 18; i++ {
		group[i] = 0x99
	}

	w := &QuantWeight{QType: "q4", Raw: group, Groups: 1, Rows: 1, Cols: 32}
	normed := make([]float32, 32)
	for i := range normed {
		normed[i] = 1.0
	}
	dst := make([]float32, 1)

	result := modelMatmulRowQuantIntoF32(w, normed, dst)
	if math.Abs(float64(result[0])-32.0) > 0.01 {
		t.Errorf("q4 into = %f, want ~32.0", result[0])
	}
}

func TestModelMatmulRowQuantIntoF32Q41(t *testing.T) {
	group := make([]byte, 20)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	binary.LittleEndian.PutUint16(group[2:4], 0x0000)
	for i := 4; i < 20; i++ {
		group[i] = 0x55
	}

	w := &QuantWeight{QType: "q4_1", Raw: group, Groups: 1, Rows: 1, Cols: 32}
	normed := make([]float32, 32)
	for i := range normed {
		normed[i] = 1.0
	}
	dst := make([]float32, 1)

	result := modelMatmulRowQuantIntoF32(w, normed, dst)
	if math.IsNaN(float64(result[0])) || math.IsInf(float64(result[0]), 0) {
		t.Errorf("q4_1 into = %f, expected finite", result[0])
	}
}

func TestModelMatmulRowQuantIntoF32Q5(t *testing.T) {
	group := make([]byte, 22)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	binary.LittleEndian.PutUint32(group[2:6], 0)
	for i := 6; i < 22; i++ {
		group[i] = byte(i)
	}

	w := &QuantWeight{QType: "q5", Raw: group, Groups: 1, Rows: 1, Cols: 32}
	normed := make([]float32, 32)
	for i := range normed {
		normed[i] = float32(i) * 0.1
	}
	dst := make([]float32, 1)

	result := modelMatmulRowQuantIntoF32(w, normed, dst)
	if math.IsNaN(float64(result[0])) || math.IsInf(float64(result[0]), 0) {
		t.Errorf("q5 into = %f, expected finite", result[0])
	}
}

func TestModelMatmulRowQuantIntoF32Unknown(t *testing.T) {
	w := &QuantWeight{QType: "unknown", Raw: nil, Groups: 0, Rows: 0, Cols: 0}
	dst := make([]float32, 0)
	result := modelMatmulRowQuantIntoF32(w, []float32{}, dst)
	if result != nil {
		t.Errorf("unknown type should return nil, got %v", result)
	}
}

func TestModelMatmulRowQuantF32Q4(t *testing.T) {
	group := make([]byte, 18)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	for i := 2; i < 18; i++ {
		group[i] = 0x99
	}

	w := &QuantWeight{QType: "q4", Raw: group, Groups: 1, Rows: 1, Cols: 32}
	normed := make([]float32, 32)
	for i := range normed {
		normed[i] = 1.0
	}

	result := modelMatmulRowQuantF32(w, normed)
	if len(result) != 1 {
		t.Fatalf("len = %d, want 1", len(result))
	}
	if math.Abs(float64(result[0])-32.0) > 0.01 {
		t.Errorf("q4 row = %f, want ~32.0", result[0])
	}
}

func TestModelMatmulRowQuantF32Q41(t *testing.T) {
	group := make([]byte, 20)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	binary.LittleEndian.PutUint16(group[2:4], 0x0000)
	for i := 4; i < 20; i++ {
		group[i] = 0x55
	}

	w := &QuantWeight{QType: "q4_1", Raw: group, Groups: 1, Rows: 1, Cols: 32}
	normed := make([]float32, 32)
	for i := range normed {
		normed[i] = 1.0
	}

	result := modelMatmulRowQuantF32(w, normed)
	if len(result) != 1 {
		t.Fatalf("len = %d, want 1", len(result))
	}
	if math.IsNaN(float64(result[0])) || math.IsInf(float64(result[0]), 0) {
		t.Errorf("q4_1 row = %f, expected finite", result[0])
	}
}

func TestModelMatmulRowQuantF32Q5(t *testing.T) {
	group := make([]byte, 22)
	binary.LittleEndian.PutUint16(group[0:2], 0x3C00)
	binary.LittleEndian.PutUint32(group[2:6], 0)
	for i := 6; i < 22; i++ {
		group[i] = byte(i)
	}

	w := &QuantWeight{QType: "q5", Raw: group, Groups: 1, Rows: 1, Cols: 32}
	normed := make([]float32, 32)
	for i := range normed {
		normed[i] = float32(i) * 0.1
	}

	result := modelMatmulRowQuantF32(w, normed)
	if len(result) != 1 {
		t.Fatalf("len = %d, want 1", len(result))
	}
	if math.IsNaN(float64(result[0])) || math.IsInf(float64(result[0]), 0) {
		t.Errorf("q5 row = %f, expected finite", result[0])
	}
}

func TestModelMatmulRowQuantF32Unknown(t *testing.T) {
	w := &QuantWeight{QType: "unknown", Raw: nil, Groups: 0, Rows: 0, Cols: 0}
	result := modelMatmulRowQuantF32(w, []float32{})
	if result != nil {
		t.Errorf("unknown type should return nil, got %v", result)
	}
}

func TestStripThinkTags(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"no tags", "hello world", "hello world"},
		{"open only", "before<think stuff", "before"},
		{"paired", "before<think hidden</thinkafter", "beforeafter"},
		{"multiple", "a<think x</thinkb<think y</thinkc", "abc"},
		{"only think block", "<think hidden</think", ""},
		{"whitespace", "  hello  ", "hello"},
		{"think at start", "<think reasoning</thinkanswer", "answer"},
		{"empty after strip", "<think entire thing</think", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripThinkTags(tt.in)
			if got != tt.want {
				t.Errorf("stripThinkTags(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestSessionStoreF32(t *testing.T) {
	mc := &modelCacheF32{
		models:   make(map[string]*InferenceModelF32),
		sessions: make(map[string]map[string]*sessionEntryF32),
	}

	if entry := mc.getSession("model.gguf", "s1"); entry != nil {
		t.Error("getSession on empty should return nil")
	}

	caches := []KVCacheF32{
		{K: [][]float32{{1.0, 2.0}}, V: [][]float32{{3.0, 4.0}}},
	}
	mc.saveSession("model.gguf", "s1", caches, 10)

	entry := mc.getSession("model.gguf", "s1")
	if entry == nil {
		t.Fatal("getSession after save should return entry")
	}
	if entry.kvPos != 10 {
		t.Errorf("kvPos = %d, want 10", entry.kvPos)
	}

	mc.clearSession("model.gguf", "s1")
	if mc.getSession("model.gguf", "s1") != nil {
		t.Error("getSession after clear should return nil")
	}
}
