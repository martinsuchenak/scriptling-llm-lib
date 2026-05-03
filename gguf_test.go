package scriptlingllmlib

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func TestGGUFTypeSizes(t *testing.T) {
	expected := map[int]int{
		0:  4,
		1:  2,
		2:  18,
		3:  20,
		6:  22,
		8:  34,
		12: 144,
		14: 210,
	}
	for k, v := range expected {
		if ggufTypeSizes[k] != v {
			t.Errorf("ggufTypeSizes[%d] = %d, want %d", k, ggufTypeSizes[k], v)
		}
	}
}

func TestGGUFTensorNameMapping(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"blk.0.attn_norm.weight", "blocks.0.attn_norm.weight"},
		{"blk.0.attn_q.weight", "blocks.0.attn.w_q.weight"},
		{"blk.0.attn_k.weight", "blocks.0.attn.w_k.weight"},
		{"blk.0.attn_v.weight", "blocks.0.attn.w_v.weight"},
		{"blk.0.attn_output.weight", "blocks.0.attn.w_o.weight"},
		{"blk.0.ffn_gate.weight", "blocks.0.ffn.w_gate.weight"},
		{"blk.0.ffn_up.weight", "blocks.0.ffn.w_up.weight"},
		{"blk.0.ffn_down.weight", "blocks.0.ffn.w_down.weight"},
		{"blk.0.ffn_norm.weight", "blocks.0.ffn_norm.weight"},
		{"token_embd.weight", "token_embedding.weight"},
		{"output_norm.weight", "final_norm.weight"},
		{"output.weight", "output.weight"},
	}
	for _, tc := range tests {
		got := mapTensorName(tc.input)
		if got != tc.want {
			t.Errorf("mapTensorName(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestGGUFLoadNonexistentFile(t *testing.T) {
	_, err := LoadGGUF("/nonexistent/path/model.gguf")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestGGUFLoadInvalidFile(t *testing.T) {
	f, err := os.CreateTemp("", "invalid-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	f.Write([]byte("NOT_A_GGUF_FILE"))
	f.Close()

	_, err = LoadGGUF(f.Name())
	if err == nil {
		t.Error("expected error for invalid GGUF file")
	}
}

func TestGGUFLoadRealFile(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GGUF load test in short mode")
	}

	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("model file not found")
	}

	model, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF failed: %v", err)
	}

	if model.Version != 3 {
		t.Errorf("GGUF version = %d, want 3", model.Version)
	}

	if model.Config.VocabSize == 0 {
		t.Error("VocabSize should not be 0")
	}
	if model.Config.DModel == 0 {
		t.Error("DModel should not be 0")
	}
	if model.Config.NLayers == 0 {
		t.Error("NLayers should not be 0")
	}
	if model.Config.NHeads == 0 {
		t.Error("NHeads should not be 0")
	}

	if model.Tokenizer == nil {
		t.Fatal("Tokenizer should not be nil")
	}
	if len(model.Tokenizer.Tokens) == 0 {
		t.Error("Tokenizer tokens should not be empty")
	}
	if len(model.Tokenizer.Vocab) == 0 {
		t.Error("Tokenizer vocab should not be empty")
	}

	if len(model.Tensors) == 0 {
		t.Error("Tensors should not be empty")
	}

	expectedTensors := []string{
		"token_embedding.weight",
		"final_norm.weight",
	}
	for _, name := range expectedTensors {
		if _, ok := model.Tensors[name]; !ok {
			t.Errorf("missing tensor: %s", name)
		}
	}

	if _, ok := model.Tensors["output.weight"]; !ok {
		t.Log("output.weight not present (tied weights, uses token_embedding)")
	}

	if model.Tokenizer.ChatTemplate != "" {
		t.Logf("Chat template: %s", model.Tokenizer.ChatTemplate[:min(80, len(model.Tokenizer.ChatTemplate))])
	}
}

func TestGGUFLoadTensorTokenEmbedding(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping tensor load test in short mode")
	}

	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("model file not found")
	}

	model, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF failed: %v", err)
	}
	model.Metadata["_path"] = path

	emb, err := model.LoadTensor("token_embedding.weight")
	if err != nil {
		t.Fatalf("LoadTensor failed: %v", err)
	}

	embMatrix, ok := emb.([][]float64)
	if !ok {
		t.Fatal("token_embedding should be [][]float64")
	}

	if len(embMatrix) != model.Config.VocabSize {
		t.Errorf("embedding rows = %d, want VocabSize %d", len(embMatrix), model.Config.VocabSize)
	}
	if len(embMatrix[0]) != model.Config.DModel {
		t.Errorf("embedding cols = %d, want DModel %d", len(embMatrix[0]), model.Config.DModel)
	}

	for i := 0; i < min(10, len(embMatrix)); i++ {
		for _, v := range embMatrix[i] {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("embedding[%d] contains NaN/Inf", i)
				break
			}
		}
	}
}

func TestGGUFLoadTensorOutputWeight(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping tensor load test in short mode")
	}

	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("model file not found")
	}

	model, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF failed: %v", err)
	}
	model.Metadata["_path"] = path

	if _, ok := model.Tensors["output.weight"]; !ok {
		t.Skip("output.weight not present (tied weights)")
	}

	output, err := model.LoadTensor("output.weight")
	if err != nil {
		t.Fatalf("LoadTensor output failed: %v", err)
	}

	switch w := output.(type) {
	case *QuantWeight:
		if w.Rows == 0 || w.Cols == 0 {
			t.Errorf("QuantWeight rows=%d cols=%d, both should be > 0", w.Rows, w.Cols)
		}
		if len(w.Raw) == 0 {
			t.Error("QuantWeight Raw should not be empty")
		}
		t.Logf("output weight: QuantWeight type=%s rows=%d cols=%d", w.QType, w.Rows, w.Cols)
	case [][]float64:
		if len(w) == 0 || len(w[0]) == 0 {
			t.Error("output weight should have non-zero dimensions")
		}
		t.Logf("output weight: [][]float64 %dx%d", len(w), len(w[0]))
	default:
		t.Errorf("unexpected output weight type: %T", output)
	}
}

func TestGGUFLoadTensorNorm(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping tensor load test in short mode")
	}

	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("model file not found")
	}

	model, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF failed: %v", err)
	}
	model.Metadata["_path"] = path

	norm, err := model.LoadTensor("final_norm.weight")
	if err != nil {
		t.Fatalf("LoadTensor norm failed: %v", err)
	}

	normVec, ok := norm.([]float64)
	if !ok {
		t.Fatalf("final_norm should be []float64, got %T", norm)
	}
	if len(normVec) != model.Config.DModel {
		t.Errorf("norm length = %d, want DModel %d", len(normVec), model.Config.DModel)
	}
}

func TestQuantWeightStructure(t *testing.T) {
	qw := &QuantWeight{
		QType:  "q8",
		Raw:    make([]byte, 34),
		Groups: 1,
		Rows:   1,
		Cols:   32,
	}
	if qw.QType != "q8" {
		t.Error("QType mismatch")
	}
	if len(qw.Raw) != 34 {
		t.Error("Raw length mismatch")
	}
}

func TestModelConfigDefaults(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:    49152,
		DModel:       576,
		NHeads:       9,
		NKVHeads:     3,
		NLayers:      30,
		MaxSeqLen:    8192,
		DFF:          1536,
		NormEps:      1e-5,
		RopeFreqBase: 100000.0,
		RopeDim:      32,
	}
	if cfg.DModel%cfg.NHeads != 0 {
		t.Errorf("DModel (%d) must be divisible by NHeads (%d)", cfg.DModel, cfg.NHeads)
	}
	dK := cfg.DModel / cfg.NHeads
	if dK != 64 {
		t.Errorf("dK = %d, want 64", dK)
	}
	if cfg.NHeads%cfg.NKVHeads != 0 {
		t.Errorf("NHeads (%d) must be divisible by NKVHeads (%d)", cfg.NHeads, cfg.NKVHeads)
	}
}

func TestFloat16Conversion(t *testing.T) {
	tests := []struct {
		f    float64
		want uint16
	}{
		{0.0, 0x0000},
		{1.0, 0x3C00},
		{-1.0, 0xBC00},
		{0.5, 0x3800},
		{2.0, 0x4000},
		{65504.0, 0x7BFF},
		{-65504.0, 0xFBFF},
	}
	for _, tc := range tests {
		got := float64ToFloat16(tc.f)
		if got != tc.want {
			t.Errorf("float64ToFloat16(%v) = 0x%04X, want 0x%04X", tc.f, got, tc.want)
		}
	}
}

func TestFloat16Roundtrip(t *testing.T) {
	values := []float64{0.0, 1.0, -1.0, 0.5, 127.0, 0.12345, 10000.0}
	for _, v := range values {
		bits := float64ToFloat16(v)
		roundtrip := float16ToFloat64(bits)
		err := math.Abs(roundtrip - v)
		relErr := err / math.Max(math.Abs(v), 1e-10)
		if relErr > 0.01 {
			t.Errorf("float16 roundtrip %f -> 0x%04X -> %f (rel err %f)", v, bits, roundtrip, relErr)
		}
	}
}

func TestQ8QuantizeDequantRoundtrip(t *testing.T) {
	nRows := 2
	nCols := 32
	data := make([]float64, nRows*nCols)
	for i := range data {
		data[i] = float64(i-32) * 0.1
	}

	raw := quantizeQ8RowsPure(data, nRows, nCols)

	dequantized := make([][]float64, nRows)
	for r := 0; r < nRows; r++ {
		dequantized[r] = make([]float64, nCols)
		for g := 0; g < nCols/32; g++ {
			off := r*(nCols/32)*34 + g*34
			scaleBits := binary.LittleEndian.Uint16(raw[off : off+2])
			scale := float16ToFloat64(scaleBits)
			for i := 0; i < 32; i++ {
				q := int8(raw[off+2+i])
				dequantized[r][g*32+i] = float64(q) * scale
			}
		}
	}

	var maxErr float64
	for i := range data {
		row := i / nCols
		col := i % nCols
		err := math.Abs(data[i] - dequantized[row][col])
		if err > maxErr {
			maxErr = err
		}
	}
	if maxErr > 0.5 {
		t.Errorf("Q8 roundtrip max error = %f, want < 0.5", maxErr)
	}
}

func quantizeQ8RowsPure(data []float64, nRows, nCols int) []byte {
	groupsPerRow := nCols / 32
	totalGroups := nRows * groupsPerRow
	result := make([]byte, 0, totalGroups*34)
	scaleBytes := make([]byte, 2)

	for r := 0; r < nRows; r++ {
		rowOff := r * nCols
		for g := 0; g < groupsPerRow; g++ {
			base := rowOff + g*32
			var maxAbs float64
			for i := 0; i < 32; i++ {
				v := math.Abs(data[base+i])
				if v > maxAbs {
					maxAbs = v
				}
			}
			var scale float64
			if maxAbs > 0 {
				scale = maxAbs / 127.0
			}
			binary.LittleEndian.PutUint16(scaleBytes, float64ToFloat16(scale))
			result = append(result, scaleBytes...)
			invScale := 1.0 / scale
			for i := 0; i < 32; i++ {
				q := int8(data[base+i] * invScale)
				if data[base+i]*invScale > 127 {
					q = 127
				} else if data[base+i]*invScale < -128 {
					q = -128
				}
				result = append(result, byte(q))
			}
		}
	}
	return result
}
