package scriptlingllmlib

import (
	"math"
	"os"
	"testing"
)

func TestKVCacheStructure(t *testing.T) {
	cache := KVCache{
		K: [][]float64{{1.0, 2.0}, {3.0, 4.0}},
		V: [][]float64{{5.0, 6.0}, {7.0, 8.0}},
	}
	if len(cache.K) != 2 || len(cache.V) != 2 {
		t.Error("KVCache dimension mismatch")
	}
	if cache.K[0][0] != 1.0 || cache.V[1][1] != 8.0 {
		t.Error("KVCache values incorrect")
	}
}

func TestTransformerBlockStructure(t *testing.T) {
	block := TransformerBlock{
		AttnNormW: []float64{1.0, 2.0},
		WQ:        [][]float64{{0.1, 0.2}, {0.3, 0.4}},
		WK:        [][]float64{{0.5, 0.6}, {0.7, 0.8}},
		WV:        [][]float64{{0.9, 1.0}, {1.1, 1.2}},
		WO:        [][]float64{{1.3, 1.4}, {1.5, 1.6}},
		FFNNormW:  []float64{1.0},
		WGate:     [][]float64{{0.1}},
		WUp:       [][]float64{{0.2}},
		WDown:     [][]float64{{0.3}},
	}
	if len(block.AttnNormW) != 2 {
		t.Error("AttnNormW length mismatch")
	}
	if _, ok := block.WQ.([][]float64); !ok {
		t.Error("WQ should be [][]float64")
	}
}

func TestTransformerBlockQuantWeights(t *testing.T) {
	block := TransformerBlock{
		AttnNormW: []float64{1.0},
		WQ: &QuantWeight{
			QType:  "q4k",
			Raw:    make([]byte, 144),
			Groups: 1,
			Rows:   1,
			Cols:   256,
		},
		WK:       &QuantWeight{QType: "q8", Raw: make([]byte, 34), Groups: 1, Rows: 1, Cols: 32},
		WV:       &QuantWeight{QType: "q8", Raw: make([]byte, 34), Groups: 1, Rows: 1, Cols: 32},
		WO:       &QuantWeight{QType: "q8", Raw: make([]byte, 34), Groups: 1, Rows: 1, Cols: 32},
		FFNNormW: []float64{1.0},
		WGate:    &QuantWeight{QType: "q4k", Raw: make([]byte, 144), Groups: 1, Rows: 1, Cols: 256},
		WUp:      &QuantWeight{QType: "q4k", Raw: make([]byte, 144), Groups: 1, Rows: 1, Cols: 256},
		WDown:    &QuantWeight{QType: "q4k", Raw: make([]byte, 144), Groups: 1, Rows: 1, Cols: 256},
	}
	if qw, ok := block.WQ.(*QuantWeight); !ok {
		t.Error("WQ should be *QuantWeight")
	} else if qw.QType != "q4k" {
		t.Errorf("WQ QType = %q, want 'q4k'", qw.QType)
	}
}

func TestInferenceModelGenerate(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping inference test in short mode")
	}

	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("model file not found")
	}

	model, err := buildInferenceModelTest(path)
	if err != nil {
		t.Fatalf("buildInferenceModel failed: %v", err)
	}

	model.initKVCaches()

	output, nGen, nPrompt := model.Generate("Hello", 10, "greedy", 0, 0, 0, 0, 0, "", "")

	if nPrompt == 0 {
		t.Error("prompt tokens should be > 0")
	}
	if nGen == 0 {
		t.Error("generated tokens should be > 0")
	}
	if output == "" {
		t.Error("output should not be empty")
	}

	t.Logf("Generated (%d prompt + %d gen): %q", nPrompt, nGen, output)
}

func TestInferenceModelForward(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping forward pass test in short mode")
	}

	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("model file not found")
	}

	model, err := buildInferenceModelTest(path)
	if err != nil {
		t.Fatalf("buildInferenceModel failed: %v", err)
	}

	model.initKVCaches()

	ids := model.Tokenizer.Encode("Hello")
	if len(ids) == 0 {
		t.Fatal("tokenizer returned empty ids for 'Hello'")
	}

	logits := model.Forward(ids, 0)
	if len(logits) != model.Config.VocabSize {
		t.Fatalf("logits length = %d, want VocabSize %d", len(logits), model.Config.VocabSize)
	}

	hasNaN := false
	hasInf := false
	for _, v := range logits {
		if math.IsNaN(v) {
			hasNaN = true
		}
		if math.IsInf(v, 0) {
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
	t.Logf("Top token after 'Hello': %d (%q)", maxIdx, model.Tokenizer.IDToToken[maxIdx])
}

func TestInferenceModelTokenizer(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping tokenizer integration test in short mode")
	}

	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("model file not found")
	}

	model, err := buildInferenceModelTest(path)
	if err != nil {
		t.Fatalf("buildInferenceModel failed: %v", err)
	}

	if model.Tokenizer == nil {
		t.Fatal("model tokenizer is nil")
	}

	ids := model.Tokenizer.Encode("Hello world")
	if len(ids) == 0 {
		t.Fatal("encode returned empty")
	}

	decoded := model.Tokenizer.Decode(ids)
	if decoded == "" {
		t.Error("decode returned empty string")
	}
	t.Logf("Encode/Decode: 'Hello world' -> %v -> %q", ids, decoded)
}

func buildInferenceModelTest(path string) (*InferenceModel, error) {
	gguf, err := LoadGGUF(path)
	if err != nil {
		return nil, err
	}
	gguf.Metadata["_path"] = path
	return buildInferenceModel(gguf, path)
}
