package scriptlingllmlib

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/paularlott/scriptling"
	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

func init() {
	_ = context.Background
	_ = errors.ExactArgs
	_ = scriptling.New
	_ = os.Getenv
	_ = math.NaN
}

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
			QType:  "q8",
			Raw:    make([]byte, 34),
			Groups: 1,
			Rows:   1,
			Cols:   32,
		},
		WK:       &QuantWeight{QType: "q8", Raw: make([]byte, 34), Groups: 1, Rows: 1, Cols: 32},
		WV:       &QuantWeight{QType: "q8", Raw: make([]byte, 34), Groups: 1, Rows: 1, Cols: 32},
		WO:       &QuantWeight{QType: "q8", Raw: make([]byte, 34), Groups: 1, Rows: 1, Cols: 32},
		FFNNormW: []float64{1.0},
		WGate:    &QuantWeight{QType: "q8", Raw: make([]byte, 34), Groups: 1, Rows: 1, Cols: 32},
		WUp:      &QuantWeight{QType: "q8", Raw: make([]byte, 34), Groups: 1, Rows: 1, Cols: 32},
		WDown:    &QuantWeight{QType: "q8", Raw: make([]byte, 34), Groups: 1, Rows: 1, Cols: 32},
	}
	if qw, ok := block.WQ.(*QuantWeight); !ok {
		t.Error("WQ should be *QuantWeight")
	} else if qw.QType != "q8" {
		t.Errorf("WQ QType = %q, want 'q8'", qw.QType)
	}
}

func TestInferenceModelGenerate(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping inference test in short mode")
	}
	if raceEnabled {
		t.Skip("skipping inference test under race detector")
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

	t.Logf("Generated (%d prompt + %d gen): %q", nPrompt, nGen, output)
}

func TestInferenceModelForward(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping forward pass test in short mode")
	}
	if raceEnabled {
		t.Skip("skipping forward pass test under race detector")
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
	if raceEnabled {
		t.Skip("skipping tokenizer test under race detector")
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

func TestCopyKVCaches(t *testing.T) {
	original := []KVCache{
		{
			K: [][]float64{{1.0, 2.0, 3.0}},
			V: [][]float64{{4.0, 5.0, 6.0}},
		},
	}

	copied := copyKVCaches(original)

	if len(copied) != 1 {
		t.Fatalf("copyKVCaches len = %d, want 1", len(copied))
	}

	copied[0].K[0][0] = 999.0
	if original[0].K[0][0] == 999.0 {
		t.Error("copyKVCaches did not deep copy - modifying copy affected original")
	}
}

func TestSessionStoreGetSaveClear(t *testing.T) {
	mc := &modelCache{
		models:   make(map[string]*InferenceModel),
		sessions: make(map[string]map[string]*sessionEntry),
	}

	if entry := mc.getSession("model.gguf", "s1"); entry != nil {
		t.Error("getSession on empty should return nil")
	}

	caches := []KVCache{
		{K: [][]float64{{1.0, 2.0}}, V: [][]float64{{3.0, 4.0}}},
	}
	mc.saveSession("model.gguf", "s1", caches, 10)

	entry := mc.getSession("model.gguf", "s1")
	if entry == nil {
		t.Fatal("getSession after save should return entry")
	}
	if entry.kvPos != 10 {
		t.Errorf("kvPos = %d, want 10", entry.kvPos)
	}

	mc.saveSession("model.gguf", "s2", caches, 20)
	if len(mc.sessions["model.gguf"]) != 2 {
		t.Errorf("expected 2 sessions, got %d", len(mc.sessions["model.gguf"]))
	}

	mc.clearSession("model.gguf", "s1")
	if mc.getSession("model.gguf", "s1") != nil {
		t.Error("getSession after clear should return nil")
	}
	if mc.getSession("model.gguf", "s2") == nil {
		t.Error("s2 should still exist")
	}

	mc.clearSession("model.gguf", "s2")
	if len(mc.sessions["model.gguf"]) != 0 {
		t.Error("sessions map should be empty after clearing all sessions")
	}
}

func TestClearSessionFn(t *testing.T) {
	assertError(t, fnClearSession(ctx, noopKwargs), "2 arguments")
	assertError(t, fnClearSession(ctx, noopKwargs, &object.Integer{Value: 1}, &object.String{Value: "s1"}), "STRING")
	assertError(t, fnClearSession(ctx, noopKwargs, &object.String{Value: "model.gguf"}, &object.Integer{Value: 1}), "STRING")

	result := fnClearSession(ctx, noopKwargs, &object.String{Value: "model.gguf"}, &object.String{Value: "nonexistent"})
	b, err := result.AsBool()
	if err != nil || !b {
		t.Error("clear_session on nonexistent session should return true")
	}
}

func TestGenerateWithSessionIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping session integration test in short mode")
	}
	if raceEnabled {
		t.Skip("skipping session integration test under race detector")
	}

	globalModelCacheF32.clearModels()

	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("model file not found")
	}

	kwargs1 := object.NewKwargs(map[string]object.Object{
		"session": &object.String{Value: "test-session-1"},
		"stats":   object.NewInteger(0),
	})

	result1 := fnGenerate(ctx, kwargs1,
		&object.String{Value: path},
		&object.String{Value: "Hello"},
		object.NewInteger(3),
		&object.String{Value: "greedy"},
	)

	s1, ok := result1.(*object.String)
	if !ok {
		t.Fatalf("first generate returned %s, want STRING", result1.Type().String())
	}
	if s1.Value == "" {
		t.Error("first generate returned empty string")
	}

	globalModelCacheF32.mu.Lock()
	entry := globalModelCacheF32.getSession(path, "test-session-1")
	if entry == nil {
		t.Fatal("session not found after first generate")
	}
	if entry.kvPos <= 0 {
		t.Errorf("kvPos = %d, want > 0", entry.kvPos)
	}
	savedPos := entry.kvPos
	globalModelCacheF32.mu.Unlock()

	kwargs2 := object.NewKwargs(map[string]object.Object{
		"session": &object.String{Value: "test-session-1"},
	})

	result2 := fnGenerate(ctx, kwargs2,
		&object.String{Value: path},
		&object.String{Value: "What else"},
		object.NewInteger(2),
		&object.String{Value: "greedy"},
	)

	s2, ok := result2.(*object.String)
	if !ok {
		t.Fatalf("second generate returned %s, want STRING", result2.Type().String())
	}
	if s2.Value == "" {
		t.Error("second generate returned empty string")
	}

	globalModelCacheF32.mu.Lock()
	entry2 := globalModelCacheF32.getSession(path, "test-session-1")
	if entry2.kvPos <= savedPos {
		t.Errorf("kvPos after second generate = %d, should be > first kvPos %d", entry2.kvPos, savedPos)
	}
	globalModelCacheF32.mu.Unlock()

	fnClearSession(ctx, noopKwargs, &object.String{Value: path}, &object.String{Value: "test-session-1"})

	globalModelCacheF32.mu.Lock()
	if globalModelCacheF32.getSession(path, "test-session-1") != nil {
		t.Error("session should be cleared")
	}
	globalModelCacheF32.mu.Unlock()
}

func TestSampleLogitsEmpty(t *testing.T) {
	got := sampleLogits([]float64{}, "greedy", 1.0, 0, 0.9)
	if got != 0 {
		t.Errorf("empty -> %d, want 0", got)
	}
}

func TestSampleLogitsGreedy(t *testing.T) {
	got := sampleLogits([]float64{0.1, 0.9, 0.3}, "greedy", 1.0, 0, 0.9)
	if got != 1 {
		t.Errorf("greedy -> %d, want 1", got)
	}
}

func TestSampleLogitsTemperature(t *testing.T) {
	got := sampleLogits([]float64{0.1, 0.9, 0.3}, "temperature", 1.0, 0, 0.9)
	if got < 0 || got > 2 {
		t.Errorf("temperature -> %d, out of range [0,2]", got)
	}
}

func TestSampleLogitsTopK(t *testing.T) {
	got := sampleLogits([]float64{0.1, 0.9, 0.3}, "top_k", 1.0, 0, 0.9)
	if got < 0 || got > 2 {
		t.Errorf("top_k default -> %d, out of range [0,2]", got)
	}
}

func TestSampleLogitsTopKExplicit(t *testing.T) {
	got := sampleLogits([]float64{0.1, 0.9, 0.3}, "top_k", 1.0, 2, 0.9)
	if got < 0 || got > 2 {
		t.Errorf("top_k explicit -> %d, out of range [0,2]", got)
	}
}

func TestSampleLogitsTopKClamp(t *testing.T) {
	got := sampleLogits([]float64{0.1, 0.9, 0.3}, "top_k", 1.0, 100, 0.9)
	if got < 0 || got > 2 {
		t.Errorf("top_k clamped -> %d, out of range [0,2]", got)
	}
}

func TestSampleLogitsTopP(t *testing.T) {
	got := sampleLogits([]float64{0.1, 0.9, 0.3}, "top_p", 1.0, 0, 0.9)
	if got < 0 || got > 2 {
		t.Errorf("top_p -> %d, out of range [0,2]", got)
	}
}

func TestSampleLogitsUnknownStrategy(t *testing.T) {
	got := sampleLogits([]float64{0.1, 0.9}, "unknown", 1.0, 0, 0.9)
	if got != 0 {
		t.Errorf("unknown -> %d, want 0", got)
	}
}
