package scriptlingllmlib

import (
	"math"
	"os"
	"testing"
)

// A fixed, plain-English passage. Perplexity over it is a precise, deterministic
// probe of the forward pass + quantized kernels: a kernel regression that
// corrupts the probability distribution moves perplexity sharply.
const pplText = "The quick brown fox jumps over the lazy dog. " +
	"Paris is the capital of France, and the Eiffel Tower stands beside the river Seine. " +
	"Water boils at one hundred degrees Celsius at sea level."

func loadModelForTest(t *testing.T, path string) *InferenceModelF32 {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present: %s", path)
	}
	gguf, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	gguf.Metadata["_path"] = path
	m, err := buildInferenceModelF32(gguf, path)
	gguf.ReleaseFileData()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	return m
}

// perplexity returns exp(mean token negative-log-likelihood) of ids under m,
// teacher-forced: at each position the model predicts the actual next token.
func perplexity(m *InferenceModelF32, ids []int) float64 {
	m.initKVCaches()
	var nll float64
	var cnt int
	for i := 0; i+1 < len(ids); i++ {
		logits := m.Forward([]int{ids[i]}, i)
		maxv := float32(math.Inf(-1))
		for _, v := range logits {
			if v > maxv {
				maxv = v
			}
		}
		var sum float64
		for _, v := range logits {
			sum += math.Exp(float64(v - maxv))
		}
		logZ := math.Log(sum) + float64(maxv)
		nll += logZ - float64(logits[ids[i+1]])
		cnt++
	}
	return math.Exp(nll / float64(cnt))
}

// Golden perplexities for pplText, measured on the AVX2 int8 path. The ±10% band
// absorbs cross-platform floating-point and int8-vs-scalar activation differences
// while still catching gross kernel corruption (which sends perplexity far out of
// band, or to Inf/NaN). Update these only with an intended accuracy change.
var pplGolden = []struct {
	path string
	want float64
}{
	{"models/SmolLM2-135M-Instruct-Q8_0.gguf", 12.785},
	{"models/SmolLM2-135M-Instruct-Q4_0.gguf", 16.813},
}

const pplTol = 0.10

// TestPerplexityRegression guards the quantized kernels against silent accuracy
// drift: each model's perplexity over a fixed passage must stay within the golden
// band, and finite.
func TestPerplexityRegression(t *testing.T) {
	for _, g := range pplGolden {
		m := loadModelForTest(t, g.path)
		ids := m.Tokenizer.Encode(pplText)
		got := perplexity(m, ids)
		t.Logf("%s: %d tokens, perplexity = %.4f (golden %.4f)", g.path, len(ids), got, g.want)
		if math.IsNaN(got) || math.IsInf(got, 0) || got <= 1 {
			t.Fatalf("%s: perplexity = %v, want finite > 1", g.path, got)
		}
		lo, hi := g.want*(1-pplTol), g.want*(1+pplTol)
		if got < lo || got > hi {
			t.Errorf("%s: perplexity = %.4f, want within [%.4f, %.4f] (golden %.4f ±%.0f%%)",
				g.path, got, lo, hi, g.want, pplTol*100)
		}
	}
}

// TestQuantPrecisionOrdering is a platform-independent invariant: the Q8 model
// must predict the passage at least as well as the Q4 model (higher-precision
// weights -> not-higher perplexity). A broken dequant for either type breaks it.
func TestQuantPrecisionOrdering(t *testing.T) {
	q8 := loadModelForTest(t, "models/SmolLM2-135M-Instruct-Q8_0.gguf")
	q4 := loadModelForTest(t, "models/SmolLM2-135M-Instruct-Q4_0.gguf")
	ids := q8.Tokenizer.Encode(pplText)
	pplQ8, pplQ4 := perplexity(q8, ids), perplexity(q4, ids)
	t.Logf("Q8 ppl = %.4f, Q4 ppl = %.4f", pplQ8, pplQ4)
	if pplQ8 > pplQ4 {
		t.Errorf("Q8 perplexity %.4f exceeds Q4 %.4f — expected higher precision to be no worse", pplQ8, pplQ4)
	}
}
