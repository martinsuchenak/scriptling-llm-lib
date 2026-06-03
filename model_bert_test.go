package scriptlingllmlib

import (
	"math"
	"os"
	"sync"
	"testing"
)

const bertTestModel = "models/minilm.gguf" // all-MiniLM-L6-v2 Q8_0

func bertCos(a, b []float32) float64 {
	var d, na, nb float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return d / (math.Sqrt(na) * math.Sqrt(nb))
}

// TestWordPieceMatchesBert validates the tokenizer against known bert-base-uncased
// ids — exact-match here is the strongest single check that an embedding model
// will reproduce reference embeddings.
func TestWordPieceMatchesBert(t *testing.T) {
	if _, err := os.Stat(bertTestModel); err != nil {
		t.Skip("model not present")
	}
	b, err := getOrLoadBert(bertTestModel)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		text string
		want []int
	}{
		{"hello world", []int{101, 7592, 2088, 102}},
		{"embeddings", []int{101, 7861, 8270, 4667, 2015, 102}}, // em ##bed ##ding ##s
	}
	for _, c := range cases {
		got := b.Tok.encode(c.text)
		if len(got) != len(c.want) {
			t.Errorf("%q -> %v, want %v", c.text, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("%q -> %v, want %v", c.text, got, c.want)
				break
			}
		}
	}
}

// TestBertEmbedding checks the encoder produces well-separated, unit-length,
// deterministic embeddings of the right dimension.
func TestBertEmbedding(t *testing.T) {
	if _, err := os.Stat(bertTestModel); err != nil {
		t.Skip("model not present")
	}
	emb := func(s string) []float32 {
		v, err := Embed(EmbedOptions{Model: bertTestModel, Text: s, Normalize: true})
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	a := emb("A man is playing a guitar.")
	again := emb("A man is playing a guitar.")
	similar := emb("A person plays an acoustic guitar.")
	unrelated := emb("The weather forecast predicts heavy rain tomorrow.")

	if len(a) != 384 {
		t.Fatalf("dim = %d, want 384", len(a))
	}
	for i := range a {
		if a[i] != again[i] {
			t.Fatalf("embedding not deterministic at %d", i)
		}
	}
	var ss float64
	for _, x := range a {
		ss += float64(x) * float64(x)
	}
	if math.Abs(math.Sqrt(ss)-1) > 1e-4 {
		t.Errorf("normalized embedding L2 = %.5f, want 1", math.Sqrt(ss))
	}
	sim, dif := bertCos(a, similar), bertCos(a, unrelated)
	t.Logf("cos(paraphrase)=%.4f cos(unrelated)=%.4f", sim, dif)
	if sim <= dif {
		t.Errorf("paraphrase (%.3f) should be closer than unrelated (%.3f)", sim, dif)
	}
	if sim < 0.5 {
		t.Errorf("paraphrase cosine %.3f unexpectedly low", sim)
	}
}

// TestEmbedBatchMatchesSingle verifies the batched forward (packed sequences +
// block-diagonal attention) reproduces the per-text Embed results — a sequence's
// output must not depend on what else shares its batch. Runs against whichever of
// the bert / nomic-bert test models are present (different attention/FFN paths).
func TestEmbedBatchMatchesSingle(t *testing.T) {
	models := []struct{ name, path string }{
		{"bert", bertTestModel},
		{"nomic", "models/nomic.gguf"},
	}
	texts := []string{
		"A man is playing a guitar.",
		"search_document: To bake bread, mix flour, water, yeast and salt.",
		"short",
		"The weather forecast predicts heavy rain across the region tomorrow.",
	}
	ran := false
	for _, mdl := range models {
		if _, err := os.Stat(mdl.path); err != nil {
			continue
		}
		ran = true
		t.Run(mdl.name, func(t *testing.T) {
			batch, err := EmbedBatch(EmbedBatchOptions{Model: mdl.path, Texts: texts, Normalize: true})
			if err != nil {
				t.Fatal(err)
			}
			if len(batch) != len(texts) {
				t.Fatalf("got %d vectors, want %d", len(batch), len(texts))
			}
			for i, txt := range texts {
				single, err := Embed(EmbedOptions{Model: mdl.path, Text: txt, Normalize: true})
				if err != nil {
					t.Fatal(err)
				}
				if len(batch[i]) != len(single) {
					t.Fatalf("text %d: dim %d vs %d", i, len(batch[i]), len(single))
				}
				// Per-row math is identical batched vs single, so this should be
				// essentially exact (tiny float-order slack only).
				if c := bertCos(batch[i], single); c < 0.99999 {
					t.Errorf("text %d: batched vs single cosine %.6f", i, c)
				}
				var maxAbs float64
				for d := range single {
					if a := math.Abs(float64(batch[i][d] - single[d])); a > maxAbs {
						maxAbs = a
					}
				}
				if maxAbs > 1e-4 {
					t.Errorf("text %d: max abs diff %.2e too large", i, maxAbs)
				}
			}
		})
	}
	if !ran {
		t.Skip("no embedding model present")
	}
}

// TestEmbedConcurrentMatchesSerial guards the single-flight worker pool: many
// concurrent embeds must each produce the same vector as a lone call (run with
// -race to catch shared-pool data races).
func TestEmbedConcurrentMatchesSerial(t *testing.T) {
	if _, err := os.Stat(bertTestModel); err != nil {
		t.Skip("model not present")
	}
	text := "A man is playing a guitar in the park."
	want, err := Embed(EmbedOptions{Model: bertTestModel, Text: text, Normalize: true})
	if err != nil {
		t.Fatal(err)
	}
	const G = 16
	results := make([][]float32, G)
	var wg sync.WaitGroup
	for g := 0; g < G; g++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			for r := 0; r < 8; r++ {
				v, e := Embed(EmbedOptions{Model: bertTestModel, Text: text, Normalize: true})
				if e != nil {
					t.Error(e)
					return
				}
				results[idx] = v
			}
		}(g)
	}
	wg.Wait()
	for g := 0; g < G; g++ {
		if c := bertCos(results[g], want); c < 0.99999 {
			t.Errorf("goroutine %d diverged from serial: cosine %.6f", g, c)
		}
	}
}

// TestNomicEmbedding covers the nomic-bert variant (RoPE + fused QKV + gated
// SwiGLU FFN). It checks the architecture is detected and that a query matches a
// relevant document better than an irrelevant one (nomic's retrieval use case).
// (Validated against llama-embedding offline at cosine 0.996.)
func TestNomicEmbedding(t *testing.T) {
	const model = "models/nomic.gguf"
	if _, err := os.Stat(model); err != nil {
		t.Skip("model not present")
	}
	b, err := getOrLoadBert(model)
	if err != nil {
		t.Fatal(err)
	}
	if !b.useRope || !b.ropeNeox || b.Layers[0].Wgate == nil || b.Layers[0].Wqkv == nil || b.actMode != 2 {
		t.Errorf("nomic config: rope=%v neox=%v gated=%v fusedQKV=%v actMode=%d",
			b.useRope, b.ropeNeox, b.Layers[0].Wgate != nil, b.Layers[0].Wqkv != nil, b.actMode)
	}
	emb := func(s string) []float32 {
		v, err := Embed(EmbedOptions{Model: model, Text: s, Normalize: true})
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	q := emb("search_query: how do I bake bread at home")
	rel := emb("search_document: To bake bread, mix flour, water, yeast and salt, knead, let it rise, and bake.")
	irr := emb("search_document: The Roman empire lasted for many centuries across Europe.")
	t.Logf("cos(q,relevant)=%.4f cos(q,irrelevant)=%.4f", bertCos(q, rel), bertCos(q, irr))
	if bertCos(q, rel) <= bertCos(q, irr) {
		t.Errorf("relevant doc should outrank irrelevant: %.3f vs %.3f", bertCos(q, rel), bertCos(q, irr))
	}
}
