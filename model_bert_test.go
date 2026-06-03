package scriptlingllmlib

import (
	"math"
	"os"
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
