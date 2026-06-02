package scriptlingllmlib

import (
	"math"
	"os"
	"testing"
)

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func TestEmbedShapeAndDeterminism(t *testing.T) {
	if _, err := os.Stat(concTestModel); err != nil {
		t.Skip("model not present")
	}
	a, err := Embed(EmbedOptions{Model: concTestModel, Text: "hello world"})
	if err != nil {
		t.Fatal(err)
	}
	if len(a) == 0 {
		t.Fatal("empty embedding")
	}
	// Deterministic: same input -> identical vector.
	b, err := Embed(EmbedOptions{Model: concTestModel, Text: "hello world"})
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != len(b) {
		t.Fatalf("length mismatch %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic at %d: %v vs %v", i, a[i], b[i])
		}
	}
	for _, v := range a {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("embedding contains non-finite value")
		}
	}
}

func TestEmbedNormalize(t *testing.T) {
	if _, err := os.Stat(concTestModel); err != nil {
		t.Skip("model not present")
	}
	v, err := Embed(EmbedOptions{Model: concTestModel, Text: "normalize me", Normalize: true})
	if err != nil {
		t.Fatal(err)
	}
	var ss float64
	for _, x := range v {
		ss += float64(x) * float64(x)
	}
	if math.Abs(math.Sqrt(ss)-1.0) > 1e-4 {
		t.Errorf("normalized vector has L2 norm %.6f, want 1.0", math.Sqrt(ss))
	}
}

// TestEmbedSemantics: a paraphrase should be closer (cosine) to the anchor than
// an unrelated sentence. A broken hidden-state/pooling path would not preserve
// this ordering.
func TestEmbedSemantics(t *testing.T) {
	if _, err := os.Stat(concTestModel); err != nil {
		t.Skip("model not present")
	}
	emb := func(s string) []float32 {
		v, err := Embed(EmbedOptions{Model: concTestModel, Text: s, Normalize: true})
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	anchor := emb("The cat sat on the mat.")
	similar := emb("A cat was sitting on the rug.")
	different := emb("Quarterly revenue exceeded market expectations.")

	simScore := cosine(anchor, similar)
	diffScore := cosine(anchor, different)
	t.Logf("cosine: similar=%.4f different=%.4f", simScore, diffScore)
	if simScore <= diffScore {
		t.Errorf("expected paraphrase closer than unrelated text: similar=%.4f, different=%.4f", simScore, diffScore)
	}
}
