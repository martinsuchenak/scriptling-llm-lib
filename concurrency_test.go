package scriptlingllmlib

import (
	"os"
	"sync"
	"testing"
)

const concTestModel = "models/SmolLM2-135M-Instruct-Q8_0.gguf"

func concGen(t *testing.T, prompt, session string) string {
	t.Helper()
	out, _, _, _, _, err := GenerateWithCache(concTestModel, prompt, 12, "greedy", 0.8, 50, 0.9, 1.1, 64, "", "", session)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// TestConcurrentGenerate exercises the concurrency contract: many simultaneous
// generations must not corrupt one another. Greedy decoding is deterministic, so
// every concurrent run of the same prompt must equal the single-threaded
// reference — if the per-request scratch/KV state leaked between requests, the
// outputs would diverge. Run with -race to also catch data races.
func TestConcurrentGenerate(t *testing.T) {
	if _, err := os.Stat(concTestModel); err != nil {
		t.Skip("model not present")
	}

	ref := concGen(t, "The capital of France is", "")

	const n = 8
	var wg sync.WaitGroup
	got := make([]string, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) { defer wg.Done(); got[i] = concGen(t, "The capital of France is", "") }(i)
	}
	wg.Wait()
	for i := range got {
		if got[i] != ref {
			t.Errorf("concurrent run %d diverged:\n got %q\nwant %q", i, got[i], ref)
		}
	}

	// Distinct prompts and sessions concurrently must each stay coherent.
	prompts := []string{"Hello there", "What is two plus two?", "Tell me about dogs", "The sky is"}
	wg = sync.WaitGroup{}
	for _, p := range prompts {
		wg.Add(1)
		go func(p string) { defer wg.Done(); _ = concGen(t, p, "session-"+p) }(p)
	}
	wg.Wait()
}
