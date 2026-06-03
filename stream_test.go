package scriptlingllmlib

import (
	"context"
	"os"
	"strings"
	"testing"
	"unicode/utf8"
)

// TestGenerateStream: streamed deltas must reconstruct exactly the text that the
// non-streaming call returns, each delta must be valid UTF-8, and at least one
// delta must be emitted.
func TestGenerateStream(t *testing.T) {
	if _, err := os.Stat(concTestModel); err != nil {
		t.Skip("model not present")
	}

	args := func(onToken func(string)) (string, int, int, float64, float64, error) {
		return GenerateWithCacheStream(context.Background(), concTestModel,
			"List three colors", 24, "greedy", 0.8, 50, 0.9, 1.1, 64, "", "", "", onToken)
	}

	// Reference: same call without streaming.
	ref, _, _, _, _, err := args(nil)
	if err != nil {
		t.Fatal(err)
	}

	var b strings.Builder
	calls := 0
	got, _, _, _, _, err := args(func(delta string) {
		calls++
		if !utf8.ValidString(delta) {
			t.Errorf("delta %q is not valid UTF-8", delta)
		}
		b.WriteString(delta)
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls == 0 {
		t.Fatal("onToken was never called")
	}
	if got != ref {
		t.Errorf("streamed run returned %q, want %q", got, ref)
	}
	if strings.TrimSpace(b.String()) != ref {
		t.Errorf("joined deltas = %q, want %q", strings.TrimSpace(b.String()), ref)
	}
}

// TestValidUTF8PrefixLen checks the rune-boundary helper holds back partial
// multi-byte sequences and passes through complete ones.
func TestValidUTF8PrefixLen(t *testing.T) {
	euro := "€" // 3 bytes: e2 82 ac
	cases := []struct {
		s    string
		want int
	}{
		{"", 0},
		{"abc", 3},
		{euro, 3},            // complete
		{euro[:2], 0},        // 2 of 3 bytes -> hold all back
		{euro[:1], 0},        // 1 of 3 bytes
		{"ab" + euro[:2], 2}, // keep "ab", hold the partial euro
		{"ab" + euro, 5},     // all complete
	}
	for _, c := range cases {
		if got := validUTF8PrefixLen(c.s); got != c.want {
			t.Errorf("validUTF8PrefixLen(%q) = %d, want %d", c.s, got, c.want)
		}
	}
}
