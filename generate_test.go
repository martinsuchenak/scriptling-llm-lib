package scriptlingllmlib

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestWithDefaults(t *testing.T) {
	o := GenerateOptions{Model: "m", Prompt: "p"}.withDefaults()
	if o.Context == nil {
		t.Error("Context should default to non-nil")
	}
	if o.Strategy != StrategyGreedy {
		t.Errorf("Strategy = %q, want %q", o.Strategy, StrategyGreedy)
	}
	if o.MaxTokens != 256 {
		t.Errorf("MaxTokens = %d, want 256", o.MaxTokens)
	}
	if o.Temperature != 1.0 || o.TopK != 40 || o.TopP != 0.95 {
		t.Errorf("sampling defaults wrong: temp=%v topK=%d topP=%v", o.Temperature, o.TopK, o.TopP)
	}
	if o.RepeatPenalty != 1.1 || o.RepeatLastN != 64 {
		t.Errorf("repeat defaults wrong: penalty=%v lastN=%d", o.RepeatPenalty, o.RepeatLastN)
	}

	// Explicit values are preserved.
	o2 := GenerateOptions{Model: "m", Prompt: "p", MaxTokens: 8, Strategy: StrategyTopP, TopP: 0.5, RepeatPenalty: 1.0}.withDefaults()
	if o2.MaxTokens != 8 || o2.Strategy != StrategyTopP || o2.TopP != 0.5 || o2.RepeatPenalty != 1.0 {
		t.Errorf("explicit values not preserved: %+v", o2)
	}
}

// TestGenerateOptionsParity: the options API must match the positional API for an
// equivalent greedy request.
func TestGenerateOptionsParity(t *testing.T) {
	if _, err := os.Stat(concTestModel); err != nil {
		t.Skip("model not present")
	}

	ref, refGen, refPrompt, _, _, err := GenerateWithCacheStream(
		context.Background(), concTestModel, "Name a planet", 16,
		StrategyGreedy, 1.0, 40, 0.95, 1.1, 64, "", "", "", nil)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Generate(GenerateOptions{Model: concTestModel, Prompt: "Name a planet", MaxTokens: 16})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != ref {
		t.Errorf("Text = %q, want %q", res.Text, ref)
	}
	if res.GeneratedTokens != refGen || res.PromptTokens != refPrompt {
		t.Errorf("counts = (%d,%d), want (%d,%d)", res.GeneratedTokens, res.PromptTokens, refGen, refPrompt)
	}

	// OnToken via options should reconstruct the text.
	var b strings.Builder
	res2, err := Generate(GenerateOptions{
		Model: concTestModel, Prompt: "Name a planet", MaxTokens: 16,
		OnToken: func(d string) { b.WriteString(d) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(b.String()) != res2.Text {
		t.Errorf("streamed deltas %q != result %q", strings.TrimSpace(b.String()), res2.Text)
	}
}
