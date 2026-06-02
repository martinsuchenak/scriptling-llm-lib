package scriptlingllmlib

import (
	"context"
	"os"
	"testing"
	"time"
)

// TestGenerateCancelledUpfront: a context already cancelled before the call
// returns immediately with no generated tokens and ctx.Err().
func TestGenerateCancelledUpfront(t *testing.T) {
	if _, err := os.Stat(concTestModel); err != nil {
		t.Skip("model not present")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	out, nGen, _, _, _, err := GenerateWithCacheContext(
		ctx, concTestModel, "Tell me a long story about the sea", 256,
		"greedy", 0.8, 50, 0.9, 1.1, 64, "", "", "")
	if err != context.Canceled {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if nGen != 0 {
		t.Errorf("nGen = %d, want 0 (cancelled before any token)", nGen)
	}
	if out != "" {
		t.Errorf("out = %q, want empty", out)
	}
}

// TestGenerateCancelMidway: cancelling during decode stops early and returns the
// partial text with ctx.Err(), generating fewer tokens than requested.
func TestGenerateCancelMidway(t *testing.T) {
	if _, err := os.Stat(concTestModel); err != nil {
		t.Skip("model not present")
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(300 * time.Millisecond); cancel() }()

	const want = 4096
	_, nGen, _, _, _, err := GenerateWithCacheContext(
		ctx, concTestModel, "Count slowly and describe each number in detail", want,
		"greedy", 0.8, 50, 0.9, 1.1, 64, "", "", "")
	if err != context.Canceled {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if nGen >= want {
		t.Errorf("nGen = %d, expected cancellation to stop short of %d", nGen, want)
	}
}
