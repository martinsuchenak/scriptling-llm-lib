package main

import (
	"flag"
	"fmt"
	"os"

	scriptlingllmlib "github.com/martinsuchenak/scriptling-llm-lib"
)

func main() {
	model := flag.String("model", "", "Path to GGUF model file (required)")
	prompt := flag.String("prompt", "", "Input prompt (required)")
	system := flag.String("system", "", "System prompt (optional)")
	tokens := flag.Int("tokens", 200, "Maximum tokens to generate")
	strategy := flag.String("strategy", "greedy", "Sampling strategy: greedy, temperature, top_k, top_p")
	temperature := flag.Float64("temperature", 0.8, "Sampling temperature (temperature / top_k / top_p strategies)")
	topK := flag.Int("top-k", 50, "Top-K candidates (top_k strategy)")
	topP := flag.Float64("top-p", 0.9, "Nucleus probability threshold (top_p strategy)")
	repeatPenalty := flag.Float64("repeat-penalty", 1.1, "Repetition penalty — 1.0 disables it")
	repeatLastN := flag.Int("repeat-last-n", 64, "Token window considered for repeat penalty")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s -model <path> -prompt <text> [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Run inference on a GGUF model and print the response to stdout.\n")
		fmt.Fprintf(os.Stderr, "Performance stats are written to stderr.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nSampling strategies:\n")
		fmt.Fprintf(os.Stderr, "  greedy       pick the highest-probability token every step\n")
		fmt.Fprintf(os.Stderr, "  temperature  sample with temperature scaling\n")
		fmt.Fprintf(os.Stderr, "  top_k        sample from the top-K most likely tokens\n")
		fmt.Fprintf(os.Stderr, "  top_p        nucleus sampling up to cumulative probability P\n")
	}
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "error: -model is required")
		flag.Usage()
		os.Exit(1)
	}
	if *prompt == "" {
		fmt.Fprintln(os.Stderr, "error: -prompt is required")
		flag.Usage()
		os.Exit(1)
	}

	result, nGen, nPrompt, prefillMs, decodeMs, err := scriptlingllmlib.GenerateWithCache(
		*model,
		*prompt,
		*tokens,
		*strategy,
		*temperature,
		*topK,
		*topP,
		*repeatPenalty,
		*repeatLastN,
		*system,
		"", // templateName: empty = auto-detect from model metadata
		"", // sessionID: empty = no KV-cache session
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(result)

	prefillTPS := tokensPerSec(nPrompt, prefillMs)
	decodeTPS := tokensPerSec(nGen, decodeMs)

	fmt.Fprintln(os.Stderr, "---")
	fmt.Fprintf(os.Stderr, "prompt    %4d tokens   prefill %6.0f ms   %6.1f t/s\n",
		nPrompt, prefillMs, prefillTPS)
	fmt.Fprintf(os.Stderr, "generated %4d tokens   decode  %6.0f ms   %6.1f t/s\n",
		nGen, decodeMs, decodeTPS)
}

func tokensPerSec(n int, ms float64) float64 {
	if ms <= 0 {
		return 0
	}
	return float64(n) / (ms / 1000)
}
