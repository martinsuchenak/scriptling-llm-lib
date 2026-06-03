package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"sync"
	"sync/atomic"
	"time"

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
	profPath := flag.String("prof", "", "Write a CPU profile to this path (for performance analysis)")
	embed := flag.Bool("embed", false, "Embedding mode: embed -prompt and report throughput (auto-enabled for encoder models)")
	embedConcurrency := flag.Int("embed-concurrency", 1, "Embedding mode: number of concurrent embedders (aggregate emb/s across cores)")

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

	if *profPath != "" {
		f, err := os.Create(*profPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: cannot create profile %q: %v\n", *profPath, err)
			os.Exit(1)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "error: cannot start profile: %v\n", err)
			os.Exit(1)
		}
		defer pprof.StopCPUProfile()
	}

	// Encoder models (BERT / nomic-bert) can't generate — embed and report
	// throughput instead. Auto-detected from the model architecture.
	arch, _ := scriptlingllmlib.ModelArch(*model)
	if *embed || scriptlingllmlib.IsEmbeddingArch(arch) {
		runEmbedBench(*model, *prompt, *embedConcurrency)
		return
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

// runEmbedBench embeds the prompt repeatedly and reports throughput. With
// concurrency > 1 it drives that many goroutines (Embed is goroutine-safe and
// runs on a private clone), reporting aggregate emb/s across cores — the figure
// that matters for batch workloads. Stats are tagged with "prefill"/"decode" so
// the bench harness picks them up; an encoder has no decode loop, so they carry
// per-embed latency (ms) and throughput (emb/s).
func runEmbedBench(model, text string, concurrency int) {
	if concurrency < 1 {
		concurrency = 1
	}
	v, err := scriptlingllmlib.Embed(scriptlingllmlib.EmbedOptions{Model: model, Text: text, Normalize: true})
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	dim := len(v)

	var runs int64
	var wg sync.WaitGroup
	stop := int32(0)
	start := time.Now()
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for atomic.LoadInt32(&stop) == 0 {
				_, _ = scriptlingllmlib.Embed(scriptlingllmlib.EmbedOptions{Model: model, Text: text, Normalize: true})
				atomic.AddInt64(&runs, 1)
			}
		}()
	}
	for time.Since(start) < 1500*time.Millisecond {
		time.Sleep(10 * time.Millisecond)
	}
	atomic.StoreInt32(&stop, 1)
	wg.Wait()

	elapsedMs := float64(time.Since(start).Microseconds()) / 1000
	n := atomic.LoadInt64(&runs)
	embPerSec := float64(n) / (elapsedMs / 1000)
	// Wall latency per embed averaged over the in-flight streams (so a single
	// stream still reports its true per-embed latency).
	latency := elapsedMs / (float64(n) / float64(concurrency))

	fmt.Printf("[%d-dim embedding]\n", dim)
	fmt.Fprintln(os.Stderr, "---")
	fmt.Fprintf(os.Stderr, "embed     %4d dims   %d runs in %6.0f ms   (%d stream(s))\n", dim, n, elapsedMs, concurrency)
	// "prefill"/"decode" tags so the bench harness's parser picks these up; for an
	// encoder they carry latency (ms/embed) and throughput (emb/s).
	fmt.Fprintf(os.Stderr, "prefill   %4d dims   latency    %6.1f ms\n", dim, latency)
	fmt.Fprintf(os.Stderr, "decode    %4d runs   throughput %6.1f emb/s\n", n, embPerSec)
}

func tokensPerSec(n int, ms float64) float64 {
	if ms <= 0 {
		return 0
	}
	return float64(n) / (ms / 1000)
}
