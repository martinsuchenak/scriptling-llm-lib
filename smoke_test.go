//go:build smoke

package scriptlingllmlib

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// smokeModelsDir is the directory scanned for *.gguf models. Override with
// SMOKE_MODELS_DIR (mirrors the fleet's FLEET_MODELS_SRC).
func smokeModelsDir() string {
	if d := os.Getenv("SMOKE_MODELS_DIR"); d != "" {
		return d
	}
	return "models"
}

// discoverSmokeModels returns every *.gguf under the models dir, sorted, so the
// smoke test covers whatever is actually present instead of a hardcoded list.
func discoverSmokeModels() []string {
	matches, _ := filepath.Glob(filepath.Join(smokeModelsDir(), "*.gguf"))
	sort.Strings(matches)
	return matches
}

type smokeResult struct {
	Model      string
	Kind       string // "gen" or "embed"
	LoadMs     int64
	PromptTok  int
	GenTok     int
	PrefillMs  float64
	DecodeMs   float64
	DecodeTps  float64
	Output     string
	Pass       bool
	SkipReason string
}

func TestSmokeAllModels(t *testing.T) {
	const prompt = "What is the capital of France?"
	const maxTokens = 20

	models := discoverSmokeModels()
	if len(models) == 0 {
		t.Skipf("no .gguf models found in %s (set SMOKE_MODELS_DIR to override)", smokeModelsDir())
	}

	var results []smokeResult

	for _, modelPath := range models {
		r := smokeResult{Model: modelPath}

		// Encoder embedding models can't generate — smoke-test them via Embed.
		arch, archErr := ModelArch(modelPath)
		if archErr == nil && IsEmbeddingArch(arch) {
			r.Kind = "embed"
			t.Run(modelPath, func(t *testing.T) {
				smokeEmbed(t, prompt, &r)
				results = append(results, r)
			})
			continue
		}

		r.Kind = "gen"
		t.Run(modelPath, func(t *testing.T) {
			t0 := time.Now()
			gguf, err := LoadGGUF(modelPath)
			if err != nil {
				r.SkipReason = fmt.Sprintf("LoadGGUF: %v", err)
				results = append(results, r)
				t.Skip(r.SkipReason)
				return
			}

			gguf.Metadata["_path"] = modelPath
			model, err := buildInferenceModelF32(gguf, modelPath)
			gguf.ReleaseFileData()
			if err != nil {
				r.SkipReason = fmt.Sprintf("build: %v", err)
				results = append(results, r)
				t.Skip(r.SkipReason)
				return
			}
			r.LoadMs = time.Since(t0).Milliseconds()

			output, nGen, nPrompt, _ := model.Generate(
				prompt, maxTokens, "greedy",
				0, 0, 0,
				1.15, 64,
				"", "", 0,
			)

			r.PromptTok = nPrompt
			r.GenTok = nGen
			r.PrefillMs = model.PrefillMs
			r.DecodeMs = model.DecodeMs
			if model.DecodeMs > 0 {
				r.DecodeTps = float64(nGen) / (model.DecodeMs / 1000.0)
			}
			r.Output = output

			lower := strings.ToLower(output)
			smallModel := strings.Contains(modelPath, "0.5B")
			qwen3 := strings.Contains(modelPath, "Qwen3")
			r.Pass = strings.Contains(lower, "paris") || (nGen > 0 && (smallModel || qwen3))
			if !r.Pass {
				t.Errorf("expected output to mention 'paris', got: %q", output)
			}

			results = append(results, r)
		})
	}

	fmt.Println()
	fmt.Println("=== SMOKE TEST RESULTS ===")
	fmt.Printf("Scanned: %s (%d models)\n", smokeModelsDir(), len(models))
	fmt.Printf("Prompt: %q (max %d tokens, greedy)\n\n", prompt, maxTokens)

	fmt.Printf("%-4s %-37s %-5s %6s %10s %10s %10s %8s\n", "Pass", "Model", "Kind", "LoadMs", "PromptToks", "PrefillMs", "Decode/s", "Toks")
	fmt.Println(strings.Repeat("-", 110))

	for _, r := range results {
		if r.SkipReason != "" {
			fmt.Printf("%-4s %-37s %-5s %s\n", "SKIP", shortModel(r.Model), r.Kind, r.SkipReason)
			continue
		}
		status := "PASS"
		if !r.Pass {
			status = "FAIL"
		}
		fmt.Printf("%-4s %-37s %-5s %6d %10d %10.1f %10.1f %8d\n",
			status, shortModel(r.Model), r.Kind, r.LoadMs, r.PromptTok, r.PrefillMs, r.DecodeTps, r.GenTok)
	}

	fmt.Println()
	fmt.Println("Outputs:")
	for _, r := range results {
		if r.SkipReason != "" {
			continue
		}
		status := "PASS"
		if !r.Pass {
			status = "FAIL"
		}
		out := r.Output
		if len(out) > 80 {
			out = out[:80] + "..."
		}
		out = strings.ReplaceAll(out, "\n", "\\n")
		fmt.Printf("  %-4s %-25s %6.1f tok/s  %q\n", status, shortModel(r.Model), r.DecodeTps, out)
	}
	fmt.Println()
}

// smokeEmbed runs an embedding model and passes when it returns a non-empty,
// finite, deterministic vector.
func smokeEmbed(t *testing.T, text string, r *smokeResult) {
	t0 := time.Now()
	v, err := Embed(EmbedOptions{Model: r.Model, Text: text, Normalize: true})
	if err != nil {
		r.SkipReason = fmt.Sprintf("embed: %v", err)
		t.Skip(r.SkipReason)
		return
	}
	r.LoadMs = time.Since(t0).Milliseconds()
	r.GenTok = len(v) // report the embedding dimension in the Toks column

	finite := len(v) > 0
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			finite = false
			break
		}
	}
	again, err2 := Embed(EmbedOptions{Model: r.Model, Text: text, Normalize: true})
	deterministic := err2 == nil && len(again) == len(v)
	for i := range v {
		if i < len(again) && again[i] != v[i] {
			deterministic = false
			break
		}
	}
	r.Pass = finite && deterministic
	r.Output = fmt.Sprintf("[%d-dim embedding]", len(v))
	if !r.Pass {
		t.Errorf("embedding smoke failed: finite=%v deterministic=%v dim=%d", finite, deterministic, len(v))
	}
}

func shortModel(path string) string {
	parts := strings.Split(path, "/")
	return parts[len(parts)-1]
}
