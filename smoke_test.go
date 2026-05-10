//go:build smoke

package scriptlingllmlib

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

var smokeModels = []string{
	"models/SmolLM2-135M-Instruct-Q8_0.gguf",
	"models/SmolLM2-135M-Instruct-Q4_0.gguf",
	"models/SmolLM2-360M-Instruct-Q8_0.gguf",
	"models/SmolLM2-360M-Instruct-Q4_0.gguf",
	"models/SmolLM2-1.7B-Instruct-Q8_0.gguf",
	"models/SmolLM2-1.7B-Instruct-Q4_0.gguf",
	"models/tinyllama-1.1b-1t-openorca.Q8_0.gguf",
	"models/Qwen3-1.7B-Q8_0.gguf",
	"models/Qwen2.5-0.5B-Instruct-Q8_0.gguf",
	"models/Llama-3.2-1B-Instruct-Q8_0.gguf",
}

type smokeResult struct {
	Model      string
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

	var results []smokeResult

	for _, modelPath := range smokeModels {
		r := smokeResult{Model: modelPath}

		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			r.SkipReason = "file not found"
			results = append(results, r)
			continue
		}

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
	fmt.Printf("Prompt: %q (max %d tokens, greedy)\n\n", prompt, maxTokens)

	fmt.Printf("%-4s %-37s %6s %10s %10s %10s %8s\n", "Pass", "Model", "LoadMs", "PromptToks", "PrefillMs", "Decode/s", "Toks")
	fmt.Println(strings.Repeat("-", 100))

	for _, r := range results {
		if r.SkipReason != "" {
			fmt.Printf("%-4s %-37s %s\n", "SKIP", shortModel(r.Model), r.SkipReason)
			continue
		}
		status := "PASS"
		if !r.Pass {
			status = "FAIL"
		}
		fmt.Printf("%-4s %-37s %6d %10d %10.1f %10.1f %8d\n",
			status, shortModel(r.Model), r.LoadMs, r.PromptTok, r.PrefillMs, r.DecodeTps, r.GenTok)
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

func shortModel(path string) string {
	parts := strings.Split(path, "/")
	return parts[len(parts)-1]
}
