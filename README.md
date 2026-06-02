# scriptling-llm-lib

Go library providing LLM inference primitives for the [Scriptling](https://github.com/paularlott/scriptling) runtime. Registered as the `llm` Scriptling library.

→ [Performance benchmarks](BENCHMARKS.md) — M2 Max / M5 Max / Intel Xeon (AVX2), SmolLM2 135M–1.7B Q8_0

## Quick Start

```python
import llm
result = llm.generate("model.gguf", "Hello", 40, "greedy", temperature=0.0)
print(result)
```

## Architecture

The library is a single Go package (`scriptlingllmlib`) with these logical groups:

| Area | Files | Description |
|------|-------|-------------|
| **Library registration** | `llm.go` | Registers 55+ functions as the `llm` Scriptling library |
| **Model loading** | `gguf.go`, `kquants.go` | GGUF v3 parser — F32, F16, Q4_0/1, Q5_0/1, Q8_0, and k-quants Q2_K–Q6_K |
| **Inference model** | `model.go` | Transformer forward pass, KV cache, autoregressive generation |
| **Tokenizer** | `tokenizer.go` | BPE tokenizer with sentencepiece + GPT-2 byte-fallback |
| **Chat templates** | `chat_template.go` | ChatML/Jinja2 template rendering |
| **Quantized matmul** | `q8_fast.go`, `q5_fast.go` | Fused quantized dot products (Q4_0, Q4_1, Q5_0, Q8_0) |
| **Quantize/dequantize** | `quantize.go` | Float-to-Q8 conversion, dequant helpers |
| **Fused ops** | `fused.go`, `fused_ops.go`, `fused_block.go` | Linear layers, RMS norm, RoPE, attention, FFN, output logits |
| **Primitives** | `llm_primitives.go` | Vector ops, activations, head splitting/merging, repeat KV |
| **Matrix/vector ops** | `matrix_ops.go`, `vector_ops.go`, `math_ops.go` | Matmul, softmax, dot product, sin/cos/exp |
| **Activations** | `activations.go` | sigmoid, relu, gelu, silu |
| **Sampling** | `sampling.go` | Greedy, temperature, top-k, top-p, repeat penalty |
| **Attention heads** | `heads.go` | Head splitting, merging, repeat KV |
| **Helpers** | `helpers.go` | Float conversion utilities |
| **Concurrency** | `parallel.go` | `parallelFor` — goroutine pool for parallel matmul |
| **Go public API** | `generate_cached.go` | Exported Go functions for embedding this library in other Go programs |

## Supported GGUF Tensor Types

| Type | Block / Super-block | Elements | Path | Used For |
|------|-----------|----------|------|----------|
| F32 | 4 B | 1 | float | Norms, biases |
| F16 | 2 B | 1 | float | Token embeddings |
| Q4_0 | 18 B | 32 | packed kernel | Weight matrices |
| Q4_1 | 20 B | 32 | packed kernel | Weight matrices (with min offset) |
| Q5_0 | 22 B | 32 | packed kernel | Weight matrices |
| Q5_1 | 24 B | 32 | dense float | k-quant fallback rows |
| Q8_0 | 34 B | 32 | packed kernel | Weight matrices, token embeddings |
| Q2_K | 84 B | 256 | dense float | Weight matrices |
| Q3_K | 110 B | 256 | dense float | Weight matrices |
| Q4_K | 144 B | 256 | dense float | Weight matrices |
| Q5_K | 176 B | 256 | dense float | Weight matrices |
| Q6_K | 210 B | 256 | dense float | Weight matrices |

K-quant models (e.g. `Q4_K_M`, `Q5_K_M`, `Q6_K`) load and run correctly. By default their super-blocks are dequantized to dense float32 and use the exact float matmul (correct, but ~4 bytes/weight). A **native packed path** is available opt-in via `SLLM_KQUANT_PACKED=1`, which keeps Q4_K/Q5_K/Q6_K super-blocks packed and dequantizes-and-dots them on the fly — ~4–6× less memory (lets large k-quant models fit on RAM-limited hosts), bit-exact with the dense path. The current packed kernel is scalar, so it is slower than the dense float-SIMD matmul until the packed SIMD (AVX2/NEON) kernels land; once those exist it will become the default. Unsupported quantizations (the IQ-family *i-quants*, e.g. `IQ4_NL`, which some k-quant repacks use for rows whose width isn't a multiple of 256) are rejected with a clear error rather than loading silently corrupted weights. For the fastest path, prefer `_Q8_0` or `_Q4_0` variants from [bartowski's GGUF collection](https://huggingface.co/bartowski).

## Go API

For embedding this library in other Go programs (without the Scriptling runtime), the package exposes a cached, concurrency-safe generation API. Everything uses the same thread-safe global model+session cache that the Scriptling `llm.generate` built-in uses — so models are loaded once and reused across calls.

### `Generate` (recommended)

The ergonomic entry point takes an options struct (only `Model` and `Prompt` are required; zero fields use sensible defaults) and returns a structured result:

```go
func Generate(opts GenerateOptions) (GenerateResult, error)

type GenerateOptions struct {
    Model, Prompt string          // required
    MaxTokens     int             // default 256
    Strategy      string          // StrategyGreedy (default) | StrategyTemperature | StrategyTopK | StrategyTopP
    Temperature   float64         // default 1.0 (non-greedy only)
    TopK          int             // default 40
    TopP          float64         // default 0.95
    RepeatPenalty float64         // default 1.1 (1.0 disables)
    RepeatLastN   int             // default 64
    System        string          // system prompt
    Template      string          // chat template override, e.g. "chatml"
    Session       string          // persist KV cache under this id; "" disables
    Context       context.Context // cancellation; nil => no cancellation
    OnToken       func(delta string) // stream deltas; nil => no streaming
}

type GenerateResult struct {
    Text            string
    GeneratedTokens int
    PromptTokens    int
    PrefillMs       float64
    DecodeMs        float64
}
```

```go
res, err := scriptlingllmlib.Generate(scriptlingllmlib.GenerateOptions{
    Model:   "model.gguf",
    Prompt:  "Hello!",
    Session: "chat1",
    OnToken: func(d string) { fmt.Print(d) }, // optional streaming
})
if err != nil { log.Fatal(err) }
fmt.Printf("\n(%.1f t/s)\n", float64(res.GeneratedTokens)/(res.DecodeMs/1000))
```

`Generate` is safe to call from multiple goroutines at once. The model weights are loaded once and shared read-only; each call runs on its own clone of the mutable inference state (KV cache and scratch buffers), so concurrent requests cannot corrupt one another. Turns of the *same* `Session` are serialized (a session is a single conversation). A single in-flight request fans out across all cores; when several requests are in flight, each runs serially and the cores are shared between them — so throughput scales with load without oversubscribing the CPU. If `Context` is cancelled, the error is `ctx.Err()` and `Text` holds the partial output.

### Positional functions

`GenerateWithCache`, `GenerateWithCacheContext`, and `GenerateWithCacheStream` are lower-level variants with positional parameters; `Generate` is built on top of them and is preferred for new code.

### `GenerateWithCache`

```go
func GenerateWithCache(
    modelPath    string,
    prompt       string,
    maxTokens    int,
    strategy     string,   // "greedy", "temperature", "top_k", "top_p"
    temperature  float64,
    topK         int,
    topP         float64,
    repeatPenalty float64,
    repeatLastN  int,
    systemPrompt string,
    templateName string,
    sessionID    string,   // pass "" to disable session caching
) (text string, generatedTokens int, promptTokens int, prefillMs float64, decodeMs float64, err error)
```

Runs inference with the global model cache. If `sessionID` is non-empty the KV cache is persisted between calls, enabling multi-turn conversations without reprocessing prior context.

**Concurrency:** `GenerateWithCache` is safe to call from multiple goroutines at once. The model weights are loaded once and shared read-only; each call runs on its own clone of the mutable inference state (KV cache and scratch buffers), so concurrent requests cannot corrupt one another. Turns of the *same* `sessionID` are serialized (a session is a single conversation). A single in-flight request fans out across all cores; when several requests are in flight, each runs serially and the cores are shared between them — so throughput scales with load without oversubscribing the CPU.

Return values:
- `text` — generated text
- `generatedTokens` — number of tokens produced
- `promptTokens` — number of tokens in the prompt
- `prefillMs` — time spent processing the prompt (milliseconds)
- `decodeMs` — time spent generating tokens (milliseconds)

### `GenerateWithCacheContext`

```go
func GenerateWithCacheContext(
    ctx context.Context,
    // ...same parameters as GenerateWithCache...
) (text string, generatedTokens int, promptTokens int, prefillMs float64, decodeMs float64, err error)
```

Same as `GenerateWithCache` but cancellable. When `ctx` is cancelled (client disconnect, deadline) the decode loop stops between tokens and returns the **partial text generated so far** along with `ctx.Err()` (`context.Canceled` or `context.DeadlineExceeded`). Prefill is not interruptible, so cancellation takes effect once decoding begins. `GenerateWithCache` is just this function with `context.Background()`.

```go
ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
defer cancel()
text, _, _, _, _, err := scriptlingllmlib.GenerateWithCacheContext(
    ctx, "model.gguf", prompt, 512, "greedy", 1.0, 50, 0.9, 1.15, 64, "", "", sessionID,
)
// err == context.Canceled if the client went away; text holds the partial output.
```

### `GenerateWithCacheStream`

```go
func GenerateWithCacheStream(
    ctx context.Context,
    // ...same parameters as GenerateWithCache...
    onToken func(delta string),
) (text string, generatedTokens int, promptTokens int, prefillMs float64, decodeMs float64, err error)
```

Streams output as it is generated. If `onToken` is non-nil it is called with each decoded text **delta** as tokens are produced; the complete `text` is still returned at the end. Byte-level tokens are buffered internally so every delta is valid UTF-8 (a multi-byte rune split across tokens is held back until complete). `onToken` runs on the calling goroutine inside the decode loop — keep it fast and do not call back into the same model from it. `ctx` cancels exactly as in `GenerateWithCacheContext`.

```go
_, _, _, _, _, err := scriptlingllmlib.GenerateWithCacheStream(
    ctx, "model.gguf", prompt, 512, "greedy", 1.0, 50, 0.9, 1.15, 64, "", "", sessionID,
    func(delta string) {
        fmt.Fprint(w, delta) // e.g. write a server-sent-events chunk
        flusher.Flush()
    },
)
```

### `ClearSessionWithCache`

```go
func ClearSessionWithCache(modelPath string, sessionID string)
```

Evicts the KV cache for the given `(modelPath, sessionID)` pair. Call this when a conversation is finished to free memory.

### `Embed`

```go
func Embed(opts EmbedOptions) ([]float32, error)

type EmbedOptions struct {
    Model, Text string // required
    Pooling     string // PoolingMean (default) | PoolingLast
    Normalize   bool   // L2-normalize the result (recommended for cosine similarity)
}
```

Computes a dense embedding of `Text` from the model's final hidden states (pooled across tokens), returning a `DModel`-length vector. Uses the same concurrency-safe cache as `Generate` and runs on a private clone, so it is safe to call concurrently. Pass `Normalize: true` for unit-length vectors suited to cosine similarity.

```go
a, _ := scriptlingllmlib.Embed(scriptlingllmlib.EmbedOptions{Model: "model.gguf", Text: "the cat sat on the mat", Normalize: true})
b, _ := scriptlingllmlib.Embed(scriptlingllmlib.EmbedOptions{Model: "model.gguf", Text: "a cat was sitting on the rug", Normalize: true})
// cosine(a, b) is high for paraphrases, low for unrelated text.
```

### Example

```go
import scriptlingllmlib "github.com/martinsuchenak/scriptling-llm-lib"

text, nGen, nPrompt, prefillMs, decodeMs, err := scriptlingllmlib.GenerateWithCache(
    "model.gguf", "Hello!",
    100, "greedy", 1.0, 50, 0.9, 1.15, 64,
    "", "", "chat1",
)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("%s\n(%.1f t/s)\n", text, float64(nGen)/(decodeMs/1000))

// Free memory when done
scriptlingllmlib.ClearSessionWithCache("model.gguf", "chat1")
```

## Scriptling Function Reference

All functions below are available via `import llm` in Scriptling scripts.

### Text Generation

- `llm.generate(model, prompt, max_tokens, strategy, ...)` — End-to-end generation from GGUF files. Supports `temperature`, `top_k`, `top_p`, `repeat_penalty`, `system_prompt`, `template`, `stats`, `session` kwargs.
- `llm.embed(model, text, pooling="mean", normalize=false)` — Dense embedding vector (list of floats) from the model's final hidden states.
- `llm.clear_session(model, session_id)` — Clear a cached session's KV cache.

### Linear Algebra

- `llm.linear(x, weight, bias?)` — Matrix multiply + bias
- `llm.linear_row(x, weight, bias?)` — Last-row-only linear (for logits)
- `llm.matmul(a, b)` — Matrix multiplication
- `llm.transpose(m)` — Matrix transpose
- `llm.dot(a, b)` — Vector dot product
- `llm.softmax(x)` — Softmax

### Quantized Operations

- `llm.linear_q8(x, raw, groups_per_row)` — Q8_0 quantized linear
- `llm.linear_q4(x, raw, groups_per_row)` — Q4_0 quantized linear
- `llm.dequantize_q8(raw, scales, groups)` — Q8_0 dequantization
- `llm.dequantize_q4_0(raw, n_groups)` — Q4_0 dequantization

### Transformer Primitives

- `llm.rms_norm(x, weight, eps)` — RMS normalization
- `llm.rope(x, start_pos, ...)` — Rotary position encoding
- `llm.attention(q, k, v, causal)` — Scaled dot-product attention
- `llm.silu_gate(gate, up)` — SiLU gating (SwiGLU)
- `llm.fused_qkv(x, wq, wk, wv)` — Fused QKV projection
- `llm.fused_ffn(x, gate_w, up_w, down_w)` — Fused FFN block
- `llm.split_heads(x, n)`, `llm.merge_heads(heads)` — Head reshape
- `llm.repeat_kv(heads, n)` — Repeat KV for GQA

### Vector Operations

- `llm.vec_add(a, b)`, `llm.vec_sub(a, b)`, `llm.vec_mul(a, b)` — Element-wise ops
- `llm.vec_scale(a, s)` — Scalar multiply
- `llm.vec_apply(x, fn)` — Apply activation ("sigmoid", "relu", "gelu", "silu")

### Activations

- `llm.sigmoid(x)`, `llm.relu(x)`, `llm.gelu(x)`, `llm.silu(x)`

### Utilities

- `llm.argmax(x)`, `llm.argmin(x)` — Index of max/min
- `llm.top_k(x, k)`, `llm.topk(x, k)` — Top-k selection
- `llm.clip(x, min, max)` — Clamp values
- `llm.flatten(m)` — Flatten 2D to 1D
- `llm.concat_rows(a, b)`, `llm.slice_rows(m, start, end)` — Row ops
- `llm.VERSION` — Library version string

## Examples

### Getting models

Download a set of compatible Q8_0 and Q4_0 GGUF models (~5 GB):

```bash
task models:download
```

Additional models (Llama 3.2 1B, Qwen2.5 0.5B, Qwen3 1.7B):

```bash
task models:download:extra
```

Models are saved to `models/`. Remove them with `task models:clean`.

### Building

```bash
task build:examples          # native platform
task build:examples:linux    # linux/amd64 + linux/arm64
task build:examples:darwin   # darwin/amd64 + darwin/arm64
task build:examples:windows  # windows/amd64 + windows/arm64
task build:dist              # all of the above
```

Binaries land in `bin/`.

### `examples/infer/` — standalone inference CLI

No scripting required. Pass the model, prompt and options as flags; response goes to stdout, stats to stderr.

```bash
# Greedy decoding
./bin/infer -model models/SmolLM2-360M-Instruct-Q8_0.gguf \
            -prompt "Explain recursion in one paragraph" -tokens 150

# Nucleus sampling
./bin/infer -model models/SmolLM2-1.7B-Instruct-Q8_0.gguf \
            -prompt "Write a haiku about Go" \
            -strategy top_p -temperature 0.9 -tokens 60

# With a system prompt
./bin/infer -model models/SmolLM2-1.7B-Instruct-Q8_0.gguf \
            -system "Answer in one sentence." \
            -prompt "What is SIMD?" -tokens 80
```

### `examples/sllm/` — Scriptling runtime with `llm` library

Full CLI runtime for running `.py` Scriptling scripts that call `llm.*` functions.

```bash
# Single generation via the bundled run.py script
./bin/sllm examples/generate/run.py \
    models/SmolLM2-360M-Instruct-Q8_0.gguf "Hello" 40 greedy stats

# Interactive multi-turn chat with session caching
./bin/sllm examples/session/chat.py \
    models/SmolLM2-1.7B-Instruct-Q8_0.gguf 100 greedy
```

### `examples/basic/` — LLM primitives demo

Runs a Scriptling script inline demonstrating activation functions, vector ops, quantized matmul, attention, and other primitives. No model file needed.

```bash
go run ./examples/basic/
```

### Testing

```bash
task test              # unit tests
task smoke             # end-to-end generation against all downloaded models
```

Accuracy is guarded by perplexity regression tests (`accuracy_test.go`): they compute the model's perplexity over a fixed passage and assert it stays within a golden band, and that the Q8 model is no less accurate than Q4. These catch silent corruption from changes to the hand-tuned quantized kernels. They run as part of `task test` when the SmolLM2-135M models are present, and skip otherwise.

The GGUF parser treats its input as untrusted — every read is bounds-checked and every length/count is capped against the file size — and is covered by a fuzz target:

```bash
go test -run='^$' -fuzz='^FuzzParseGGUF$' -fuzztime=60s .   # explore for parser crashes
```

CI (`.github/workflows/ci.yml`) runs vet, build, the race-enabled test suite (model-dependent tests skip without models), a short fuzz smoke run, and a cross-compile matrix (darwin/arm64, linux/arm64, linux/amd64, windows/amd64) on every push and pull request.

See [BENCHMARKS.md](BENCHMARKS.md) for measured decode/prefill throughput across platforms, and [bench/](bench/README.md) for `fleet.sh` — a harness that cross-compiles `infer`, pushes it plus the selected models to a fleet of remote hosts, and collects benchmarks/CPU profiles in one command.

## Performance tuning

The library auto-tunes itself to the host CPU at startup; in most cases nothing needs configuring.

**Quantized matmul kernel.** Init micro-benchmarks the available Q8 kernels and uses the fastest for this machine:

- *Float SIMD* (AVX2/F16C/SSE, or NEON on ARM) — int8 weights × float32 activations. Fast on most hardware.
- *Scalar Go* — fallback for hosts where float SIMD is penalized (some VMs force AVX2/F16C down a slow path, where scalar wins by a wide margin).
- *Q8×Q8* — quantizes activations to int8 so the hot loop is pure-integer SIMD (`VPMADDWD`). On hosts where the *float* SIMD path is penalized but the *integer* pipeline is not (e.g. an AMD Ryzen 7 4700U VM), this is ~3.6× faster than scalar at the kernel level and roughly doubles decode throughput. Costs ~1% activation-quantization error (negligible for inference).
- *Q4×Q8* — Q4_0 has no float SIMD kernel, so a fused AVX2 kernel decodes the 4-bit weights in SIMD and dots them against int8 activations in one pass — ~10× the old scalar Q4 path at the kernel level (e.g. ~1.8 → ~6.6 t/s decode on a 135M in the 4700U VM). Q4_0 halves weight bandwidth vs Q8_0, which matters for larger models that are memory-bound.

The selector picks whichever actually wins on the host, so no configuration is needed.

**Parallelism threshold** — `SLLM_PARALLEL_THRESHOLD` (env var, integer). Loops with fewer than this many items run serially instead of being split across worker goroutines. Fork/join has real cost (goroutine wakeup + sync), so on hosts where it is expensive, splitting the small per-token decode matmuls is a net loss. At startup the library measures whether splitting a decode-sized matmul actually beats running it serially and picks `256` (parallelize aggressively — bare metal, Apple silicon) or `8192` (keep small matmuls serial, parallelize only the large prefill/output projections — hosts with expensive fork/join, e.g. some VMs). This is measured directly and is independent of the kernel choice, so a bare-metal box that prefers the int8 kernel still parallelizes aggressively.

```bash
# Parallelize aggressively (fast bare metal with cheap goroutine wakeup)
SLLM_PARALLEL_THRESHOLD=256 ./bin/infer -model model.gguf -prompt "..."

# Force fully serial decode (hosts where fork/join is very expensive)
SLLM_PARALLEL_THRESHOLD=999999999 ./bin/infer -model model.gguf -prompt "..."
```

Worker count follows `runtime.NumCPU()`, capped at 8.

**Memory-mapped loading.** On Linux and macOS the model file is loaded with a read-only `mmap` instead of being read fully into the heap. The weights are still copied into the model's own buffers during build, but the file image itself is file-backed (clean, reclaimable, shareable across processes) rather than a dirty heap allocation — roughly halving peak resident memory during load and avoiding a full-file copy. The mapping is unmapped as soon as the build finishes. Other platforms (e.g. Windows) transparently fall back to a full read.

**Async preemption** — `GODEBUG=asyncpreemptoff=1` (Go runtime flag). Inference spends most of its time in tight compute loops; on hosts where delivering preemption signals is expensive (notably some VMs), Go's async preemption can add meaningful overhead — measured ~16% of decode CPU on an AMD Ryzen 7 4700U VM, worth ~5–10% throughput to disable. This is a global runtime tradeoff (it can delay GC and goroutine scheduling), so it is opt-in via the environment rather than a default:

```bash
GODEBUG=asyncpreemptoff=1 ./bin/infer -model model.gguf -prompt "..."
```

## Sessions

The `session` kwarg enables multi-turn conversations with persistent KV cache. The first call with a given `session` ID processes the full prompt. Subsequent calls with the same ID only process new tokens, reusing the cached attention state for faster responses.

```python
import llm

# First turn — processes full prompt, caches KV state
r1 = llm.generate("model.gguf", "Hello!", session="chat1")

# Second turn — only processes new tokens, reuses cached context
r2 = llm.generate("model.gguf", "Tell me more", session="chat1")

# Clear session when done to free memory
llm.clear_session("model.gguf", "chat1")
```

Multiple sessions can coexist on the same model (e.g. `"chat1"`, `"chat2"`). Each session maintains its own independent KV cache.
