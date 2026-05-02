# scriptling-llm-lib

Go library providing LLM inference primitives for the [Scriptling](https://github.com/paularlott/scriptling) runtime. Registered as the `llm` Scriptling library.

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
| **Library registration** | `llm.go` | Registers 50+ functions as the `llm` Scriptling library |
| **Model loading** | `gguf.go` | GGUF v3 parser — F32, F16, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q6_K |
| **Inference model** | `model.go` | Transformer forward pass, KV cache, autoregressive generation |
| **Tokenizer** | `tokenizer.go` | BPE tokenizer with sentencepiece + GPT-2 byte-fallback |
| **Chat templates** | `chat_template.go` | ChatML/Jinja2 template rendering |
| **Quantized matmul** | `q8_fast.go`, `q4k.go`, `q5_fast.go`, `q6k.go` | Fused quantized dot products (Q4, Q4_1, Q5, Q8, Q4_K, Q6_K) |
| **Quantize/dequantize** | `quantize.go` | Float-to-Q8 conversion, dequant helpers |
| **Fused ops** | `fused.go`, `fused_ops.go`, `fused_block.go` | Linear layers, RMS norm, RoPE, attention, FFN, output logits |
| **Primitives** | `llm_primitives.go` | Vector ops, activations, head splitting/merging, repeat KV |
| **Matrix/vector ops** | `matrix_ops.go`, `vector_ops.go`, `math_ops.go` | Matmul, softmax, dot product, sin/cos/exp |
| **Activations** | `activations.go` | sigmoid, relu, gelu, silu |
| **Sampling** | `sampling.go` | Greedy, temperature, top-k, top-p, repeat penalty |
| **Attention heads** | `heads.go` | Head splitting, merging, repeat KV |
| **Helpers** | `helpers.go` | Float conversion utilities |
| **Concurrency** | `parallel.go` | `parallelFor` — goroutine pool for parallel matmul |

## Supported GGUF Tensor Types

| Type | Block Size | Elements/Block | Used For |
|------|-----------|----------------|----------|
| F32 | 4 | 1 | Norms, biases |
| F16 | 2 | 1 | Token embeddings (dequantized) |
| Q4_0 | 18 | 32 | Weight matrices |
| Q4_1 | 20 | 32 | Weight matrices (with min offset) |
| Q5_0 | 22 | 32 | Weight matrices |
| Q8_0 | 34 | 32 | Weight matrices, token embeddings |
| Q4_K | 144 | 256 | Weight matrices (K-quant) |
| Q6_K | 210 | 256 | Weight matrices (K-quant, higher precision) |

## Function Reference

All functions are available via `import llm` in Scriptling.

### Text Generation

- `llm.generate(model, prompt, max_tokens, strategy, ...)` — End-to-end generation from GGUF files. Supports `temperature`, `top_k`, `top_p`, `repeat_penalty`, `system_prompt`, `template`, `stats` kwargs.

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
- `llm.linear_q4_k(x, raw, blocks_per_row)` — Q4_K quantized linear
- `llm.dequantize_q8(raw, scales, groups)` — Q8_0 dequantization
- `llm.dequantize_q4_0(raw, n_groups)` — Q4_0 dequantization
- `llm.dequantize_q4_k(raw, n_blocks)` — Q4_K dequantization

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

## Performance

Benchmarks on Apple M2 Max, greedy decoding, 40 generated tokens:

| Model | Q8_0 tok/s | Q4_K_M tok/s |
|-------|-----------|--------------|
| SmolLM2 135M | 29.7 | 20.6 |
| SmolLM2 360M | 13.8 | 9.6 |
| SmolLM2 1.7B | 3.4 | — |

## Examples

- `examples/basic/` — Minimal Go program calling LLM primitives
- `examples/sllm/` — Full Scriptling CLI runtime (the `sllm` binary)
- `examples/generate/` — Text generation script for `sllm`
