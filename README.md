# Scriptling LLM Library

Native Go library providing 50+ LLM inference functions and end-to-end text generation for [Scriptling](https://github.com/paularlott/scriptling). All functions use the Native API for zero-reflection overhead on transformer inference hot paths.

## Installation

```bash
go get github.com/martinsuchenak/scriptling-llm-lib
```

## Quick Start

### End-to-end text generation from a GGUF model:

```python
import llm
result = llm.generate("models/model.gguf", "Once upon a time", 50, "greedy")
print(result)
```

### Or use individual primitives:

```python
import llm
result = llm.argmax([0.1, 0.9, 0.3])
print(result)  # 1
```

## CLI Binary

Build the Scriptling runtime with LLM support:

```bash
go build -o sllm ./cmd/sllm
./sllm examples/generate/run.py models/model.gguf "Once upon a time" 50 greedy stats
```

## Data Conventions

| Concept | Scriptling type | Go equivalent |
|---|---|---|
| Scalar | `INTEGER` or `FLOAT` | `float64` |
| Vector | `list[float]` | `[]float64` |
| Matrix | `list[list[float]]` | `[][]float64` |
| Weight matrix | `(out_features, in_features)` | PyTorch convention |

All functions validate inputs and return descriptive error messages on type mismatches, shape mismatches, or invalid arguments.

---

## Functions

### Inference Helpers

#### `argmax(x)` -> `int`

Return the index of the maximum value in a list.

```python
import llm
llm.argmax([0.1, 0.9, 0.3])  # 1
```

**Errors:** empty list, non-list input.

---

#### `argmin(x)` -> `int`

Return the index of the minimum value in a list.

```python
llm.argmin([3.0, 1.0, 2.0])  # 1
```

**Errors:** empty list, non-list input.

---

#### `topk(x, k)` -> `list[list[int, float]]`

Return the top k `(index, value)` pairs sorted by value descending. k is clamped to the list length.

```python
llm.topk([1, 5, 3, 4, 2], 3)  # [[1, 5.0], [3, 4.0], [2, 3.0]]
```

**Errors:** k <= 0, non-list input.

---

#### `clip(x, lo, hi)` -> `float` or `list[float]`

Clamp values to `[lo, hi]`. Accepts a scalar (returns scalar) or a list (returns list). `lo` must be <= `hi`.

```python
llm.clip([-2.0, 0.5, 3.0], -1.0, 2.0)  # [-1.0, 0.5, 2.0]
llm.clip(5.0, 0.0, 3.0)                  # 3.0
```

**Errors:** lo > hi, non-numeric input.

---

### Activation Functions

All activation functions accept a single scalar and return a float.

#### `sigmoid(x)` -> `float`

Logistic sigmoid: `1 / (1 + exp(-x))`. Returns values in (0, 1).

```python
llm.sigmoid(0)     # 0.5
llm.sigmoid(100)   # ~1.0
llm.sigmoid(-100)  # ~0.0
```

#### `relu(x)` -> `float`

Rectified Linear Unit: `max(0, x)`.

```python
llm.relu(-1)  # 0.0
llm.relu(5)   # 5.0
```

#### `gelu(x)` -> `float`

Gaussian Error Linear Unit: `0.5 * x * (1 + erf(x / sqrt(2)))`. Used in BERT, GPT-2, T5.

```python
llm.gelu(1.0)  # ~0.8413
```

#### `silu(x)` -> `float`

Sigmoid Linear Unit (Swish): `x * sigmoid(x)`. Used in LLaMA, Gemma, Mistral.

```python
llm.silu(2.0)  # ~1.7616
```

---

### Vector Operations

All vector operations require same-length inputs and return a new list.

#### `vec_add(a, b)` -> `list[float]`

Element-wise addition.

```python
llm.vec_add([1, 2, 3], [4, 5, 6])  # [5.0, 7.0, 9.0]
```

#### `vec_sub(a, b)` -> `list[float]`

Element-wise subtraction: `a[i] - b[i]`.

```python
llm.vec_sub([5, 3, 1], [1, 2, 3])  # [4.0, 1.0, -2.0]
```

#### `vec_mul(a, b)` -> `list[float]`

Element-wise multiplication.

```python
llm.vec_mul([2, 3], [4, 5])  # [8.0, 15.0]
```

#### `vec_scale(a, s)` -> `list[float]`

Multiply every element by scalar `s`.

```python
llm.vec_scale([1, 2, 3], 2.0)  # [2.0, 4.0, 6.0]
```

#### `vec_apply(x, fn_name)` -> `list[float]`

Apply a named activation function element-wise. `fn_name` must be one of `"sigmoid"`, `"relu"`, `"gelu"`, `"silu"`. Dispatches directly to the Go implementation with no per-element callback overhead.

```python
llm.vec_apply([-1, 0, 1, 2], "relu")     # [0.0, 0.0, 1.0, 2.0]
llm.vec_apply([-1, 0, 1, 2], "sigmoid")  # [~0.27, 0.5, ~0.73, ~0.88]
```

**Errors:** unknown function name.

---

### LLM Inference Primitives

These are fused operations that eliminate intermediate allocations on the hottest inference paths.

#### `rms_norm(x, weight, eps=1e-5)` -> `list[list[float]]`

RMS normalization. For each row: divide by root-mean-square, then multiply element-wise by weight. Used by LLaMA, Mistral, Phi, Qwen. Called twice per transformer layer.

```python
x = [[0.5, -0.3, 0.8], [0.1, 0.2, -0.4]]
w = [1.0, 1.0, 1.0]
llm.rms_norm(x, w)       # default eps=1e-5
llm.rms_norm(x, w, 1e-6) # custom eps
```

| Parameter | Type | Shape |
|---|---|---|
| `x` | matrix | `(seq_len, dim)` |
| `weight` | vector | `(dim,)` |
| `eps` | scalar | optional, default `1e-5` |

**Returns:** matrix `(seq_len, dim)`.
**Errors:** dimension mismatch, empty input.

---

#### `rope(x, start_pos=0)` -> `list[list[float]]`

Rotary Position Embeddings. Applies position-dependent rotation to dimension pairs `(2i, 2i+1)` using frequency `1 / (10000^(2i/d))`. Standard for LLaMA, Mistral, Qwen, Phi. Called 2x per head per layer (Q and K).

```python
q = [[1.0, 0.0, 0.0, 1.0]]
llm.rope(q)       # position 0
llm.rope(q, 8)    # starting at position 8
```

| Parameter | Type | Shape |
|---|---|---|
| `x` | matrix | `(seq_len, d_k)` — `d_k` must be even |
| `start_pos` | int | optional, default `0` |

**Returns:** matrix `(seq_len, d_k)`.
**Errors:** odd `d_k`.

---

#### `silu_gate(gate, up)` -> `list[list[float]]`

Fused SiLU activation + element-wise multiply (SwiGLU FFN). Computes `silu(gate) * up` element-wise, avoiding an intermediate allocation. The core of the FFN in LLaMA, Mistral, and most modern transformers.

```python
gate = [[1.0, -1.0], [0.0, 2.0]]
up   = [[1.0,  1.0], [1.0, 1.0]]
llm.silu_gate(gate, up)
```

| Parameter | Type | Shape |
|---|---|---|
| `gate` | matrix | `(seq_len, d_ff)` |
| `up` | matrix | `(seq_len, d_ff)` — same shape as `gate` |

**Returns:** matrix `(seq_len, d_ff)`.
**Errors:** shape mismatch.

---

#### `attention(q, k, v, causal=True)` -> `list[list[float]]`

Scaled dot-product attention: `softmax(Q @ K^T / sqrt(d_k)) @ V` with optional causal masking. The single most expensive operation in transformer inference.

When `causal=True` and `q_len > 1`, positions where key index > query index are masked to `-inf`. When `q_len == 1` (generation with KV cache), no masking is applied regardless of the `causal` flag.

```python
q = [[1.0, 0.0]]
k = [[1.0, 0.0], [0.0, 1.0]]
v = [[1.0, 0.0], [0.0, 1.0]]

llm.attention(q, k, v)             # causal=True (default)
llm.attention(q, k, v, False)      # no masking
```

| Parameter | Type | Shape |
|---|---|---|
| `q` | matrix | `(q_len, d_k)` |
| `k` | matrix | `(kv_len, d_k)` |
| `v` | matrix | `(kv_len, d_k)` |
| `causal` | bool | optional, default `True` |

**Returns:** matrix `(q_len, d_k)`.
**Errors:** dimension mismatch, empty inputs, k/v row count mismatch.

---

#### `linear(x, weight, bias=None)` -> `list[list[float]]`

Fused matrix multiply + optional bias add: `x @ weight.T + bias`. Weight is stored as `(out_features, in_features)` — the PyTorch convention. Every transformer layer has 7+ linear calls (Q, K, V, O projections + gate, up, down FFN).

```python
x = [[1.0, 2.0]]
w = [[1.0, 0.0], [0.0, 1.0]]
llm.linear(x, w)               # [[1.0, 2.0]]
llm.linear(x, w, [10.0, 20.0]) # [[11.0, 22.0]]
```

| Parameter | Type | Shape |
|---|---|---|
| `x` | matrix | `(seq_len, in_features)` |
| `weight` | matrix | `(out_features, in_features)` |
| `bias` | vector | `(out_features,)` — optional |

**Returns:** matrix `(seq_len, out_features)`.
**Errors:** dimension mismatch, empty inputs.

---

#### `linear_row(x, weight, bias=None)` -> `list[float]`

Same as `linear()` but computes only the **last row** of the output. Used for the output projection during generation where only the last token's logits matter. Saves `(seq_len - 1) * out_features` multiply-adds.

```python
x = [[1.0, 2.0], [3.0, 4.0]]
w = [[1.0, 0.0], [0.0, 1.0]]
llm.linear_row(x, w)            # [3.0, 4.0]
llm.linear_row(x, w, [1, 1])    # [4.0, 5.0]
```

| Parameter | Type | Shape |
|---|---|---|
| `x` | matrix | `(seq_len, in_features)` |
| `weight` | matrix | `(out_features, in_features)` |
| `bias` | vector | `(out_features,)` — optional |

**Returns:** vector `(out_features,)`.
**Errors:** dimension mismatch, empty inputs.

---

#### `top_k(logits, k)` -> `list[list[int, float]]`

Find the k highest-scoring elements using O(n) partial sort (maintained top-k buffer with binary search insertion). Returns `(index, value)` pairs sorted descending. Used by top-k and top-p sampling.

```python
llm.top_k([0.1, 0.5, 0.3, 0.9, 0.7], 3)
# [[3, 0.9], [4, 0.7], [1, 0.5]]
```

**Errors:** k <= 0, non-list input.

---

#### `dequantize_q8(data, scales, group_size)` -> `list[float]`

Dequantize int8 data using per-group scales: `float = int8 * scale[group_index]`. Compatible with the Q8_0 format used by llama.cpp/GGUF.

```python
llm.dequantize_q8([10, -5, 20, 15], [0.1, 0.2], 2)
# [1.0, -0.5, 4.0, 3.0]
#  group 0: 10*0.1=1.0, -5*0.1=-0.5
#  group 1: 20*0.2=4.0,  15*0.2=3.0
```

| Parameter | Type | Description |
|---|---|---|
| `data` | list[int] | Integers in range [-128, 127] |
| `scales` | list[float] | One scale per group |
| `group_size` | int | Elements per group (typically 64) |

**Returns:** list of dequantized floats.
**Errors:** group_size <= 0, insufficient scales, int8 range violation.

---

### Matrix Utilities

#### `concat_rows(a, b)` -> `list[list[float]]`

Concatenate two matrices along the row axis. Both must have the same number of columns.

```python
llm.concat_rows([[1, 2]], [[3, 4], [5, 6]])
# [[1, 2], [3, 4], [5, 6]]
```

**Errors:** column count mismatch.

#### `slice_rows(x, start, end)` -> `list[list[float]]`

Extract rows `[start, end)` from a matrix. Indices are clamped to valid bounds.

```python
llm.slice_rows([[1,2],[3,4],[5,6],[7,8]], 1, 3)
# [[3, 4], [5, 6]]
```

#### `flatten(x)` -> `list[float]`

Flatten a 2D matrix into a 1D list in row-major order.

```python
llm.flatten([[1, 2], [3, 4]])  # [1.0, 2.0, 3.0, 4.0]
```

---

## Constants

| Name | Type | Value |
|---|---|---|
| `VERSION` | string | `"1.1.0"` |

```python
import llm
print(llm.VERSION)  # "1.1.0"
```

---

## Complete Example

```python
import llm

# Token selection from logits
logits = [0.1, 2.3, 0.5, -0.1, 1.8]
token = llm.argmax(logits)                    # 1
top = llm.top_k(logits, 3)                    # [(1, 2.3), (4, 1.8), (2, 0.5)]

# Activation functions
x = llm.sigmoid(0)                            # 0.5
h = llm.gelu(1.0)                             # ~0.8413

# Vector operations
a = [1.0, 2.0, 3.0]
b = llm.vec_scale(a, 2.0)                     # [2.0, 4.0, 6.0]
c = llm.vec_add(a, [0.1, 0.2, 0.3])          # [1.1, 2.2, 3.3]

# Single transformer layer (simplified)
x = [[0.5, -0.3, 0.8]]
normed = llm.rms_norm(x, weight, 1e-5)
rotated = llm.rope(normed, start_pos=0)
output = llm.attention(q, k, v, causal=True)
hidden = llm.silu_gate(gate_proj, up_proj)
down = llm.linear(hidden, down_weight)
logits = llm.linear_row(down, output_weight)
token = llm.argmax(logits)
```

## Running Examples

```bash
# Go example exercising all functions
go run examples/basic/main.go

# Build CLI and run generation
go build -o sllm ./cmd/sllm
./sllm examples/generate/run.py models/model.gguf "Once upon a time" 50 greedy stats
```

## Running Tests

```bash
go test -v -race ./...
```

## Architecture Notes

### Why Native API?

All 50+ functions use Scriptling's Native API (`func(ctx, kwargs, args...) object.Object`) instead of the Builder API. This eliminates reflection-based type conversion overhead at registration time and avoids any per-call marshaling cost. For hot-path functions like `attention` (called ~270 times per token) and `rope` (~540 times per token), this matters.

### End-to-end Generation

`llm.generate()` loads a GGUF model file and runs complete transformer inference entirely in Go:
- GGUF parsing with support for F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K tensor types
- BPE tokenization (sentencepiece + GPT-2 byte-fallback)
- Full transformer forward pass with KV caching and GQA support
- Sampling with greedy, temperature, top_k, top_p strategies and repeat penalty
- Model caching — loaded models are cached by path

This eliminates all Python→Go call overhead per token, as the entire generation loop runs natively.

### Why not use Scriptling's built-in `math` library?

Scriptling's `math` stdlib already provides `softmax`, `matmul`, `transpose`, `mat_add`, `dot`, `sin`, `cos`, `exp`, `erf`, and `sqrt`. This library does **not** duplicate those. It provides the fused, domain-specific operations that don't exist in the stdlib:
- Fused ops like `rms_norm`, `silu_gate`, `linear` (matmul + transpose + bias in one call)
- LLM-specific ops like `rope`, `attention` (with causal masking and numerical stability)
- Inference-specific ops like `top_k` (O(n) partial sort), `linear_row` (skip unused rows)
- Quantization support via `dequantize_q8`

### Relationship to Scriptling stdlib

This library complements the built-in `math` library. Use both together:

```python
import math
import llm

scores = math.matmul(q, math.transpose(k))  # if you need raw matmul
logits = llm.linear(x, weight)               # fused matmul + bias
probs = math.softmax(logits_flat)             # from stdlib
token = llm.argmax(probs)                     # from this library
```

## Dependencies

- [Scriptling](https://github.com/paularlott/scriptling) >= v0.6.1

## Project Structure

```
├── llm.go              # Package docs, Library registration, HelpText
├── helpers.go          # Type conversion helpers (Scriptling objects <-> Go slices)
├── math_ops.go         # argmax, argmin, topk, clip
├── activations.go      # sigmoid, relu, gelu, silu
├── vector_ops.go       # vec_add, vec_sub, vec_mul, vec_scale, vec_apply
├── llm_primitives.go   # rms_norm, rope, silu_gate, attention, linear, linear_row
├── matrix_ops.go       # concat_rows, slice_rows, flatten, reshape, zeros, arange
├── sampling.go         # sample (greedy, temperature, top_k, top_p), repeat penalty
├── heads.go            # split_heads, merge_heads, repeat_kv
├── quantize.go         # quantize_q8, quantize_q8_rows
├── fused_ops.go        # fused_qkv, fused_ffn, fused_rope_batch, fused_attention
├── fused_block.go      # fused_block (full transformer block in one call)
├── fused.go            # q4kDotBlockFast, output_logits
├── parallel.go         # parallelFor (goroutine work-stealing)
├── q8_fast.go          # Q8_0 quantized dot-product (unsafe SIMD)
├── q5_fast.go          # Q5_0 quantized dot-product
├── q4k.go              # Q4_K quantized matmul
├── q6k.go              # Q6_K quantized matmul
├── gguf.go             # GGUF binary parser (metadata, tensors, dequantization)
├── tokenizer.go        # BPE tokenizer (sentencepiece + GPT-2 byte-fallback)
├── chat_template.go    # Chat template engine (ChatML, Llama, Mistral)
├── model.go            # Transformer model, forward pass, KV cache, generate()
├── cmd/sllm/           # CLI binary (Scriptling runtime with LLM lib)
├── examples/
│   ├── basic/          # Go example exercising all functions
│   ├── cli/            # Go CLI example + demo.py
│   └── generate/       # run.py — end-to-end generation example
├── docs/               # Architecture and API documentation
├── llm_test.go         # Unit tests
└── README.md
```
