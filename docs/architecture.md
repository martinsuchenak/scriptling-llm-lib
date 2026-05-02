# sllm Architecture

## 1. Overview

sllm is a pure-Python LLM inference engine designed to run inside the Scriptling sandboxed Python environment. It implements a complete transformer forward pass and autoregressive text generation loop using only Python built-in types and the standard library -- no numpy, no tensors, no external dependencies.

The project exists to make LLM inference understandable and accessible. Every operation is expressed in plain Python: vectors are `list[float]`, matrices are `list[list[float]]`, and every algorithm is readable without framework abstractions.

**Design constraints imposed by Scriptling:**

- No type annotations
- No external dependencies (only `math`, `re`, `json`, `struct`, `os`, `sys`, `random` from stdlib)
- No walrus operator, no async, no yield
- No `random.choices()` (only `random.random()`)
- No `math.tanh` (built from `exp`)

**Supported models:**

- GPT-2 family (LayerNorm, learned positional embeddings, standard MLP with GELU)
- Llama family (RMSNorm, RoPE, gated MLP with SwiGLU, no bias)

Input formats accepted by the build-time converter: llama2.c `.bin` (v0 and v1) and GGUF (F32, F16, Q4\_0, Q5\_0, Q8\_0).

---

## 2. Data Flow

The inference pipeline transforms a text prompt into generated text through these stages:

```
text
  |
  v
tokenizer.encode(text)
  |
  v
token_ids : list[int]
  |
  v
model.forward(token_ids)
  |   token_embedding.forward(ids)    -> x : (seq_len, d_model)
  |   [+ positional encoding]
  |   for each TransformerBlock:
  |     norm(x) -> attention(normed) -> residual add
  |     norm(x) -> ffn(normed)       -> residual add
  |   final_norm(x)
  |   output projection (logits only for last token)
  |
  v
logits : list[float]                    (vocab_size,)
  |
  v
sample(logits) -> next_token_id : int
  |
  v
tokenizer.decode(token_ids)
  |
  v
text
```

During autoregressive generation, the first forward pass processes the full prompt and caches all K and V vectors. Subsequent steps process only the single new token and attend over the cached context, avoiding recomputation of prior tokens.

The generation loop (`generate.py`) repeats: encode prompt, forward pass, sample, append token, forward pass with KV cache, sample, append, ... until max tokens or end-of-sequence.

---

## 3. Data Representation

All numerical data is stored as plain Python lists:

| Concept | Type | Example |
|---|---|---|
| Vector | `list[float]` | `[0.1, 0.2, 0.3]` |
| Matrix | `list[list[float]]` | `[[1.0, 2.0], [3.0, 4.0]]` |
| Tensor (3D) | `list[list[list[float]]]` | Heads in split attention |

Weight tensors from JSON are either 1D (norms, biases) stored as `list[float]`, or 2D (linear projections, embeddings) stored as `list[list[float]]` after reshaping from a flat array via `reshape_2d`.

There is no broadcasting. Every operation is explicit: bias addition iterates rows, scalar multiplication iterates elements.

---

## 4. Module Map

```
sllm/
  linalg.py      Pure-Python linear algebra primitives
  tokenizer.py   SimpleTokenizer and BPETokenizer
  layers.py      Neural network building blocks
  model.py       Top-level LLMModel forward pass
  generate.py    Autoregressive generation loop and sampling
  weights.py     JSON weight loader, Q8 dequantization, model construction
  convert.py     Build-time converter (llama2.c .bin, GGUF -> JSON)
  run.py         CLI entry point
```

### linalg.py

Core linear algebra operations over plain lists. Every function operates on `list[float]` or `list[list[float]]`.

**Operations:**

| Function | Signature | Description |
|---|---|---|
| `tanh(x)` | `float -> float` | Manual tanh from exp (Scriptling lacks `math.tanh`) |
| `matmul(a, b)` | `(M,K) x (K,N) -> (M,N)` | General matrix-matrix multiply |
| `matmul_t(a, bt)` | `(M,K) x (N,K) -> (M,N)` | Multiply with pre-transposed B for fast dot-product |
| `matmul_last_row_t(a, bt)` | `last row of (1,K) x (N,K) -> (N,)` | Optimized output projection (only last token) |
| `vec_add(a, b)` | `(K,) + (K,) -> (K,)` | Element-wise vector addition |
| `vec_matmul(v, m)` | `(K,) @ (K,N) -> (N,)` | Vector-matrix multiply |
| `matvec_mul(m, v)` | `(M,K) @ (K,) -> (M,)` | Matrix-vector multiply |
| `transpose(a)` | `(M,N) -> (N,M)` | Matrix transpose |
| `add_bias(a, bias)` | `(M,N) + (N,) -> (M,N)` | Add bias to each row |
| `mat_add(a, b)` | `(M,N) + (M,N) -> (M,N)` | Element-wise matrix addition |
| `mat_mul_scalar(a, s)` | `(M,N) * scalar -> (M,N)` | Scalar multiply |
| `apply_relu(a)` | `(M,N) -> (M,N)` | ReLU activation |
| `apply_silu(a)` | `(M,N) -> (M,N)` | SiLU/Swish: x * sigmoid(x) |
| `apply_gelu(a)` | `(M,N) -> (M,N)` | Approximate GELU (tanh version) |
| `softmax(x)` | `(K,) -> (K,)` | Numerically stable softmax |
| `softmax_rows(x)` | `(M,K) -> (M,K)` | Row-wise softmax |
| `softmax_masked(x, mask)` | `(M,K) -> (M,K)` | Softmax with binary mask |
| `layer_norm(x, w, b)` | `(M,K) -> (M,K)` | Layer normalization (GPT-2) |
| `rms_norm(x, w)` | `(M,K) -> (M,K)` | RMS normalization (Llama) |

The `matmul_t` family is performance-critical. Because weight matrices are stored pre-transposed at initialization, the inner loop becomes a series of `_dot` product calls using `zip`, which is faster than indexed access in CPython.

### tokenizer.py

Two tokenizer implementations sharing the same interface (`encode` / `decode`):

**SimpleTokenizer** -- Whitespace-split word-level lookup. Each word is mapped directly to a token ID. Unknown words map to `<unk>`. Used for tiny test models.

**BPETokenizer** -- Byte Pair Encoding subword tokenizer supporting both GPT-2 and Llama/sentencepiece conventions:

- **Sentencepiece mode** (Llama): prepends a space to input, splits into characters, applies BPE merges.
- **GPT-2 mode**: pre-tokenizes with a regex pattern (`'s|'t|'re|...| ?[a-zA-Z]+|...`), then applies BPE to each word.

Both modes detect automatically based on whether the vocabulary contains space-prefixed tokens. Merge rules are stored as `"left|right"` string keys because Scriptling cannot use tuple keys in dicts.

Special tokens by convention: `<pad>` = 0, `<s>` = 1, `</s>` = 2, `<unk>` = 3.

### layers.py

Neural network building blocks assembled into a transformer:

**Embedding** -- Token ID lookup table. `forward([1, 5, 3])` returns `[weight[1], weight[5], weight[3]]`.

**PositionalEmbedding** -- Learned position vectors (GPT-2 only). Returns the first `seq_len` rows.

**apply\_rope(x, start\_pos)** -- Rotary Position Embeddings. Applies rotation to consecutive dimension pairs using frequency `1.0 / (10000^(2i/d))`. The `start_pos` parameter offsets positions for KV-cache generation.

**attention(q, k, v, causal)** -- Scaled dot-product attention: `softmax(Q @ K^T / sqrt(d_k)) @ V`. When `causal=True`, future positions are masked to `-inf`.

**MultiHeadAttention** -- Splits input into `n_heads` heads of dimension `d_k = d_model / n_heads`, computes attention per head, concatenates, and projects with `w_o`. Supports RoPE (applied to Q and K after projection) and KV caching.

**KVCache** -- Stores previously computed K and V vectors. Each generation step appends new K/V rows, so subsequent steps attend over the full context without recomputation.

**MLP** -- Standard 2-layer feed-forward with bias and GELU (GPT-2): `down(GELU(up(x)))`.

**GatedMLP** -- SwiGLU feed-forward without bias (Llama): `down(SiLU(gate(x)) * up(x))`.

**TransformerBlock** -- Pre-norm residual block. Applies normalization, then the sublayer (attention or FFN), then adds the residual. Configurable as LayerNorm or RMSNorm.

### model.py

`LLMModel` orchestrates the full forward pass:

1. Token embedding lookup
2. Positional encoding (learned addition for GPT-2, or RoPE inside attention for Llama)
3. N TransformerBlock layers
4. Final normalization (LayerNorm or RMSNorm)
5. Output projection to vocabulary logits

The `logits_only` flag controls whether only the last token's logits are computed (generation) or all positions (training/evaluation). When `logits_only=True`, `matmul_last_row_t` avoids computing logits for the full sequence.

### generate.py

Autoregressive text generation with four sampling strategies:

| Strategy | Function | Description |
|---|---|---|
| Greedy | `sample_greedy` | Picks the token with the highest logit |
| Temperature | `sample_temperature` | Scales logits by `1/T` before softmax sampling |
| Top-k | `sample_top_k` | Samples from the k highest-scoring tokens |
| Top-p (nucleus) | `sample_top_p` | Samples from the smallest token set whose cumulative probability >= p |

`weighted_sample` implements weighted random choice using cumulative distribution, replacing `random.choices()` which is unavailable in Scriptling.

The `generate` function runs the full loop: encode prompt, initial forward pass with KV cache initialization, then single-token forward passes until max tokens or EOS.

### weights.py

Loads JSON model files and constructs the `LLMModel` object:

- `load_model_file(path)` -- Reads a JSON file containing `config`, `vocab`, and `weights`
- `load_weights` -- Iterates over weight entries, calling `load_weight` for each
- `load_weight` -- Dequantizes Q8 if needed, reshapes flat arrays to 2D
- `dequantize_q8` -- Converts int8 values back to float32 using per-group scales
- `build_model` -- Detects architecture from config keys and assembles layers into `LLMModel`
- `load_tokenizer` -- Creates `SimpleTokenizer` or `BPETokenizer` from vocab data

### convert.py

Build-time tool (runs in standard Python, not Scriptling) that converts binary model files to the JSON format:

**Input formats:**

- **llama2.c `.bin`** (v0 legacy and v1 with header) -- Karpathy's format (stories15M, stories42M, stories110M). Reads tensor data sequentially from the binary layout. Optionally loads a separate `tokenizer.bin` for BPE vocabulary and merge rules.
- **GGUF** (llama.cpp format) -- Parses GGUF header, metadata, and tensor info. Dequantizes F16, Q4\_0, Q5\_0, Q8\_0 tensor types to float32. Extracts tokenizer from embedded metadata (tokens, scores, merges).

**Output:** A single JSON file containing `config`, `vocab`, and `weights`.

### run.py

CLI entry point:

```
python run.py <model.json> "prompt text" [max_tokens] [strategy]
```

Loads the model, prints configuration, runs generation, and prints the result.

---

## 5. Architecture Families

### GPT-2

```
x = token_embedding(ids) + pos_embedding(:seq_len)
for each block:
    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))        # GELU activation, with bias
x = LayerNorm(x)
logits = x @ output_weight
```

**Characteristics:**

- `norm_type = "layernorm"` -- Layer normalization with learnable weight and bias
- `pos_encoding = "learned"` -- Fixed lookup table of position vectors, added to token embeddings
- `mlp_type = "standard"` -- Two linear projections with GELU activation and bias terms
- `bias = true` -- All linear layers include bias vectors
- Attention uses standard absolute positional information from the learned embeddings

### Llama

```
x = token_embedding(ids)                      # no positional embedding
for each block:
    x = x + MultiHeadAttention(RMSNorm(x))    # RoPE applied inside attention
    x = x + GatedMLP(RMSNorm(x))              # SwiGLU activation, no bias
x = RMSNorm(x)
logits = x @ output_weight
```

**Characteristics:**

- `norm_type = "rmsnorm"` -- RMS normalization with learnable weight, no bias, no mean subtraction
- `pos_encoding = "rope"` -- Rotary position embeddings applied to Q and K inside each attention head
- `mlp_type = "gated"` -- Three projections: gate (SiLU), up (linear), element-wise multiply, down projection. No bias.
- `bias = false` -- No bias terms anywhere in the model
- `shared_classifier` -- When true, the output projection reuses the token embedding matrix

---

## 6. JSON Model Format

Models are stored as a single JSON file with three top-level keys:

```json
{
  "config": {
    "vocab_size": 32000,
    "d_model": 288,
    "n_heads": 6,
    "n_kv_heads": 6,
    "n_layers": 6,
    "max_seq_len": 256,
    "d_ff": 768,
    "mlp_type": "gated",
    "bias": false,
    "norm_type": "rmsnorm",
    "pos_encoding": "rope",
    "shared_classifier": true
  },
  "vocab": {
    "vocab": {
      "<pad>": 0,
      "<s>": 1,
      "</s>": 2,
      "<unk>": 3,
      " the": 4
    },
    "special": {
      "<pad>": 0,
      "<s>": 1,
      "</s>": 2,
      "<unk>": 3
    },
    "merges": [["t", "h"], ["th", "e"]],
    "scores": [0.0, 0.0, 0.0, 0.0, -1.2],
    "type": "bpe"
  },
  "weights": {
    "token_embedding.weight": {
      "shape": [32000, 288],
      "data": [0.01, -0.02, ...]
    },
    "blocks.0.attn_norm.weight": {
      "shape": [288],
      "data": [1.0, 1.0, ...]
    },
    "blocks.0.attn.w_q.weight": {
      "shape": [288, 288],
      "encoding": "q8",
      "group_size": 64,
      "data": [12, -5, 73, ...],
      "scales": [0.023, 0.019, ...]
    },
    "blocks.0.ffn.w_gate.weight": {
      "shape": [768, 288],
      "encoding": "q8",
      "group_size": 64,
      "data": [-3, 41, ...],
      "scales": [0.031, ...]
    }
  }
}
```

**Weight naming convention:**

```
token_embedding.weight                 (vocab_size, d_model)
pos_embedding.weight                   (max_seq_len, d_model)     [GPT-2 only]
blocks.{i}.attn_norm.weight            (d_model,)
blocks.{i}.attn_norm.bias              (d_model,)                 [GPT-2 only]
blocks.{i}.attn.w_q.weight             (d_model, d_model)
blocks.{i}.attn.w_k.weight             (d_model, d_model)
blocks.{i}.attn.w_v.weight             (d_model, d_model)
blocks.{i}.attn.w_o.weight             (d_model, d_model)
blocks.{i}.ffn_norm.weight             (d_model,)
blocks.{i}.ffn_norm.bias               (d_model,)                 [GPT-2 only]
blocks.{i}.ffn.w_up.weight             (d_model, d_ff)
blocks.{i}.ffn.w_down.weight           (d_ff, d_model)
blocks.{i}.ffn.b_up.bias               (d_ff,)                    [GPT-2 only]
blocks.{i}.ffn.b_down.bias             (d_model,)                 [GPT-2 only]
blocks.{i}.ffn.w_gate.weight           (d_model, d_ff)            [Llama only]
final_norm.weight                      (d_model,)
final_norm.bias                        (d_model,)                 [GPT-2 only]
output.weight                          (vocab_size, d_model) or (d_model, vocab_size)
```

The `vocab` section contains:
- `vocab` -- Token string to integer ID mapping
- `special` -- Special token names to IDs
- `merges` -- BPE merge rules as `[[left, right], ...]` (optional, for BPE tokenizer)
- `scores` -- Token scores from sentencepiece (optional)
- `type` -- `"simple"` or `"bpe"`

---

## 7. Weight Encoding

### fp32

Raw IEEE 754 float values stored as a JSON array. Exact representation, but large. A 15M-parameter model produces roughly 60 MB of JSON.

```json
{
  "shape": [288, 288],
  "data": [0.0123, -0.0456, 0.0789, ...]
}
```

1D tensors (norms, biases) are stored directly as a flat array with a 1-element shape.

### Q8 (symmetric int8 with per-group scales)

Quantized representation using the same scheme as llama.cpp Q8\_0. Weight values are scaled to the `[-127, 127]` integer range in groups. Each group shares a single float scale factor.

**Quantization process** (`quantize_q8` in `convert.py`):

1. Divide the flat weight array into groups of `group_size` elements (default: 64).
2. For each group, find `abs_max = max(|x|)`.
3. Compute `scale = abs_max / 127.0`.
4. Quantize each value: `q = round(x / scale)`, clamped to `[-127, 127]`.

```json
{
  "shape": [288, 288],
  "encoding": "q8",
  "group_size": 64,
  "data": [12, -5, 73, 0, -127, ...],
  "scales": [0.0234, 0.0189, 0.0312, ...]
}
```

**Dequantization** (`dequantize_q8` in `weights.py`):

```
float_val = int_val * scales[group_index]
```

where `group_index = i // group_size`.

The group size defaults to 64 but adapts downward if the tensor length is not evenly divisible. Norms and biases are always stored as fp32 because quantization noise in normalization parameters significantly degrades output quality.

A 15M-parameter model at Q8 encoding produces roughly 15 MB of JSON -- approximately 75% smaller than fp32.

---

## 8. Weight Transposition

The engine computes matrix-vector products as `matmul(x, W)` or `matmul_t(x, W_t)`, where `x` is the input activation and `W` is a weight matrix. For a linear layer mapping `d_in` to `d_out`, the engine expects `W` shaped `(d_in, d_out)`.

**Source format mismatch:** llama2.c and GGUF store linear layer weights as `(out_features, in_features)` -- the conventional PyTorch layout where each row is one output neuron. The engine needs `(in_features, out_features)` so that `matmul(x, W)` produces the correct output dimension.

**Solution -- two transpositions:**

1. **At conversion time** (`convert.py`): `make_weight` with `transpose=True` transposes linear weight matrices from `(out_features, in_features)` to `(in_features, out_features)` before writing to JSON. This is the storage format in the model file. Embedding matrices are not transposed (they are lookup tables, not multiplied).

2. **At load time** (`weights.py` and `layers.py`): `build_model` loads weights in their stored shape. Each layer class (`MultiHeadAttention`, `MLP`, `GatedMLP`) transposes the weight again in its `__init__` via `linalg.transpose`. This produces `W_t` shaped `(out_features, in_features)`, where each row is a column of the original matrix. The layer then uses `matmul_t(x, W_t)` which computes `x @ W` via fast dot products: `output[j] = dot(x_row, W_t[j])`.

The double transposition means the JSON file stores weights as `(in_features, out_features)`, but at runtime the engine holds them as `(out_features, in_features)` for efficient dot-product access in `matmul_t`.

| Stage | Shape | Purpose |
|---|---|---|
| Source (llama2.c/GGUF) | `(out_features, in_features)` | Conventional layout |
| JSON file (after convert) | `(in_features, out_features)` | Engine matmul convention |
| Runtime (after layer init) | `(out_features, in_features)` | Pre-transposed for `matmul_t` dot products |
