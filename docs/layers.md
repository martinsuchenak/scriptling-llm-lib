# layers.py — API Reference

Neural network building blocks for transformer models. Supports two architecture families:

- **GPT-2**: LayerNorm + learned positional embeddings + standard MLP (GELU, bias)
- **Llama**: RMSNorm + RoPE + gated MLP (SwiGLU, no bias)

All operations use only `math` from the standard library and linear algebra primitives from `linalg.py`.

---

## Embedding

Token ID lookup table. Maps integer token IDs to dense vectors by indexing into a weight matrix of shape `(vocab_size, d_model)`.

### Constructor

```python
Embedding(weight)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | `list[list[float]]` | Embedding matrix. Row `i` is the vector for token ID `i`. Shape: `(vocab_size, d_model)`. |

### forward

```python
forward(token_ids) -> list[list[float]]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `token_ids` | `list[int]` | Sequence of integer token IDs. |

Returns a list of vectors, one per token ID, by looking up each ID in the weight table.

### Data Flow

```
token_ids  [3, 1, 7]
    |
    v
weight[3] weight[1] weight[7]    (lookup)
    |        |        |
    v        v        v
  [vec]    [vec]    [vec]         output: (seq_len, d_model)
```

---

## PositionalEmbedding

Learned positional embedding table. Returns position vectors for positions 0 through `seq_len - 1`. Used in GPT-2 models. Llama models use RoPE instead.

### Constructor

```python
PositionalEmbedding(weight)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | `list[list[float]]` | Position embedding matrix. Row `i` is the vector for position `i`. Shape: `(max_seq_len, d_model)`. |

### forward

```python
forward(seq_len) -> list[list[float]]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `seq_len` | `int` | Number of positions to return. |

Returns the first `seq_len` rows of the weight matrix.

### Data Flow

```
seq_len = 4
    |
    v
weight[0] weight[1] weight[2] weight[3]    (slice)
    |       |         |         |
    v       v         v         v
 output: (seq_len, d_model)
```

---

## apply_rope

```python
apply_rope(x, start_pos=0) -> list[list[float]]
```

Applies Rotary Position Embeddings (RoPE) to an input tensor. Encodes position information by rotating pairs of dimensions at angles proportional to position. Used in Llama and Mistral style models.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `list[list[float]]` | Input tensor of shape `(seq_len, d_k)`. Typically one attention head of Q or K vectors. |
| `start_pos` | `int` | Position offset for the first token. Set to the current sequence length during autoregressive generation when using a KV cache. Defaults to `0`. |

### Returns

`list[list[float]]` — Rotated tensor of the same shape as `x`.

### Mathematical Detail

For each pair of dimensions `(2i, 2i+1)` at absolute position `p = start_pos + pos`:

```
freq_i  = 1.0 / (10000 ^ (2i / d))

angle_i = freq_i * p

x'[2i]   = x[2i]   * cos(angle_i) - x[2i+1] * sin(angle_i)
x'[2i+1] = x[2i]   * sin(angle_i) + x[2i+1] * cos(angle_i)
```

Each pair of dimensions is treated as a 2D vector and rotated by an angle `angle_i`. Low-frequency pairs (small `i`) produce large rotations that vary slowly with position, capturing coarse positional structure. High-frequency pairs (large `i`) produce small rotations that vary quickly, capturing fine-grained position. The base `10000` controls the frequency range.

### Data Flow

```
x: (seq_len, d_k)
    |
    v
for each position p:
    for each dimension pair (2i, 2i+1):
        compute freq_i = 1 / 10000^(2i/d)
        compute angle  = freq_i * (start_pos + p)
        rotate (x[2i], x[2i+1]) by angle
    |
    v
output: (seq_len, d_k)
```

---

## attention

```python
attention(q, k, v, causal=False) -> list[list[float]]
```

Scaled dot-product attention.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `list[list[float]]` | Query tensor of shape `(q_len, d_k)`. |
| `k` | `list[list[float]]` | Key tensor of shape `(kv_len, d_k)`. |
| `v` | `list[list[float]]` | Value tensor of shape `(kv_len, d_k)`. |
| `causal` | `bool` | If `True`, apply causal masking so each query position can only attend to key positions at or before it. Defaults to `False`. |

### Returns

`list[list[float]]` — Attention output of shape `(q_len, d_k)`.

### Mathematical Detail

```
scores = Q @ K^T                    shape: (q_len, kv_len)
scaled = scores / sqrt(d_k)         per-element scaling
```

When `causal=True` and `q_len > 1`, positions where the key index `j` exceeds the query index `i` are set to `-inf` before softmax, preventing attention to future tokens.

When `q_len == 1` (single-token generation step with KV cache), no masking is applied regardless of the `causal` flag. A single query attending to all previous keys is inherently causal since there are no future positions to mask.

```
weights = softmax(scaled)           shape: (q_len, kv_len)
output  = weights @ V               shape: (q_len, d_k)
```

### Data Flow

```
Q: (q_len, d_k)    K: (kv_len, d_k)    V: (kv_len, d_k)
    |                    |                    |
    |              transpose(K)              |
    |                    |                    |
    +--- Q @ K^T --------+                    |
    |                                         |
    v                                         |
scores: (q_len, kv_len)                       |
    |                                         |
    / sqrt(d_k)                               |
    |                                         |
[causal mask if q_len > 1]                    |
    |                                         |
    softmax(rows)                             |
    |                                         |
    v                                         |
weights: (q_len, kv_len)                      |
    |                                         |
    +--- weights @ V -------------------------+
    |
    v
output: (q_len, d_k)
```

---

## MultiHeadAttention

Multi-head self-attention with linear Q/K/V projections and optional RoPE.

### Constructor

```python
MultiHeadAttention(n_heads, d_model, w_q, w_k, w_v, w_o, use_rope=False)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_heads` | `int` | Number of attention heads. |
| `d_model` | `int` | Model hidden dimension. Must be divisible by `n_heads`. |
| `w_q` | `list[list[float]]` | Query projection weight. Shape: `(d_model, d_model)`. |
| `w_k` | `list[list[float]]` | Key projection weight. Shape: `(d_model, d_model)`. |
| `w_v` | `list[list[float]]` | Value projection weight. Shape: `(d_model, d_model)`. |
| `w_o` | `list[list[float]]` | Output projection weight. Shape: `(d_model, d_model)`. |
| `use_rope` | `bool` | If `True`, apply rotary position embeddings to Q and K before attention. Used for Llama models. Defaults to `False`. |

All weight matrices are transposed at construction time so that `forward()` can use `matmul_t` for efficient `x @ W^T` projections.

### forward

```python
forward(x, causal=False, start_pos=0, kv_cache=None, use_cache=False) -> list[list[float]]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `list[list[float]]` | Input tensor of shape `(seq_len, d_model)`. |
| `causal` | `bool` | Apply causal masking in attention. Defaults to `False`. |
| `start_pos` | `int` | Position offset for RoPE. Defaults to `0`. |
| `kv_cache` | `KVCache or None` | If provided, prepend cached K/V to the current K/V before attention. When `use_cache` is also `True`, the combined K/V are written back into the cache. |
| `use_cache` | `bool` | If `True` and `kv_cache` is provided, update the cache with the new K/V. Defaults to `False`. |

### KV Cache Integration

During autoregressive generation, each step produces new K and V vectors for the current token. When `kv_cache` is provided:

1. The new K and V are concatenated with the cached history: `k = cached_k + new_k`.
2. The combined K and V are used for attention (the query attends over the full sequence history).
3. If `use_cache=True`, the combined K and V are written back into the cache for the next step.

This avoids recomputing K and V for all previous tokens at every generation step.

### Data Flow

```
x: (seq_len, d_model)
    |
    +-- x @ w_q^T --> Q: (seq_len, d_model)
    +-- x @ w_k^T --> K: (seq_len, d_model)  --[+ kv_cache.k]--> K: (kv_len, d_model)
    +-- x @ w_v^T --> V: (seq_len, d_model)  --[+ kv_cache.v]--> V: (kv_len, d_model)
    |
    v
split_heads:
    Q -> n_heads x (seq_len, d_k)
    K -> n_heads x (kv_len, d_k)
    V -> n_heads x (kv_len, d_k)
    |
    v
[if use_rope]: apply_rope to each head of Q and K
    |
    v
for each head h:
    attention(Q_h, K_h, V_h, causal)
    |
    v
head_outputs: n_heads x (seq_len, d_k)
    |
    v
merge_heads: concatenate along d_k -> (seq_len, d_model)
    |
    v
output projection: x @ w_o^T -> (seq_len, d_model)
```

---

## KVCache

Stores previously computed Key and Value vectors for autoregressive generation. Avoids recomputing K/V for all prior tokens at each generation step.

### Constructor

```python
KVCache()
```

Takes no arguments. Initializes empty storage for `k` and `v`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `k` | `list[list[float]]` | Cached key vectors. Grows by one row per generation step. |
| `v` | `list[list[float]]` | Cached value vectors. Grows by one row per generation step. |

### update

```python
update(new_k, new_v) -> tuple[list[list[float]], list[list[float]]]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `new_k` | `list[list[float]]` | New key vectors to append. |
| `new_v` | `list[list[float]]` | New value vectors to append. |

Appends `new_k` and `new_v` to the existing cache and returns the full cached K and V.

### Data Flow

```
Step 0 (prefill):  new_k=[k0,k1,...,kN]  -> cache.k=[k0,k1,...,kN]
Step 1 (generate): new_k=[kN+1]          -> cache.k=[k0,k1,...,kN,kN+1]
Step 2 (generate): new_k=[kN+2]          -> cache.k=[k0,k1,...,kN,kN+1,kN+2]
...
```

---

## MLP

Standard two-layer feed-forward network with bias terms. Used in GPT-2 models.

### Constructor

```python
MLP(w_up, b_up, w_down, b_down)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `w_up` | `list[list[float]]` | Up-projection weight. Shape: `(d_model, d_ff)`. |
| `b_up` | `list[float]` | Up-projection bias. Shape: `(d_ff,)`. |
| `w_down` | `list[list[float]]` | Down-projection weight. Shape: `(d_ff, d_model)`. |
| `b_down` | `list[float]` | Down-projection bias. Shape: `(d_model,)`. |

Both weight matrices are transposed at construction time for `matmul_t` usage.

### forward

```python
forward(x) -> list[list[float]]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `list[list[float]]` | Input tensor of shape `(seq_len, d_model)`. |

### Data Flow

```
x: (seq_len, d_model)
    |
    v
x @ w_up^T + b_up         (seq_len, d_ff)
    |
    v
GELU activation            (seq_len, d_ff)
    |
    v
x @ w_down^T + b_down      (seq_len, d_model)
    |
    v
output: (seq_len, d_model)
```

---

## GatedMLP

SwiGLU feed-forward network without bias terms. Used in Llama models.

### Constructor

```python
GatedMLP(w_gate, w_up, w_down)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `w_gate` | `list[list[float]]` | Gate projection weight. Shape: `(d_model, d_ff)`. |
| `w_up` | `list[list[float]]` | Up projection weight. Shape: `(d_model, d_ff)`. |
| `w_down` | `list[list[float]]` | Down projection weight. Shape: `(d_ff, d_model)`. |

All weight matrices are transposed at construction time for `matmul_t` usage.

### forward

```python
forward(x) -> list[list[float]]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `list[list[float]]` | Input tensor of shape `(seq_len, d_model)`. |

The gate and up projections are computed in parallel from the same input. The gate path applies the SiLU (Sigmoid Linear Unit) activation function before the element-wise multiply with the up projection.

### Data Flow

```
x: (seq_len, d_model)
    |
    +-- x @ w_gate^T --> SiLU --> gate: (seq_len, d_ff) --+
    |                                                       |
    +-- x @ w_up^T  --> up:   (seq_len, d_ff) ----------+  |
    |                                                      |*
    +------------------------------------------------------+
    |
    v
gate * up: (seq_len, d_ff)
    |
    v
x @ w_down^T: (seq_len, d_model)
    |
    v
output: (seq_len, d_model)
```

---

## TransformerBlock

One transformer layer with pre-normalization architecture. Applies attention followed by a feed-forward network, each preceded by normalization and wrapped with a residual connection.

### Constructor

```python
TransformerBlock(attn_norm_w, attn_norm_b, attn, ffn_norm_w, ffn_norm_b, ffn, norm_type="layernorm")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `attn_norm_w` | `list[float]` | Attention pre-normalization weight. Shape: `(d_model,)`. |
| `attn_norm_b` | `list[float] or None` | Attention pre-normalization bias. Shape: `(d_model,)`. Unused when `norm_type="rmsnorm"`. |
| `attn` | `MultiHeadAttention` | The multi-head attention sublayer. |
| `ffn_norm_w` | `list[float]` | FFN pre-normalization weight. Shape: `(d_model,)`. |
| `ffn_norm_b` | `list[float] or None` | FFN pre-normalization bias. Shape: `(d_model,)`. Unused when `norm_type="rmsnorm"`. |
| `ffn` | `MLP or GatedMLP` | The feed-forward sublayer. |
| `norm_type` | `str` | Normalization type. `"layernorm"` for GPT-2, `"rmsnorm"` for Llama. Defaults to `"layernorm"`. |

### forward

```python
forward(x, causal=False, start_pos=0, kv_cache=None, use_cache=False) -> list[list[float]]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `list[list[float]]` | Input tensor of shape `(seq_len, d_model)`. |
| `causal` | `bool` | Apply causal masking in attention. Defaults to `False`. |
| `start_pos` | `int` | Position offset for RoPE. Defaults to `0`. |
| `kv_cache` | `KVCache or None` | KV cache for autoregressive generation. Passed through to the attention sublayer. Defaults to `None`. |
| `use_cache` | `bool` | Whether to update the KV cache. Defaults to `False`. |

### Data Flow

```
x: (seq_len, d_model)
    |
    v
+--- residual connection --------------------------+
|                                                  |
|   x --> norm(x, attn_norm) --> Attention --------+--> x + attn_out
|                                      ^           |
|                           kv_cache, start_pos    |
|                                                  |
+--- residual connection --------------------------+
|                                                  |
|   x --> norm(x, ffn_norm) --> FFN ---------------+--> x + ffn_out
|                                                  |
+--------------------------------------------------+
    |
    v
output: (seq_len, d_model)
```

The normalization call dispatches based on `norm_type`:

- `"layernorm"`: calls `layer_norm(x, weight, bias)` — standard LayerNorm with learnable scale and shift.
- `"rmsnorm"`: calls `rms_norm(x, weight)` — RMSNorm with learnable scale only (bias parameter ignored).
