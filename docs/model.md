# model.py API Reference

## LLMModel

Top-level class that orchestrates the full transformer forward pass. Supports both GPT-2 (learned positional embeddings, LayerNorm) and Llama (RoPE, RMSNorm) architecture families.

---

### `__init__(config, token_embedding, pos_embedding, blocks, final_norm_w, final_norm_b, output_weight)`

Constructs the model and pre-transposes the output weight matrix for fast inference.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `config` | `dict` | Model configuration. Keys read at runtime: `pos_encoding` (`"learned"` or `"rope"`), `norm_type` (`"layernorm"` or `"rmsnorm"`) |
| `token_embedding` | `Embedding` | Token ID lookup table, shape `(vocab_size, d_model)` |
| `pos_embedding` | `PositionalEmbedding` or `None` | Learned position vectors (GPT-2). `None` for Llama (uses RoPE instead) |
| `blocks` | `list[TransformerBlock]` | Transformer block layers |
| `final_norm_w` | `list[float]` | Final normalization weight, shape `(d_model,)` |
| `final_norm_b` | `list[float]` or `None` | Final normalization bias (LayerNorm only). Unused for RMSNorm |
| `output_weight` | `list[list[float]]` | Output projection matrix, shape `(vocab_size, d_model)` |

**Initialization details:**

- Calls `transpose(output_weight)` to produce `self.output_weight_t` shaped `(d_model, vocab_size)`. This enables `matmul_t` / `matmul_last_row_t` to compute logits via fast dot products.
- Sets `self.kv_caches` to `None`. Call `init_kv_cache()` before generation to enable KV caching.

---

### `init_kv_cache()`

Creates one `KVCache` instance per transformer block. Must be called before autoregressive generation. Each cache stores previously computed K and V vectors so that subsequent forward passes attend over the full context without recomputing prior tokens.

**Side effects:**

- Sets `self.kv_caches` to a `list[KVCache]` of length `len(self.blocks)`.

---

### `forward(token_ids, start_pos=0, logits_only=True)`

Runs the full forward pass: embedding lookup, positional encoding, transformer blocks, final normalization, and output projection.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `token_ids` | `list[int]` | required | Sequence of token IDs to process |
| `start_pos` | `int` | `0` | Position offset for KV cache generation. During prompt processing this is `0`. During single-token generation steps this is the current sequence length |
| `logits_only` | `bool` | `True` | If `True`, return only the last token's logits as a vector `(vocab_size,)`. If `False`, return all positions' logits as a matrix `(seq_len, vocab_size)` |

**Returns:**

| `logits_only` | Return type | Shape | Use case |
|---|---|---|---|
| `True` | `list[float]` | `(vocab_size,)` | Autoregressive generation. Only the last token's logits are needed for sampling the next token |
| `False` | `list[list[float]]` | `(seq_len, vocab_size)` | Testing and debugging. Logits for every position in the sequence |

**Behavior when `kv_caches` is `None`:**

Each transformer block receives `kv_cache=None`, so no key/value vectors are stored. This is the default state after construction.

**Behavior when `kv_caches` is initialized:**

Each transformer block receives its corresponding `KVCache` object. The block appends new K/V rows to the cache and attends over the full accumulated context. On the initial prompt pass (`start_pos=0`), all positions are cached. On subsequent single-token steps, only the new token's K/V is appended.

---

## Forward Pass Data Flow

```
token_ids                          list[int]           (seq_len,)

    |
    v
[1. Token Embedding]
    token_embedding.forward(token_ids)
    |
    v
x                                  list[list[float]]   (seq_len, d_model)

    |
    v
[2. Positional Encoding]

    GPT-2 ("learned"):
        pos_embedding.forward(seq_len) -> pos
        x = mat_add(x, pos)                                (seq_len, d_model)

    Llama ("rope"):
        (positional info applied inside each attention head)

    |
    v
x                                  list[list[float]]   (seq_len, d_model)

    |
    v
[3. Transformer Blocks]           x N blocks
    for each block:
        cache = kv_caches[i] if kv_caches is not None else None
        x = block.forward(x, causal=True, start_pos=start_pos,
                          kv_cache=cache, use_cache=use_cache)
        |
        v
    x                              list[list[float]]   (seq_len, d_model)

    |
    v
[4. Final Normalization]

    GPT-2 ("layernorm"):
        x = layer_norm(x, final_norm_w, final_norm_b)       (seq_len, d_model)

    Llama ("rmsnorm"):
        x = rms_norm(x, final_norm_w)                        (seq_len, d_model)

    |
    v
x                                  list[list[float]]   (seq_len, d_model)

    |
    v
[5. Output Projection]

    logits_only=True (generation):
        logits = matmul_last_row_t(x, output_weight_t)       (vocab_size,)
        (computes only the last row of x @ output_weight_t)

    logits_only=False (debug):
        logits = matmul_t(x, output_weight_t)                (seq_len, vocab_size)
        (computes all positions)

    |
    v
logits                             list[float]          (vocab_size,)
       or                          list[list[float]]    (seq_len, vocab_size)
```

---

## Configuration Reference

The `config` dict is read at forward-pass time to select behavior. Relevant keys:

| Key | Values | Default | Description |
|---|---|---|---|
| `pos_encoding` | `"learned"`, `"rope"` | `"learned"` | Positional encoding strategy. `"learned"` adds position vectors in step 2. `"rope"` applies rotary embeddings inside each attention layer |
| `norm_type` | `"layernorm"`, `"rmsnorm"` | `"layernorm"` | Final normalization. `"layernorm"` uses weight and bias. `"rmsnorm"` uses weight only |

All other configuration values (e.g. `d_model`, `n_heads`, `n_layers`) are consumed at model construction time by `weights.py` and are not read by `LLMModel` directly.
