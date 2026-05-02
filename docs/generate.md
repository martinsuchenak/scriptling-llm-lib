# generate.py API Reference

Autoregressive text generation with multiple sampling strategies. Provides greedy decoding, temperature scaling, top-k sampling, and nucleus (top-p) sampling. The main `generate()` function runs a token-by-token generation loop with KV caching.

Uses `random.random()` instead of `random.choices()` for Scriptling compatibility. No type annotations, no external dependencies.

---

## Sampling Utilities

### `weighted_sample(indices, weights)`

Weighted random sample using cumulative distribution. Replaces `random.choices()` which is not available in the Scriptling runtime.

```
weighted_sample(indices, weights) -> int
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `indices` | `list[int]` | Candidate token indices |
| `weights` | `list[float]` | Corresponding probabilities (need not sum to 1.0) |
| **return** | `int` | A single sampled index from `indices` |

Algorithm: generates a uniform random value in `[0, sum(weights))`, then walks the cumulative sum until the threshold is exceeded. Falls back to the last index if no threshold is crossed due to floating-point rounding.

---

### `sample_greedy(logits)`

Greedy decoding. Returns the index of the highest logit (argmax).

```
sample_greedy(logits) -> int
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `logits` | `list[float]` | Raw model output scores (one per token) |
| **return** | `int` | Index of the maximum value |

Deterministic. Always produces the same output for the same input.

---

### `sample_temperature(logits, temperature)`

Temperature sampling. Divides each logit by `temperature`, computes softmax to obtain a probability distribution, then draws a weighted sample.

```
sample_temperature(logits, temperature=1.0) -> int
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `logits` | `list[float]` | Raw model output scores |
| `temperature` | `float` | Scaling factor (default `1.0`) |
| **return** | `int` | Sampled token index |

| Temperature | Effect |
|-------------|--------|
| `< 1.0` | Sharper distribution. Amplifies differences between logits, making high-scoring tokens more dominant. Approaches greedy as temperature approaches 0. |
| `= 1.0` | Unchanged distribution. Softmax is applied to the raw logits. |
| `> 1.0` | Flatter distribution. Reduces differences between logits, increasing randomness. |

---

### `sample_top_k(logits, k, temperature)`

Top-k sampling. Applies temperature scaling, selects the `k` highest-scoring tokens, computes softmax over only those tokens, and draws a weighted sample.

```
sample_top_k(logits, k=50, temperature=1.0) -> int
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `logits` | `list[float]` | Raw model output scores |
| `k` | `int` | Number of top tokens to consider (default `50`) |
| `temperature` | `float` | Temperature scaling factor (default `1.0`) |
| **return** | `int` | Sampled token index (one of the top-k) |

All tokens outside the top-k receive zero probability. Lower `k` values produce more focused output; higher values increase diversity.

---

### `sample_top_p(logits, p, temperature)`

Nucleus (top-p) sampling. Applies temperature scaling, computes softmax over all tokens, sorts by probability in descending order, and accumulates probability mass until the cumulative sum reaches `p`. Samples from this minimal set.

```
sample_top_p(logits, p=0.9, temperature=1.0) -> int
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `logits` | `list[float]` | Raw model output scores |
| `p` | `float` | Cumulative probability threshold (default `0.9`) |
| `temperature` | `float` | Temperature scaling factor (default `1.0`) |
| **return** | `int` | Sampled token index |

| `p` value | Effect |
|-----------|--------|
| `0.1` | Highly restrictive. Only the most likely tokens are considered. |
| `0.9` | Default. Typically selects a small set of plausible tokens. |
| `1.0` | All tokens are included. Equivalent to plain temperature sampling. |

Adaptive by nature: when the model is confident, fewer tokens are needed to reach the threshold; when uncertain, more tokens are included.

---

## Generation Loop

### `generate(model, tokenizer, prompt, max_tokens, strategy, temperature, top_k, top_p)`

Autoregressive text generation with KV caching. Encodes the prompt, runs an initial forward pass over the full context, then generates tokens one at a time.

```
generate(model, tokenizer, prompt, max_tokens=100, strategy="greedy",
         temperature=1.0, top_k=50, top_p=0.9) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `Transformer` | Transformer model instance. Must expose `forward(token_ids, start_pos)` and `init_kv_cache()`. |
| `tokenizer` | `Tokenizer` | Tokenizer instance. Must expose `encode(text)`, `decode(ids)`, and `eos_id`. |
| `prompt` | `str` | Input text to condition generation on |
| `max_tokens` | `int` | Maximum number of tokens to generate (default `100`) |
| `strategy` | `str` | Sampling strategy: `"greedy"`, `"temperature"`, `"top_k"`, or `"top_p"` (default `"greedy"`) |
| `temperature` | `float` | Temperature for non-greedy strategies (default `1.0`) |
| `top_k` | `int` | `k` value for `"top_k"` strategy (default `50`) |
| `top_p` | `float` | `p` value for `"top_p"` strategy (default `0.9`) |
| **return** | `str` | Decoded output text including the original prompt |

**Termination.** The loop stops when the model produces the end-of-sequence token (`tokenizer.eos_id`) or when `max_tokens` have been generated.

**Prompt truncation.** If the encoded prompt exceeds `model.config["max_seq_len"]`, only the last `max_seq_len` tokens are used as context.

---

## Generation Algorithm and KV Caching

### Naive Approach

A naive autoregressive loop recomputes the full forward pass at every step. To generate token `t`, the model processes the entire sequence `[token_0, token_1, ..., token_{t-1}]`. This means step `t` performs work proportional to `t`, and the total work across all steps grows quadratically:

```
Step 0:  forward([p_0, p_1, ..., p_n])        -> token_{n+1}
Step 1:  forward([p_0, p_1, ..., p_n, t_0])    -> token_{n+2}
Step 2:  forward([p_0, p_1, ..., p_n, t_0, t_1]) -> token_{n+3}
...
```

At each step, all prior tokens are re-processed through every transformer layer, even though their representations have not changed.

### KV Caching

KV caching avoids this redundancy. During the initial forward pass on the full prompt, each attention layer stores the key and value projections for every position in fixed-size buffers. On subsequent steps, only the single new token is processed:

```
Step 0:  forward([p_0, p_1, ..., p_n], start_pos=0)  -> caches K, V for all prompt tokens
Step 1:  forward([t_0],             start_pos=n+1)    -> appends new K, V to cache
Step 2:  forward([t_1],             start_pos=n+2)    -> appends new K, V to cache
...
```

Each step computes key and value projections for just one token, then attends over the full cached sequence. Attention sees the same context as the naive approach, but the work per step is constant rather than growing with sequence length.

### Implementation in sllm

1. **Encode** the prompt into token IDs.
2. **Initialize** the KV cache via `model.init_kv_cache()`. This allocates empty buffers in every attention layer.
3. **Prefill**: run `model.forward(context, start_pos=0)` on the full prompt (truncated to `max_seq_len`). This populates the cache with key-value pairs for every prompt position.
4. **Sample** the first generated token from the output logits.
5. **Decode loop**: for each subsequent step, call `model.forward([next_id], start_pos=pos)` with only the new token. The model reads the cached keys and values, computes attention over the full context, and appends the new key-value pair to the cache.
6. **Stop** when an end-of-sequence token is produced or `max_tokens` is reached.
7. **Decode** the full token sequence back to text.

This reduces total computation from O(n^2) forward passes over the full context to one O(n) prefill pass plus O(1) per-token steps, making generation practical even in a pure-Python runtime.

---

## Strategy Selection Guide

| Strategy | When to use |
|----------|-------------|
| `"greedy"` | Deterministic output, factual tasks, reproducibility required |
| `"temperature"` | General-purpose controlled randomness |
| `"top_k"` | Limit vocabulary to a fixed candidate set; good for structured output |
| `"top_p"` | Adaptive truncation; recommended for open-ended text generation |

Strategies can be combined with temperature tuning. For example, `"top_p"` with `temperature=0.7` and `p=0.9` is a common configuration for coherent but varied output.
