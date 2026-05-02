# Performance

Optimization notes and benchmarks for sllm. Pure-Python inference is inherently slow compared to native C/GPU implementations. These optimizations aim to make small models practical for interactive use in the Scriptling environment.

## Benchmarks

Tested with stories15M (dim=288, hidden=768, 6 layers, 6 heads, vocab=32000, 24M params), Q8 quantized JSON, generating 40 tokens from "Once upon a time" prompt.

| Environment | Time | Notes |
|---|---|---|
| Python 3 (CPython) | ~5s | Native CPython with float operations |
| Scriptling | ~150s | Go-based interpreter, slower float math |

## Profiling Breakdown (Scriptling, 8-token prompt)

| Component | Time | Notes |
|---|---|---|
| Transformer blocks (6x) | ~12s total | ~2s per block for 8 tokens |
| Output projection | ~2.5s | (1, 288) @ (288, 32000) = 9.2M float ops |
| Final norm | ~0.001s | Trivial compared to matmuls |
| **Total forward pass** | **~14.5s** | Initial prompt processing |

During generation with KV cache, each subsequent token processes in ~3.5s (1 token through 6 blocks + output projection).

## Optimizations Applied

### 1. KV Caching

Without KV cache, each generation step reprocesses all previous tokens:
- Step N processes N tokens through all layers
- Total work proportional to sum(1..N) = N^2/2

With KV cache, each step after the first processes only 1 new token:
- Total work proportional to prompt_len + (N-1)
- For 40 tokens (8 prompt): 47 vs 1100 token-processing steps (~23x reduction)

**Files affected:** `layers.py` (KVCache class, MultiHeadAttention), `model.py` (init_kv_cache), `generate.py` (two-phase generation loop)

### 2. Pre-Transposed Weight Matrices

Every `matmul` call previously transposed B on the fly (O(K*N) extra work per call). All weight matrices are now pre-transposed during model construction:

- Attention Q/K/V/O weights: transposed in `MultiHeadAttention.__init__`
- FFN gate/up/down weights: transposed in `GatedMLP.__init__` / `MLP.__init__`
- Output projection: transposed in `LLMModel.__init__`

The engine uses `matmul_t(a, bt)` which skips the transpose and does direct dot products via `zip`.

**Impact:** ~30% speedup (212s -> 150s for 40 tokens in Scriptling)

### 3. Last-Row-Only Output Projection

During generation, only the last token's logits are needed for sampling. The output projection uses `matmul_last_row_t()` which computes only the last row of the result matrix, avoiding computation for all other positions.

**Impact:** Eliminates (seq_len - 1) * d_model * vocab_size operations per generation step.

### 4. Q8 Quantized Weights

Weights are stored as int8 with per-group float scales, reducing JSON file size by ~75%. Dequantization happens at load time (float arithmetic at inference time remains the same).

**Impact:** Faster loading, less memory. No inference speed change (weights are dequantized to float before use).

## Performance Expectations

| Model Size | Params | Python 3 | Scriptling |
|---|---|---|---|
| Tiny (d=64, 2 layers) | ~100K | Instant | <1s/token |
| Small (d=256, 4 layers) | ~5M | ~0.5s/token | ~5s/token |
| stories15M (d=288, 6 layers) | ~24M | ~0.1s/token | ~3.5s/token |
| stories42M (d=512, 8 layers) | ~42M | ~0.5s/token | ~15s/token |

The bottleneck is pure interpreter float performance. The output projection (d_model x vocab_size = 9.2M operations for stories15M) dominates per-token cost.

## Future Optimization Opportunities

1. **Quantized integer matmul** — Keep weights as int8, compute int8 * int8 dot products without dequantizing. Would reduce float operations but adds quantization overhead.

2. **Vocabulary pruning** — During generation, most of the 32000 logits are never selected. Computing only the top-k logits using partial dot products could reduce output projection cost.

3. **Flash attention** — Process attention in chunks to avoid materializing the full (seq_len x seq_len) attention matrix. More relevant for longer sequences.

4. **Weight sharding** — Split weight matrices across chunks for incremental computation.

5. **Native extensions** — If Scriptling adds native array operations or C extensions, matmul could be dramatically accelerated.
