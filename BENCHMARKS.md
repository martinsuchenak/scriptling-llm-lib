# Performance Benchmarks

Measured with the `bench/fleet.sh` harness — greedy decoding, a fixed ~30-token
prompt, 120 generated tokens, one `infer` process per host:

```bash
MODELS=all ./bench/fleet.sh bench all
```

**Decode t/s** is the primary metric (sustained autoregressive generation).
**Prefill t/s** is the prompt-processing pass. All models are SmolLM2, run on
bare-metal hosts (virtualized runs are excluded — their scheduler/fork-join
behaviour is host-specific and not representative).

---

## Headline: k-quant models run on fast SIMD kernels

k-quant weights (`Q4_K`, `Q5_K`, `Q6_K`) and the `IQ4_NL` i-quant are mapped onto
the library's existing fused int8 SIMD kernels (`q41q8` / `q8q8`) rather than being
expanded to dense float32. Where the int8 kernel is available this is the default
path and runs the common k-quant models **~3× faster** than the dense-float
fallback (Apple M5 Max, SmolLM2-1.7B):

| 1.7B model | dense-float fallback | int8 fast path | speedup |
|------------|---------------------:|---------------:|:-------:|
| `Q4_K_M`   | ~10.6 t/s            | **33.6 t/s**   | 3.2×    |
| `Q5_K_M`   | ~10.0 t/s            | **26.6 t/s**   | 2.7×    |

No hand-written k-quant assembly: `Q4_K` is re-expressed as `Q4_1`, `Q5_K` as two
`Q4_1` weights, `Q6_K`/`IQ4_NL` as `Q8_0` — all reusing kernels that already exist.

---

## Decode throughput — SmolLM2-1.7B (t/s, higher is better)

| Quant     | M5 Max | 9900X | M2 Max | i5-8500T | X5675¹ |
|-----------|-------:|------:|-------:|---------:|-------:|
| Q4_0      | **55.4** | 35.6 | 26.5 | 11.9 | 1.3 |
| Q4_K_M    | **33.6** | 22.8 | 13.4 | 4.9  | 1.5 |
| Q8_0      | 29.0   | 25.7 | 11.7 | 8.4  | 5.6 |
| Q5_K_M    | 26.6   | 17.0 | 10.8 | 3.7  | 1.5 |
| Q3_K_M    | 15.2   | 11.8 |  5.3 | 2.0  | 1.5 |
| Q2_K      | 10.8   |  8.6 |  3.6 | 1.4  | 1.5 |

## Decode throughput — SmolLM2-135M (t/s)

| Quant     | M5 Max | 9900X | M2 Max | i5-8500T | X5675 |
|-----------|-------:|------:|-------:|---------:|------:|
| Q4_0      | **208.1** | 179.8 | 119.7 | 54.4 | 11.1 |
| Q8_0      | 160.4  | 170.0 |  81.6 | 49.5 | 32.7 |
| Q6_K      | 136.5  | 137.6 |  66.7 | 31.7 | 25.7 |
| Q4_K_M    |  96.6  |  74.8 |  48.9 | 15.1 | 10.5 |
| Q3_K_L    |  77.5  |  75.3 |  38.0 | 15.1 | 12.6 |
| Q5_K_M    |  76.5  |  77.1 |  38.1 | 14.6 | 13.2 |
| Q2_K      |  74.0  |  72.2 |  34.7 |  4.7 | 12.7 |

¹ **X5675 has no AVX2.** The int8 SIMD kernels (`q41q8`/`q8q8`) require AVX2/NEON, so
on this host k-quants fall back to dense float32, which is slow on this 2011 CPU.
Only the scalar `Q8_0` path stays usable. It is included as the worst-case floor;
any CPU from the last decade does much better.

---

## Platforms

| Host       | CPU                       | ISA path                | Notes |
|------------|---------------------------|-------------------------|-------|
| M5 Max     | Apple M5 Max              | ARM64 NEON (`SDOT`)     | unified memory, very high bandwidth |
| M2 Max     | Apple M2 Max              | ARM64 NEON (`SDOT`)     | ~½ the M5 Max throughput |
| 9900X      | AMD Ryzen 9 9900X (Zen 5) | x86-64 AVX2 + AVX-VNNI  | fast desktop; dual-channel DDR5 |
| i5-8500T   | Intel Core i5-8500T       | x86-64 AVX2             | low-power desktop |
| X5675      | Intel Xeon X5675 (2011)   | x86-64 SSE4.2 (no AVX2) | dense-float fallback only |

One binary runs on all of them: `GOAMD64=v1`, with SIMD kernels selected at runtime
via CPUID (and a self-checked AVX-VNNI kernel where available).

---

## How the k-quant fast paths work

Each k-quant sub-block is re-expressed in terms of a format that already has a fused
int8 SIMD kernel — no new k-quant assembly:

| Source | Mapping | Kernel | Fidelity |
|--------|---------|--------|----------|
| `Q4_K` | → `Q4_1` (same per-32 `scale·q4 + min`) | `q41q8` (1 pass) | lossless |
| `Q5_K` | → two `Q4_1` (low nibbles + high bit) | `q41q8` (2 passes) | lossless |
| `Q6_K` | → `Q8_0` (both symmetric; 8 bits cover 6) | `q8q8` | near-lossless |
| `IQ4_NL` | → `Q8_0` (codebook values are int8) | `q8q8` | lossless |

The fast path is the default wherever the int8 kernel is available; hosts without it
keep dense float32 (no regression). See the [supported tensor types](README.md#supported-gguf-tensor-types).

## What the numbers show

- **Apple Silicon leads on decode**, mostly from memory bandwidth: the M5 Max's
  unified memory feeds the weight stream faster than the x86 hosts' DDR, and runs
  ~2× the M2 Max throughout.
- **`Q4_K_M` decodes slower than `Q4_0` despite both being 4-bit.** `Q4_0` uses the
  simpler `q4q8` kernel; `Q4_K` (`q41q8`) carries a per-group `min` correction and an
  f16 scalar tail. The gap is structural, and the leaner NEON path hides it better
  than x86 — on the 9900X `Q4_K_M` even trails `Q8_0`.
- **Decode is compute-bound on fast CPUs.** The M5 Max at 26–33 t/s on a 1.7B
  k-quant moves only ~50–60 GB/s, well under its memory ceiling, so throughput tracks
  kernel speed (int8 SIMD) rather than bytes moved. Slow and old hosts are the
  reverse: dense-float compute dominates.
- **k-quants run on every supported host**, including the small-model `Q2_K` /
  `Q3_K_L` repacks that are mostly `IQ4_NL` (576-wide rows, not a multiple of 256).

See [Performance tuning](README.md#performance-tuning) for the `SLLM_Q8_KERNEL`,
`SLLM_PARALLEL_THRESHOLD`, and `SLLM_KQUANT_PACKED` knobs.

---

## Raw stats

Decode / prefill t/s per host, `MODELS=all`.

### Apple M5 Max (ARM64)

```
model                              prefill   decode
SmolLM2-1.7B-Q2_K                    11.8     10.8
SmolLM2-1.7B-Q3_K_M                  17.3     15.2
SmolLM2-1.7B-Q4_0                    83.3     55.4
SmolLM2-1.7B-Q4_K_M                  42.1     33.6
SmolLM2-1.7B-Q5_K_M                  31.5     26.6
SmolLM2-1.7B-Q8_0                    37.1     29.0
SmolLM2-135M-Q2_K                   154.7     74.0
SmolLM2-135M-Q3_K_L                 176.0     77.5
SmolLM2-135M-Q4_0                   446.1    208.1
SmolLM2-135M-Q4_K_M                 194.4     96.6
SmolLM2-135M-Q5_K_M                 156.4     76.5
SmolLM2-135M-Q6_K                   281.2    136.5
SmolLM2-135M-Q8_0                   366.5    160.4
SmolLM2-360M-Q4_0                   229.2    123.4
SmolLM2-360M-Q8_0                   155.1     96.5
tinyllama-1.1b-Q8_0                  55.4     35.3
```

### Apple M2 Max (ARM64)

```
model                              prefill   decode
SmolLM2-1.7B-Q2_K                     4.2      3.6
SmolLM2-1.7B-Q3_K_M                   6.2      5.3
SmolLM2-1.7B-Q4_0                    39.1     26.5
SmolLM2-1.7B-Q4_K_M                  17.3     13.4
SmolLM2-1.7B-Q5_K_M                  12.8     10.8
SmolLM2-1.7B-Q8_0                    14.2     11.7
SmolLM2-135M-Q2_K                    55.7     34.7
SmolLM2-135M-Q3_K_L                  65.7     38.0
SmolLM2-135M-Q4_0                   265.7    119.7
SmolLM2-135M-Q4_K_M                  82.0     48.9
SmolLM2-135M-Q5_K_M                  60.7     38.1
SmolLM2-135M-Q6_K                   105.6     66.7
SmolLM2-135M-Q8_0                   155.9     81.6
SmolLM2-360M-Q4_0                   118.1     65.6
SmolLM2-360M-Q8_0                    59.4     42.9
tinyllama-1.1b-Q8_0                  22.2     15.6
```

### AMD Ryzen 9 9900X (x86-64, AVX2 + AVX-VNNI)

```
model                              prefill   decode
SmolLM2-1.7B-Q2_K                    12.6      8.6
SmolLM2-1.7B-Q3_K_M                  17.3     11.8
SmolLM2-1.7B-Q4_0                    75.7     35.6
SmolLM2-1.7B-Q4_K_M                  36.9     22.8
SmolLM2-1.7B-Q5_K_M                  28.2     17.0
SmolLM2-1.7B-Q8_0                    77.0     25.7
SmolLM2-135M-Q2_K                   163.1     72.2
SmolLM2-135M-Q3_K_L                 181.0     75.3
SmolLM2-135M-Q4_0                   405.4    179.8
SmolLM2-135M-Q4_K_M                 147.7     74.8
SmolLM2-135M-Q5_K_M                 173.8     77.1
SmolLM2-135M-Q6_K                   323.4    137.6
SmolLM2-135M-Q8_0                   451.6    170.0
SmolLM2-360M-Q4_0                   211.7    105.1
SmolLM2-360M-Q8_0                   255.9     86.2
tinyllama-1.1b-Q8_0                 105.4     36.5
```

### Intel Core i5-8500T (x86-64, AVX2)

```
model                              prefill   decode
SmolLM2-1.7B-Q2_K                     1.5      1.4
SmolLM2-1.7B-Q3_K_M                   2.3      2.0
SmolLM2-1.7B-Q4_0                    20.4     11.9
SmolLM2-1.7B-Q4_K_M                   6.2      4.9
SmolLM2-1.7B-Q5_K_M                   4.7      3.7
SmolLM2-1.7B-Q8_0                    12.2      8.4
SmolLM2-135M-Q2_K                    17.8      4.7
SmolLM2-135M-Q3_K_L                  25.0     15.1
SmolLM2-135M-Q4_0                   114.6     54.4
SmolLM2-135M-Q4_K_M                  23.1     15.1
SmolLM2-135M-Q5_K_M                  23.3     14.6
SmolLM2-135M-Q6_K                    53.3     31.7
SmolLM2-135M-Q8_0                   104.7     49.5
SmolLM2-360M-Q4_0                    61.1     28.1
SmolLM2-360M-Q8_0                    48.0     25.0
tinyllama-1.1b-Q8_0                  18.1     11.7
```

### Intel Xeon X5675 (x86-64, no AVX2 — dense-float fallback)

```
model                              prefill   decode
SmolLM2-1.7B-Q2_K                     1.6      1.5
SmolLM2-1.7B-Q3_K_M                   1.6      1.5
SmolLM2-1.7B-Q4_0                     1.3      1.3
SmolLM2-1.7B-Q4_K_M                   1.6      1.5
SmolLM2-1.7B-Q5_K_M                   1.6      1.5
SmolLM2-1.7B-Q8_0                     7.7      5.6
SmolLM2-135M-Q2_K                    19.2     12.7
SmolLM2-135M-Q3_K_L                  19.2     12.6
SmolLM2-135M-Q4_0                    16.8     11.1
SmolLM2-135M-Q4_K_M                  15.0     10.5
SmolLM2-135M-Q5_K_M                  19.5     13.2
SmolLM2-135M-Q6_K                    38.9     25.7
SmolLM2-135M-Q8_0                    62.8     32.7
SmolLM2-360M-Q4_0                     6.2      5.6
SmolLM2-360M-Q8_0                    27.2     17.5
tinyllama-1.1b-Q8_0                  11.3      7.4
```
