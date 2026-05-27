# Performance Benchmarks

All runs use the `infer` binary with greedy decoding and 200 generated tokens:

```bash
./bin/infer \
  -model models/<model>.gguf \
  -prompt "Write a detailed explanation of how transformers work" \
  -tokens 200 \
  -strategy greedy
```

Decode t/s is the primary metric — it measures sustained autoregressive generation throughput. Prefill t/s reflects how fast the prompt is processed (single forward pass, less memory-bandwidth-bound for short prompts).

---

## Decode throughput (t/s) — higher is better

| Model | M2 Max | M5 Max | Xeon Gold 6271C (VM) |
|-------|-------:|-------:|---------------------:|
| SmolLM2-135M-Q8_0 | 48.0 | **75.6** | 33.3 |
| SmolLM2-360M-Q8_0 | 24.6 | **36.5** | 18.0 |
| SmolLM2-1.7B-Q8_0 |  9.7 | **14.2** |  7.9 |

## Prefill throughput (t/s) — higher is better

| Model | M2 Max | M5 Max | Xeon Gold 6271C (VM) |
|-------|-------:|-------:|---------------------:|
| SmolLM2-135M-Q8_0 | 138.8 | **231.9** | 119.7 |
| SmolLM2-360M-Q8_0 |  44.3 |  **91.8** |  57.5 |
| SmolLM2-1.7B-Q8_0 |  12.9 |  **20.6** |  17.0 |

---

## Platform notes

| Platform | Architecture | Notes |
|----------|-------------|-------|
| Apple M2 Max | ARM64 (NEON) | NEON FMLA path (`dot_arm64.s`) |
| Apple M5 Max | ARM64 (NEON) | NEON FMLA path (`dot_arm64.s`) |
| Intel Xeon Gold 6271C | x86-64 (AVX2) | AVX2 path (`dot_avx2_amd64.s`); run inside a VM |

**M5 Max vs M2 Max:** ~1.5× faster across all models — consistent with the memory bandwidth advantage of the M5 Max.

**Apple Silicon vs Xeon:** M5 Max is ~2× faster; M2 Max is ~1.3–1.4× faster. The Xeon result is from a VM (additional virtualisation overhead) and uses AVX2 whereas native bare-metal x86 would be higher.

---

## Raw stats

### Apple M2 Max

```
SmolLM2-135M-Instruct-Q8_0.gguf
  prompt      29 tokens   prefill    209 ms   138.8 t/s
  generated  200 tokens   decode    4170 ms    48.0 t/s

SmolLM2-360M-Instruct-Q8_0.gguf
  prompt      29 tokens   prefill    655 ms    44.3 t/s
  generated  200 tokens   decode    8136 ms    24.6 t/s

SmolLM2-1.7B-Instruct-Q8_0.gguf
  prompt      29 tokens   prefill   2252 ms    12.9 t/s
  generated  200 tokens   decode   20689 ms     9.7 t/s
```

### Apple M5 Max

```
SmolLM2-135M-Instruct-Q8_0.gguf
  prompt      29 tokens   prefill    125 ms   231.9 t/s
  generated  200 tokens   decode    2644 ms    75.6 t/s

SmolLM2-360M-Instruct-Q8_0.gguf
  prompt      29 tokens   prefill    316 ms    91.8 t/s
  generated  200 tokens   decode    5481 ms    36.5 t/s

SmolLM2-1.7B-Instruct-Q8_0.gguf
  prompt      29 tokens   prefill   1406 ms    20.6 t/s
  generated  200 tokens   decode   14058 ms    14.2 t/s
```

### Intel Xeon Gold 6271C (VM)

```
SmolLM2-135M-Instruct-Q8_0.gguf
  prompt      29 tokens   prefill    242 ms   119.7 t/s
  generated  200 tokens   decode    6000 ms    33.3 t/s

SmolLM2-360M-Instruct-Q8_0.gguf
  prompt      29 tokens   prefill    505 ms    57.5 t/s
  generated  200 tokens   decode   11126 ms    18.0 t/s

SmolLM2-1.7B-Instruct-Q8_0.gguf
  prompt      29 tokens   prefill   1704 ms    17.0 t/s
  generated  200 tokens   decode   25390 ms     7.9 t/s
```
