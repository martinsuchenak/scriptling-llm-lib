# Fleet benchmarking

`fleet.sh` cross-compiles the `infer` binary, pushes just the ~4 MB binary to
each host (models stay on the remote), runs benchmarks / CPU profiles, and
collects results. Remotes need **no Go toolchain** — only the binary and the
`.gguf` models.

## Setup

1. Create your host config from the template (it is git-ignored, so your real
   hostnames/IPs never get committed):

   ```bash
   cp bench/hosts.conf.example bench/hosts.conf
   ```

   Edit it with your hosts (ssh target, GOOS/GOARCH, remote scratch dir, and the
   directory for the models). `ssh_target` may be a `~/.ssh/config` alias; use
   `local` to run on the current machine. Unix paths may use `~` (but not
   spaces); Windows hosts need OpenSSH with the default cmd.exe shell and
   forward-slash paths (`C:/...`). If no `hosts.conf` exists, `fleet.sh` falls
   back to `hosts.conf.example`.
2. Put the `.gguf` files in a local source dir (default `<repo>/models`,
   override with `FLEET_MODELS_SRC`). `deploy`/`bench`/`run`/`profile`
   **auto-create the remote dirs and copy the selected models** there — skipping
   any already present (set `FORCE=1` to re-copy). The remote needs no Go
   toolchain and no manual file placement, just ssh access.
3. Run from a machine that can reach the fleet (your laptop/workstation — not a
   sandboxed CI box).

## Commands

```bash
./bench/fleet.sh check all                 # connectivity + CPU info for every host
./bench/fleet.sh bench all                 # t/s matrix (prefill/decode) across the fleet
./bench/fleet.sh bench 9900x               # just one host
./bench/fleet.sh run 9900x SmolLM2-1.7B-Instruct-Q4_0.gguf -tokens 100
./bench/fleet.sh profile 9900x SmolLM2-1.7B-Instruct-Q4_0.gguf 150   # CPU profile -> pprof -top
./bench/fleet.sh models                     # list what MODELS=all would resolve to
```

### Testing every model (k-quants included)

`bench`/`deploy` act on the model set in `MODELS`. Set `MODELS=all` to use **every
`.gguf` in the local source dir** — the simplest way to test all models, including
the k-quant variants (`Q4_K_M`, `Q5_K_M`, `Q6_K`, …):

```bash
MODELS=all ./bench/fleet.sh models           # preview the full list first
MODELS=all ./bench/fleet.sh bench all        # bench every model on every host
MODELS=all ./bench/fleet.sh deploy m2max     # just push every model to one host
```

Missing models are copied to each host on demand (large k-quant files are pushed
once and reused). To bench a specific subset, pass an explicit list:

```bash
MODELS="SmolLM2-1.7B-Instruct-Q4_K_M.gguf SmolLM2-1.7B-Instruct-Q8_0.gguf" \
    ./bench/fleet.sh bench 9900x
```

### Embedding models

Encoder embedding models (`bert` / `nomic-bert` — all-MiniLM, BGE, E5, GTE,
nomic-embed-text) are detected automatically: `infer` embeds the prompt instead
of generating and reports embedding throughput. They appear in the same matrix —
just drop the `.gguf` into the models dir and bench as usual. An encoder has no
decode loop, so the two columns carry **latency** (`ms`/embed) and **throughput**
(`emb/s`) instead of prefill/decode `t/s`:

```bash
MODELS="all-MiniLM-L6-v2-Q8_0.gguf nomic-embed-text-v1.5.Q8_0.gguf" \
    ./bench/fleet.sh bench m5max
```

The matrix is **single-stream** (one embed at a time ≈ one core for short texts).
For **aggregate** throughput across all cores — the figure that matters for
embedding a corpus, and what a batched `llama-embedding` run reports — use one or
both of:

- `-embed-batch B` — embed B texts in one packed forward pass (`EmbedBatch`); each
  weight is read once per batch. This is the main throughput lever, especially for
  larger models.
- `-embed-concurrency N` — drive N concurrent embedders (`Embed`/`EmbedBatch` are
  goroutine-safe).

```bash
./bench/fleet.sh run 9900x nomic-embed-text-v1.5.Q8_0.gguf -embed-batch 64
./bench/fleet.sh run 9900x nomic-embed-text-v1.5.Q8_0.gguf -embed-batch 32 -embed-concurrency 4
```

`bench` parses the prefill/decode `t/s` lines; `profile` fetches the `.prof`
back and runs `go tool pprof -top` against the matching local build (cross-arch
symbolization works because Go binaries carry their symbol table).

## Useful env overrides

| Var | Default | Purpose |
|-----|---------|---------|
| `MODELS` | 135M/360M/1.7B Q8 + 1.7B Q4 | space-separated basenames, or `all` for every `.gguf` in the source dir |
| `FLEET_MODELS_SRC` | `<repo>/models` | local dir the models are copied from |
| `FORCE` | unset | `1` re-copies models even if already present |
| `FLEET_TOKENS` | 120 | tokens to generate |
| `FLEET_ENV` | unset | `VAR=value` pairs exported on the **remote** for each run (ssh doesn't forward your local env), e.g. `FLEET_ENV="SLLM_KQUANT_PACKED=1"` |
| `FLEET_PROMPT` | a story prompt | prompt text |
| `SSH_OPTS` | `-o BatchMode=yes -o ConnectTimeout=8` | extra ssh/scp options |
| `FLEET_CONF` | `bench/hosts.conf` | alternate config |

Library tuning knobs are passed to the remote binary with `FLEET_ENV` (ssh does
not forward your local environment). For example, to compare the opt-in packed
k-quant path against the dense default on a host:

```bash
# dense float k-quant (default)
MODELS="SmolLM2-1.7B-Instruct-Q4_K_M.gguf" ./bench/fleet.sh bench m5max
# native packed k-quant (~4-6x less memory)
FLEET_ENV="SLLM_KQUANT_PACKED=1" \
    MODELS="SmolLM2-1.7B-Instruct-Q4_K_M.gguf" ./bench/fleet.sh bench m5max
```

The same mechanism works for `SLLM_Q8_KERNEL`, `SLLM_PARALLEL_THRESHOLD`,
`GODEBUG=asyncpreemptoff=1`, etc. — e.g. `FLEET_ENV="SLLM_Q8_KERNEL=float GODEBUG=asyncpreemptoff=1"`.

Build artifacts land in `bench/bin/` (git-ignored).
