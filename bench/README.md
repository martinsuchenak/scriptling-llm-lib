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
```

`bench` parses the prefill/decode `t/s` lines; `profile` fetches the `.prof`
back and runs `go tool pprof -top` against the matching local build (cross-arch
symbolization works because Go binaries carry their symbol table).

## Useful env overrides

| Var | Default | Purpose |
|-----|---------|---------|
| `MODELS` | 135M/360M/1.7B Q8 + 1.7B Q4 | space-separated basenames for `bench` |
| `FLEET_MODELS_SRC` | `<repo>/models` | local dir the models are copied from |
| `FORCE` | unset | `1` re-copies models even if already present |
| `FLEET_TOKENS` | 120 | tokens to generate |
| `FLEET_PROMPT` | a story prompt | prompt text |
| `SSH_OPTS` | `-o BatchMode=yes -o ConnectTimeout=8` | extra ssh/scp options |
| `FLEET_CONF` | `bench/hosts.conf` | alternate config |

Tuning knobs still apply on the remote via the env, e.g.
`SSH_OPTS=... FLEET_TOKENS=100 ./bench/fleet.sh run ...`, and the library's own
`SLLM_Q8_KERNEL` / `SLLM_PARALLEL_THRESHOLD` / `GODEBUG=asyncpreemptoff=1` can be
set by prefixing them in `FLEET_PROMPT`-style wrappers or editing the invocation.

Build artifacts land in `bench/bin/` (git-ignored).
