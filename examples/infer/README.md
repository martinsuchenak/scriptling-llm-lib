# infer

Standalone CLI for running inference on GGUF models. No scripting required — pass the model path, prompt and options as flags, get the response on stdout and performance stats on stderr.

## Build

```bash
go build -o ../../bin/infer .
```

Or via the task runner from the repo root:

```bash
task build:examples
```

## Usage

```
infer -model <path> -prompt <text> [options]

Options:
  -model string          Path to GGUF model file (required)
  -prompt string         Input prompt (required)
  -system string         System prompt (optional)
  -tokens int            Maximum tokens to generate (default 200)
  -strategy string       Sampling strategy: greedy, temperature, top_k, top_p (default "greedy")
  -temperature float     Sampling temperature — used by temperature, top_k, top_p (default 0.8)
  -top-k int             Top-K candidates for top_k strategy (default 50)
  -top-p float           Nucleus probability threshold for top_p strategy (default 0.9)
  -repeat-penalty float  Repetition penalty — 1.0 disables it (default 1.1)
  -repeat-last-n int     Token window considered for repeat penalty (default 64)
```

The generated text is written to **stdout** so it can be piped or redirected. Stats (token counts, latency, tokens/sec) are written to **stderr**.

## Examples

```bash
# Greedy decoding — deterministic, fastest
./infer \
  -model ../../models/SmolLM2-360M-Instruct-Q8_0.gguf \
  -prompt "Explain recursion in one paragraph" \
  -tokens 150

# Nucleus sampling — more varied output
./infer \
  -model ../../models/SmolLM2-1.7B-Instruct-Q8_0.gguf \
  -prompt "Write a haiku about distributed systems" \
  -strategy top_p -temperature 0.9 -top-p 0.95 \
  -tokens 60

# With a system prompt
./infer \
  -model ../../models/SmolLM2-1.7B-Instruct-Q8_0.gguf \
  -system "You are a concise assistant. Answer in one sentence." \
  -prompt "What is SIMD?" \
  -tokens 80

# Pipe the response into another command
./infer -model ../../models/SmolLM2-360M-Instruct-Q8_0.gguf \
        -prompt "List three Go proverbs" | wc -w

# Redirect stats to a file while displaying the response
./infer -model ../../models/SmolLM2-1.7B-Instruct-Q8_0.gguf \
        -prompt "What is the speed of light?" 2>stats.txt
```

## Output format

```
The capital of France is Paris.
---
prompt       27 tokens   prefill   1638 ms     16.5 t/s
generated     8 tokens   decode     877 ms      9.1 t/s
```

The separator line and stats go to stderr; only the generated text goes to stdout.
