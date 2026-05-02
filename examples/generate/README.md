# generate

Scripting script for text generation from GGUF model files. Designed to be run via the `sllm` binary.

## Usage

```bash
./sllm run.py <model.gguf> "prompt" [max_tokens] [strategy] [stats]

# Options:
#   stats        - print generation stats (tokens/s, timing)
#   -t <name>    - use a named chat template (smollm2, chatml)
#   -rp <value>  - set repeat penalty (default: 1.15)
```

## Examples

```bash
# Greedy generation
./sllm run.py ../../models/SmolLM2-135M-Instruct-Q8_0.gguf "Hello" 20 greedy stats

# With temperature sampling
./sllm run.py ../../models/SmolLM2-1.7B-Instruct-Q8_0.gguf "Write a poem" 50 temperature stats
```
