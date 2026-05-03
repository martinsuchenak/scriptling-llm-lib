# session

Multi-turn chat using persistent KV cache sessions. Each response builds on the previous context without reprocessing the full conversation.

## Usage

```bash
./sllm chat.py <model.gguf> [max_tokens] [strategy]
```

Type prompts interactively (one per line). Type `quit` to exit, `clear` to reset the session.

## Example

```bash
cd examples/sllm
go build -o sllm .
cd ../session
./sllm chat.py ../../models/SmolLM2-135M-Instruct-Q8_0.gguf 20 greedy
```

Each subsequent prompt reuses the cached KV state from the previous turns, so only new tokens are processed.
