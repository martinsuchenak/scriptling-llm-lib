# sllm

Full-featured Scriptling CLI runtime with LLM inference support. This is the primary way to run Scriptling scripts that use `llm.generate()`.

Includes all standard libraries: `llm`, `math`, `os`, `fs`, `sys`, plus file-based library loading (can import local `.py` files).

## Build

```bash
go build -o ../../bin/sllm .
```

## Usage

```bash
# Run a script
./sllm script.py

# Evaluate an expression
./sllm -eval 'import llm; print(llm.VERSION)'

# Lint a script
./sllm -lint script.py
```

## Example

```bash
./sllm ../generate/run.py ../../models/SmolLM2-135M-Instruct-Q8_0.gguf "Hello" 20 greedy
```
