# Scriptling Compatibility Guide

This document catalogs every compatibility issue encountered while developing sllm on the [Scriptling](https://scriptling.dev) runtime, along with the workarounds applied. Scriptling implements a Python-like language in a sandboxed environment. Most standard Python syntax works, but several constructs require adjustment.

## Syntax Restrictions

### No parenthesized multi-line imports

`from X import (a, b, c)` fails with a parse error. Only single-line import forms are accepted.

**Workaround:** Write each import on its own line, or fit imports on a single line.

```python
# Rejected by Scriptling
from os import (
    path,
    listdir,
    read_file,
)

# Accepted
from os import path
from os import listdir
from os import read_file
```

**Affected files:** Every module with multi-symbol imports.

### No walrus operator

The assignment expression operator (`:=`) is not part of the Scriptling grammar.

**Workaround:** Use a separate assignment statement before the expression that needs the value.

**Affected files:** None currently. Avoided by convention.

### No async/await

The `async` and `await` keywords are not recognized. The runtime is entirely synchronous.

**Workaround:** Write all I/O and computation synchronously. No coroutines, no event loop.

**Affected files:** All modules.

### No type annotations

Function signatures with type annotations (`def f(x: int) -> str:`) produce parse errors.

**Workaround:** Omit all annotations. Use plain parameter lists and rely on docstrings or comments where clarity is needed.

```python
# Rejected
def softmax(x: list) -> list:

# Accepted
def softmax(x):
```

**Affected files:** All modules.

## Missing Built-ins and Standard Library Functions

### No `open()` built-in

The built-in `open()` function is not available. Files must be read through the Scriptling-specific API.

**Workaround:** Use `os.read_file(path)` which returns the entire file contents as a string. The `weights.py` module wraps this in a compatibility helper:

```python
def _read_file(path):
    try:
        return os.read_file(path)
    except Exception:
        return open(path, "r").read()
```

This allows the same code to run under both CPython and Scriptling.

**Affected files:** `weights.py`

### No `math.tanh`

The `math` module exposes `exp`, `log`, `sqrt`, and others, but `tanh` is absent.

**Workaround:** Implemented from `exp` in `linalg.py`:

```python
def tanh(x):
    e = math.exp(2 * x)
    return (e - 1) / (e + 1)
```

**Affected files:** `linalg.py`

### No `random.choices()`

The `random` module provides `random()` and `randint()` but not `choices()` for weighted sampling.

**Workaround:** Implemented `weighted_sample()` using `random.random()` and a cumulative distribution:

```python
def weighted_sample(weights):
    total = sum(weights)
    r = random.random() * total
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w
        if r < cumulative:
            return i
    return len(weights) - 1
```

**Affected files:** `linalg.py`

### No direct dict iteration

Iterating over a dict directly (`for k in my_dict:`) raises "expected iterable, got DICT".

**Workaround:** Call `.keys()` explicitly:

```python
# Rejected
for k in my_dict:
    ...

# Accepted
for k in my_dict.keys():
    ...
```

Value iteration uses `.values()`, and key-value pairs use `.items()`. Both work as expected.

**Affected files:** `tokenizer.py`, `model.py`

### No `yield` / generators

Generator functions using `yield` are not supported.

**Workaround:** Build and return lists instead of yielding values incrementally.

```python
# Rejected
def tokenize(text):
    for token in split(text):
        yield token

# Accepted
def tokenize(text):
    result = []
    for token in split(text):
        result.append(token)
    return result
```

**Affected files:** `tokenizer.py`, `layers.py`

## Regex Engine

### RE2 only — no lookahead or lookbehind

Scriptling uses the RE2 regex engine, which does not support `(?=...)`, `(?!...)`, `(?<=...)`, or `(?<!...)` assertions.

**Workaround:** The GPT-2 pre-tokenization regex pattern was simplified to avoid lookaheads. Where the original pattern uses lookahead to enforce boundary conditions, the simplified version relies on broader matches followed by post-processing in Python.

**Affected files:** `tokenizer.py`

## What Works Without Changes

The following standard Python features are available in Scriptling and required no workaround:

| Feature | Notes |
|---|---|
| `sys.argv` | Returns a list of strings, identical to CPython behavior. |
| `json.loads` / `json.dumps` | Standard library, works as expected. |
| `os.read_file(path)` | Scriptling-specific. Reads the entire file as a string. Not a CPython function. |
| Import resolution | `import foo` resolves `foo.py` relative to the running script's directory. |
| `__file__` | Set when running from a file. Behaves like CPython's `__file__`. |
| `math.exp`, `math.log`, `math.sqrt` | Available and functionally equivalent. |
| `random.random()`, `random.randint()` | Available and functionally equivalent. |
| List comprehensions | Work as expected. |
| String methods | `.split()`, `.join()`, `.strip()`, `.encode()`, `.decode()` all available. |
| Class definitions | Classes with `__init__` and instance methods work. |
| Exception handling | `try` / `except` / `finally` works. |

## Quick Reference

| Python feature | Scriptling status | Workaround |
|---|---|---|
| Multi-line parenthesized imports | Not supported | One import per line |
| `open()` | Not available | `os.read_file(path)` |
| Direct dict iteration | Not supported | `.keys()` / `.values()` / `.items()` |
| `math.tanh` | Not available | Implement via `math.exp` |
| `random.choices()` | Not available | Custom cumulative distribution |
| Type annotations | Not supported | Omit annotations |
| Lookahead/lookbehind regex | Not supported (RE2) | Simplify pattern, post-process |
| `yield` generators | Not supported | Return lists |
| Walrus operator `:=` | Not supported | Separate assignment |
| `async` / `await` | Not supported | Synchronous only |
