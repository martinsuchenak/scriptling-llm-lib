# Weight Loading API Reference

## Overview

`weights.py` loads model weights from JSON files and assembles a fully constructed `LLMModel`. It supports two weight encodings (fp32 and int8 quantized) and two architecture families (GPT-2 and Llama). A companion tokenizer is reconstructed from the same file.

---

## Weight Encodings

Each weight tensor in the JSON file carries an `encoding` field:

| Encoding | Description | Size vs fp32 |
|----------|-------------|--------------|
| `fp32` | Raw float array. Exact precision. | 1x |
| `q8` | Symmetric int8 with per-group float scales. | ~25% |

When `encoding` is omitted it defaults to `fp32`.

---

## Functions

### `reshape_2d(flat, rows, cols)`

Converts a flat list into a 2D list-of-lists with the given dimensions.

**Parameters:**

- `flat` (`list`) -- 1D source list of length `rows * cols`
- `rows` (`int`) -- number of rows in the output
- `cols` (`int`) -- number of columns in the output

**Returns:** `list[list]` -- 2D list with shape `(rows, cols)`

---

### `dequantize_q8(data, scales, group_size)`

Dequantizes int8 values back to float32 using per-group scale factors. Each group of `group_size` consecutive values shares one scale entry:

```
float_val = int_val * scale[group_idx]
```

**Parameters:**

- `data` (`list[int]`) -- quantized int8 values
- `scales` (`list[float]`) -- one scale factor per group; length is `len(data) / group_size`
- `group_size` (`int`) -- number of values per scale group

**Returns:** `list[float]` -- dequantized float32 values

---

### `load_weight(raw)`

Loads a single weight tensor from its JSON representation. Handles both `fp32` and `q8` encodings.

**Parameters:**

- `raw` (`dict`) -- weight entry with keys:
  - `shape` (`list[int]`) -- tensor shape (1D or 2D)
  - `data` (`list`) -- raw values (float for fp32, int for q8)
  - `encoding` (`str`, optional) -- `"fp32"` (default) or `"q8"`
  - `scales` (`list[float]`, q8 only) -- per-group scale factors
  - `group_size` (`int`, optional, q8 only) -- defaults to `64`

**Returns:**

- 1D shape `[n]` -> `list[float]`
- 2D shape `[rows, cols]` -> `list[list[float]]`

---

### `load_weights(raw_weights)`

Iterates over all weight entries in a dict and loads each one.

**Parameters:**

- `raw_weights` (`dict[str, dict]`) -- maps weight names to their raw JSON entries

**Returns:** `dict[str, list | list[list]]` -- loaded weight tensors keyed by name

---

### `load_tokenizer(raw_vocab, tokenizer_type)`

Reconstructs a tokenizer from the vocab section of the model file.

**Parameters:**

- `raw_vocab` (`dict`) -- vocab entry with keys:
  - `vocab` (`dict[str, int]`) -- token-to-id mapping
  - `special` (`dict[str, int]`, optional) -- special token mapping. Defaults to `{"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}`
  - `merges` (`list[str]`, BPE only) -- merge rules for BPE tokenization
- `tokenizer_type` (`str`) -- `"simple"` (default) or `"bpe"`

**Returns:** `SimpleTokenizer` or `BPETokenizer`

---

### `build_model(config, weights)`

Constructs a full `LLMModel` from a config dict and loaded weight tensors. Architecture is detected from config keys:

| Config key | GPT-2 | Llama |
|------------|-------|-------|
| `norm_type` | `"layernorm"` | `"rmsnorm"` |
| `pos_encoding` | `"learned"` | `"rope"` |
| `mlp_type` | `"standard"` | `"gated"` |
| `bias` | `true` | `false` |

**Config fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vocab_size` | `int` | required | vocabulary size |
| `d_model` | `int` | required | model hidden dimension |
| `n_heads` | `int` | required | number of attention heads |
| `n_layers` | `int` | required | number of transformer blocks |
| `max_seq_len` | `int` | `512` | maximum sequence length |
| `d_ff` | `int` | `d_model * 4` | feed-forward hidden dimension |
| `mlp_type` | `str` | `"standard"` | `"standard"` or `"gated"` |
| `bias` | `bool` | `true` | whether linear layers use bias |
| `norm_type` | `str` | `"layernorm"` | `"layernorm"` or `"rmsnorm"` |
| `pos_encoding` | `str` | `"learned"` | `"learned"` or `"rope"` |
| `shared_classifier` | `bool` | `false` | tie output weights to token embedding |

**Parameters:**

- `config` (`dict`) -- model configuration
- `weights` (`dict`) -- loaded weight tensors from `load_weights`

**Returns:** `LLMModel`

**Weight keys consumed:**

Global weights:
- `token_embedding.weight`
- `pos_embedding.weight` (learned positional encoding only)
- `final_norm.weight`, `final_norm.bias`
- `output.weight`

Per-layer weights (indexed as `blocks.{i}.*`):
- `attn.w_q.weight`, `attn.w_k.weight`, `attn.w_v.weight`, `attn.w_o.weight`
- `attn_norm.weight`, `attn_norm.bias`
- `ffn_norm.weight`, `ffn_norm.bias`
- `ffn.w_up.weight`, `ffn.b_up.bias` (standard MLP only)
- `ffn.w_down.weight`, `ffn.b_down.bias` (standard MLP only)
- `ffn.w_gate.weight`, `ffn.w_up.weight`, `ffn.w_down.weight` (gated MLP)

---

### `load_model(config_json, weights_json)`

Loads a model from raw JSON strings or pre-parsed dicts. Does not load a tokenizer.

**Parameters:**

- `config_json` (`str | dict`) -- model config as JSON string or dict
- `weights_json` (`str | dict`) -- weight entries as JSON string or dict

**Returns:** `LLMModel`

---

### `_read_file(path)`

Internal compatibility helper. Reads a file to a string using `os.read_file()` when running in the Scriptling environment, falling back to the standard `open()` builtin.

---

### `load_model_file(path)`

Main entry point. Reads a JSON model file, parses config/weights/vocab sections, and returns a fully constructed model and tokenizer.

**Parameters:**

- `path` (`str`) -- path to the JSON model file

**Returns:** `(model, tokenizer)` tuple of `LLMModel` and `SimpleTokenizer | BPETokenizer`

---

## JSON Model File Format

Model files are JSON with three top-level keys: `config`, `vocab`, and `weights`.

```json
{
  "config": {
    "vocab_size": 32000,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 4,
    "max_seq_len": 512,
    "d_ff": 2048,
    "mlp_type": "gated",
    "bias": false,
    "norm_type": "rmsnorm",
    "pos_encoding": "rope",
    "shared_classifier": false
  },

  "vocab": {
    "type": "bpe",
    "vocab": { "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "a": 4, "b": 5 },
    "special": { "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3 },
    "merges": ["a b", "b a"]
  },

  "weights": {
    "token_embedding.weight": {
      "shape": [32000, 512],
      "encoding": "fp32",
      "data": [0.001, -0.002, 0.003, "..."]
    },
    "blocks.0.attn.w_q.weight": {
      "shape": [512, 512],
      "encoding": "q8",
      "group_size": 64,
      "data": [12, -7, 3, "..."],
      "scales": [0.023, 0.019, "..."]
    },
    "blocks.0.attn.w_k.weight": { "shape": [512, 512], "data": ["..."] },
    "blocks.0.attn.w_v.weight": { "shape": [512, 512], "data": ["..."] },
    "blocks.0.attn.w_o.weight": { "shape": [512, 512], "data": ["..."] },
    "blocks.0.attn_norm.weight": { "shape": [512], "data": ["..."] },
    "blocks.0.ffn_norm.weight": { "shape": [512], "data": ["..."] },
    "blocks.0.ffn.w_gate.weight": { "shape": [2048, 512], "data": ["..."] },
    "blocks.0.ffn.w_up.weight": { "shape": [2048, 512], "data": ["..."] },
    "blocks.0.ffn.w_down.weight": { "shape": [512, 2048], "data": ["..."] },
    "final_norm.weight": { "shape": [512], "data": ["..."] },
    "output.weight": { "shape": [32000, 512], "data": ["..."] }
  }
}
```

The `vocab` section uses `"type": "simple"` for character-level tokenization (no `merges` key needed) or `"type": "bpe"` for byte-pair encoding with a `merges` list. Individual weight entries default to `"encoding": "fp32"` when the `encoding` key is omitted.

Use `convert.py` to produce these JSON files from llama2.c `.bin` or GGUF checkpoints.
