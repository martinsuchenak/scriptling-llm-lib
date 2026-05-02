# Model Converter (convert.py)

Build-time tool that converts binary model files into JSON format loadable by the sllm engine. Runs in standard Python -- not Scriptling-compatible.

---

## 1. Usage

```
python convert.py <model.bin|model.gguf> [output.json] [--encoding q8|fp32] [--tokenizer tokenizer.bin]
```

**Arguments:**

| Argument | Description |
|---|---|
| `<input>` | Path to a llama2.c `.bin` or GGUF file |
| `[output.json]` | Optional output path. Defaults to `<input_stem>.json` |
| `--encoding q8` | Symmetric int8 quantization (default, ~75% size reduction) |
| `--encoding fp32` | Raw float32 arrays (exact, no loss) |
| `--tokenizer PATH` | Path to `tokenizer.bin` for llama2.c models |

**Examples:**

```bash
# Convert llama2.c model with Q8 quantization (default)
python convert.py models/stories15M.bin

# Convert with explicit output path and fp32 encoding
python convert.py models/stories15M.bin output/stories15m.json --encoding fp32

# Convert llama2.c model with external tokenizer
python convert.py models/stories15M.bin --tokenizer tokenizer.bin

# Convert GGUF model (tokenizer is embedded in the file)
python convert.py model.gguf --encoding q8
```

---

## 2. Supported Input Formats

### llama2.c `.bin`

Karpathy's binary format used by the [llama2.c](https://github.com/karpathy/llama2.c) project. Two variants are supported:

- **v0 (legacy):** No magic header. Parameters start at byte offset 0. Format is detected when the first 4 bytes do not match any known magic number.
- **v1 (with header):** Starts with magic `0x616b3432`, followed by a version field. Parameters begin at byte offset 256 (the header is zero-padded to 256 bytes). Includes a `shared_classifier` flag indicating whether the output projection reuses the token embedding matrix.

Typical models: `stories15M.bin`, `stories42M.bin`, `stories110M.bin`.

### GGUF

The [llama.cpp GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) container format. Identified by magic `0x46475547` ("GGUF" in little-endian). Supports the following tensor types:

| Type ID | Name | Description |
|---|---|---|
| 0 | F32 | 32-bit float (lossless) |
| 1 | F16 | 16-bit IEEE float |
| 2 | Q4\_0 | 4-bit quantization, 32-element blocks with f16 scale |
| 6 | Q5\_0 | 5-bit quantization, 32-element blocks with f16 scale |
| 8 | Q8\_0 | 8-bit quantization, 32-element blocks with f16 scale |

All quantized tensors are dequantized to float32 during conversion, then re-quantized with the selected output encoding.

---

## 3. Output Encoding Options

### q8 (default)

Symmetric int8 quantization with per-group scaling. Weights are divided into groups (default 64 elements), and each group gets its own float scale factor. Reduces model size by approximately 75% compared to fp32.

Output format for a weight tensor:

```json
{
  "shape": [rows, cols],
  "encoding": "q8",
  "group_size": 64,
  "data": [int, ...],
  "scales": [float, ...]
}
```

### fp32

Raw float32 arrays. No quantization applied. Exact representation of weights at the cost of larger file size.

Output format:

```json
{
  "shape": [rows, cols],
  "data": [float, ...]
}
```

1-D tensors (norms, biases) are always stored as fp32 regardless of the encoding setting.

---

## 4. `quantize_q8(flat, group_size)`

Quantizes a flat float32 array to int8 using symmetric per-group quantization.

**Algorithm:**

1. **Group sizing:** Start with the requested `group_size` (default 64). If the array length is not evenly divisible, halve the group size repeatedly until it is. Minimum group size is 1.
2. **Per-group processing:** For each group of `group_size` elements:
   - Find the absolute maximum value `abs_max`.
   - Clamp `abs_max` to a minimum of `1e-10` to avoid division by zero.
   - Compute `scale = abs_max / 127.0`.
   - For each element: `q = round(value / scale)`, clamped to `[-127, 127]`.
3. **Result:** Returns the quantized integer array, the per-group scale array, and the effective group size.

The symmetric range `[-127, 127]` (rather than `[-128, 127]`) ensures that the negation of a quantized value is always representable.

---

## 5. `make_weight(data, shape, encoding, group_size, transpose)`

Creates a weight JSON object from a flat float array, applying optional transposition and encoding.

**Parameters:**

- `data` -- flat `list[float]` in row-major order
- `shape` -- tensor dimensions (1-D for norms, 2-D for matrices)
- `encoding` -- `"q8"` or `"fp32"`
- `group_size` -- quantization group size (default 64)
- `transpose` -- whether to transpose the matrix before encoding

**Behavior:**

1. 1-D tensors are returned directly as fp32 with no processing.
2. If `transpose=True`, the matrix is transposed in-place (see section 9 for rationale).
3. If encoding is `"q8"` and the tensor has enough elements, `quantize_q8` is applied.
4. Otherwise, the raw float array is returned.

---

## 6. `convert_llama2c(path, encoding, group_size, tokenizer_path)`

Converts a llama2.c binary model file into config and weights dictionaries.

### Binary Format Layout

The file is a flat sequence of little-endian values. The layout differs between v0 and v1:

**Header (v1 only):**

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | Magic (`0x616b3432`) |
| 4 | 4 | Version (int32) |

**Parameters (v0 starts at offset 0, v1 at offset 8):**

| Field | Type |
|---|---|
| dim | int32 |
| hidden\_dim | int32 |
| n\_layers | int32 |
| n\_heads | int32 |
| n\_kv\_heads | int32 |
| vocab\_size | int32 (negative if shared classifier in v1) |
| max\_seq\_len | int32 |

In v1, a `shared_classifier` byte follows at offset 28, then the header is zero-padded to byte 256.

**Tensor data (v0 starts after the 7 parameters at offset 28, v1 at offset 256):**

1. Token embedding matrix: `vocab_size * dim` floats
2. Attention norms: `n_layers` tensors of `dim` floats each
3. Query weights (w\_q): `n_layers` tensors of `dim * dim` floats
4. Key weights (w\_k): `n_layers` tensors of `dim * dim` floats
5. Value weights (w\_v): `n_layers` tensors of `dim * dim` floats
6. Output projection (w\_o): `n_layers` tensors of `dim * dim` floats
7. FFN norms: `n_layers` tensors of `dim` floats each
8. Gate weights (w\_gate): `n_layers` tensors of `hidden_dim * dim` floats
9. Down projection (w\_down): `n_layers` tensors of `dim * hidden_dim` floats
10. Up projection (w\_up): `n_layers` tensors of `hidden_dim * dim` floats
11. Final norm: `dim` floats
12. Precomputed RoPE frequencies: `max_seq_len * (dim / n_heads / 2) * 2` floats (skipped -- the engine computes RoPE itself)
13. Output classifier: `vocab_size * dim` floats (only if `shared_classifier` is false)

**Weight naming convention:**

```
token_embedding.weight
blocks.{i}.attn_norm.weight
blocks.{i}.attn.w_q.weight
blocks.{i}.attn.w_k.weight
blocks.{i}.attn.w_v.weight
blocks.{i}.attn.w_o.weight
blocks.{i}.ffn_norm.weight
blocks.{i}.ffn.w_gate.weight
blocks.{i}.ffn.w_down.weight
blocks.{i}.ffn.w_up.weight
final_norm.weight
output.weight
```

When `shared_classifier` is true, `output.weight` is set to a copy of the token embedding matrix instead of reading a separate tensor.

---

## 7. `convert_gguf(path, encoding, group_size)`

Converts a GGUF file into config, weights, and tokenizer data.

### GGUF Parsing

The GGUF container is read sequentially:

1. **File header:** magic (`0x46475547`), version, tensor count, metadata count.
2. **Metadata key-value pairs:** Each entry is a string key followed by a type-tagged value. Supported types include UINT8 through FLOAT64, STRING, BOOL, and ARRAY. Architecture parameters are stored under keys like `llama.embedding_length` and `llama.block_count`.
3. **Tensor info array:** Each entry has a name, dimension count, per-dimension sizes, GGUF tensor type, and byte offset into the data section.
4. **Data section:** Starts at the next aligned boundary (default alignment: 32 bytes). Tensor data is located at `data_start + tensor_offset`.

### Tensor Dequantization

Each tensor is dequantized to float32 based on its GGUF type before being re-encoded with the chosen output encoding:

- **F32:** Direct unpacking.
- **F16:** IEEE 754 half-precision to float via `struct` with format `<e`.
- **Q4\_0:** 32-element blocks. Each block is 18 bytes: 2-byte f16 scale + 16 bytes of packed 4-bit values (two values per byte, offset by -8).
- **Q5\_0:** 32-element blocks. Each block is 22 bytes: 2-byte f16 scale + 4-bit high-order bits (uint32) + 16 bytes of low-order nibbles. Values offset by -16, with the high bit adding 16.
- **Q8\_0:** 32-element blocks. Each block is 34 bytes: 2-byte f16 scale + 32 signed int8 values.

### Metadata Extraction

The converter reads these architecture-specific keys (prefixed with the architecture name, e.g. `llama.`):

| Key | Field |
|---|---|
| `{arch}.vocab_size` | Vocabulary size |
| `{arch}.block_count` | Number of transformer layers |
| `{arch}.attention.head_count` | Number of attention heads |
| `{arch}.attention.head_count_kv` | Number of KV heads |
| `{arch}.embedding_length` | Model dimension (d\_model) |
| `{arch}.feed_forward_length` | FFN hidden dimension |
| `{arch}.context_length` | Maximum sequence length |
| `{arch}.attention.layer_norm_rms_epsilon` | RMSNorm epsilon |

Tokenizer data is extracted from `tokenizer.ggml.tokens`, `tokenizer.ggml.scores`, and `tokenizer.ggml.merges`.

### GGUF Tensor Name Mapping

GGUF tensor names are mapped to the engine's naming convention:

| GGUF Name | Engine Name |
|---|---|
| `token_embd.weight` | `token_embedding.weight` |
| `output_norm.weight` | `final_norm.weight` |
| `output.weight` | `output.weight` |
| `blk.{i}.attn_norm.weight` | `blocks.{i}.attn_norm.weight` |
| `blk.{i}.attn_q.weight` | `blocks.{i}.attn.w_q.weight` |
| `blk.{i}.attn_k.weight` | `blocks.{i}.attn.w_k.weight` |
| `blk.{i}.attn_v.weight` | `blocks.{i}.attn.w_v.weight` |
| `blk.{i}.attn_output.weight` | `blocks.{i}.attn.w_o.weight` |
| `blk.{i}.ffn_norm.weight` | `blocks.{i}.ffn_norm.weight` |
| `blk.{i}.ffn_gate.weight` | `blocks.{i}.ffn.w_gate.weight` |
| `blk.{i}.ffn_down.weight` | `blocks.{i}.ffn.w_down.weight` |
| `blk.{i}.ffn_up.weight` | `blocks.{i}.ffn.w_up.weight` |

Non-parameterized names (e.g. `token_embd.weight`) are matched first by exact string comparison. Parameterized names (those containing `{i}`) are matched by prefix/suffix pattern extraction, and the block index is parsed from the GGUF name and substituted into the engine name.

---

## 8. `load_llama2c_tokenizer(path, vocab_size)`

Parses the `tokenizer.bin` file shipped with llama2.c models and extracts a vocabulary, special tokens, and BPE merge rules.

### Binary Format

| Field | Type | Description |
|---|---|---|
| max\_token\_length | int32 | Maximum byte length of any token |
| For each token (vocab\_size total): | | |
| score | float32 | Token score from sentencepiece |
| len | int32 | Byte length of the token string |
| bytes | `len` bytes | UTF-8 token string |

### BPE Merge Extraction

Since the sentencepiece vocabulary does not contain explicit merge rules, they are reconstructed by decomposing each multi-character token (at index 4 or higher, skipping special tokens) into two existing vocabulary tokens. For each token, every possible split position is tried. The first split where both halves exist in the vocabulary becomes a merge rule.

Merges are sorted by score in descending order to preserve sentencepiece's priority ordering.

Output structure:

```json
{
  "vocab": {"<token>": index, ...},
  "special": {"<s>": 1, "</s>": 2, ...},
  "merges": [["left", "right"], ...],
  "scores": [float, ...],
  "type": "bpe"
}
```

If no merges are extracted, `type` is set to `"simple"`.

---

## 9. Weight Transposition

llama2.c stores linear layer weights in row-major order with shape `(out_features, in_features)`. This is the standard convention where row `i` contains the weights that produce output feature `i`.

The sllm engine computes linear projections as `matmul(x, W)`, where `x` has shape `(1, in_features)` and `W` must have shape `(in_features, out_features)` to produce output of shape `(1, out_features)`. This requires the weight matrix to be stored as `(in_features, out_features)`.

To bridge this convention gap, `make_weight` transposes 2-D tensors when `transpose=True`. The transposition is performed element-by-element on the flat array: for a matrix of shape `(R, C)`, element `[r, c]` in the original becomes element `[c, r]` in the transposed array, and the shape is swapped to `(C, R)`.

Embedding matrices are not transposed because they are used as lookup tables (indexed by token ID), not as matmul operands.

---

## 10. Output File Structure

The converter produces a single JSON file with this top-level structure:

```json
{
  "config": {
    "vocab_size": 32000,
    "d_model": 288,
    "n_heads": 6,
    "n_kv_heads": 6,
    "n_layers": 6,
    "max_seq_len": 256,
    "d_ff": 768,
    "mlp_type": "gated",
    "bias": false,
    "norm_type": "rmsnorm",
    "pos_encoding": "rope"
  },
  "vocab": { ... },
  "weights": { ... }
}
```

The `vocab` section contains the tokenizer data (either extracted from GGUF metadata or parsed from `tokenizer.bin`). The `weights` section maps engine-style weight names to encoded weight objects.
