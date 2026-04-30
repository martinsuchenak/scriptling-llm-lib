// Package scriptlingllmlib provides LLM inference primitives as a Scriptling library.
//
// This library exposes 24 native functions under the "llm" namespace, covering
// the core operations needed to implement transformer model inference in Scriptling
// scripts. All functions are implemented using the Scriptling Native API for
// zero-reflection overhead on every call.
//
// # Registration
//
// Import the package and register the pre-built [Library] with a Scriptling
// interpreter:
//
//	p := scriptling.New()
//	p.RegisterLibrary(scriptlingllmlib.Library)
//
// Scripts then access all functions via "import llm":
//
//	import llm
//	token = llm.argmax(logits)
//	out = llm.attention(q, k, v, causal=True)
//
// # Function Categories
//
// The library is organised into four groups:
//
//   - Inference helpers: argmax, argmin, topk, clip
//   - Activation functions: sigmoid, relu, gelu, silu
//   - Vector operations: vec_add, vec_sub, vec_mul, vec_scale, vec_apply
//   - LLM primitives: rms_norm, rope, silu_gate, attention, linear, linear_row,
//     linear_q8, linear_row_q8, top_k, dequantize_q8, dequantize_q8_0
//   - Matrix utilities: concat_rows, slice_rows, flatten
//
// # Data Conventions
//
// Scalars are plain float64 values (Scriptling INTEGER or FLOAT).
// Vectors are list[float] (Scriptling LIST of numbers).
// Matrices are list[list[float]] (Scriptling LIST of LIST of numbers).
// Weight matrices follow PyTorch convention: shape (out_features, in_features).
package scriptlingllmlib

import (
	"github.com/paularlott/scriptling/object"
)

// LibraryName is the Scriptling import name for this library ("llm").
const LibraryName = "llm"

// Library is the pre-built Scriptling library containing all LLM inference
// functions. Register it with a Scriptling interpreter via
// [scriptling.Scriptling.RegisterLibrary].
//
// The library exposes 27 functions and one constant (VERSION).
var Library = object.NewLibrary(LibraryName,
	map[string]*object.Builtin{
		// --- Inference helpers ---

		"argmax": {
			Fn: fnArgmax,
			HelpText: `argmax(x) - Return the index of the maximum value in a list

Parameters:
  x - list of numbers

Returns:
  Integer index of the largest element.

Example:
  argmax([0.1, 0.9, 0.3])  # returns 1`,
		},
		"argmin": {
			Fn: fnArgmin,
			HelpText: `argmin(x) - Return the index of the minimum value in a list

Parameters:
  x - list of numbers

Returns:
  Integer index of the smallest element.

Example:
  argmin([3.0, 1.0, 2.0])  # returns 1`,
		},
		"topk": {
			Fn: fnTopk,
			HelpText: `topk(x, k) - Return the top k (index, value) pairs sorted descending

Parameters:
  x - list of numbers
  k - number of top elements (clamped to list length)

Returns:
  List of [index, value] pairs sorted by value descending.

Example:
  topk([1, 5, 3, 4, 2], 3)  # [[1, 5.0], [3, 4.0], [2, 3.0]]`,
		},
		"clip": {
			Fn: fnClip,
			HelpText: `clip(x, lo, hi) - Clamp values to [lo, hi]

Parameters:
  x  - number or list of numbers
  lo - lower bound (number)
  hi - upper bound (number)

Returns:
  Clamped value or list. lo must be <= hi.

Example:
  clip([-2.0, 0.5, 3.0], -1.0, 2.0)  # [-1.0, 0.5, 2.0]
  clip(5.0, 0.0, 3.0)                 # 3.0`,
		},

		// --- Activation functions ---

		"sigmoid": {
			Fn: fnSigmoid,
			HelpText: `sigmoid(x) - Logistic sigmoid: 1 / (1 + exp(-x))

Parameters:
  x - number

Returns:
  Float in (0, 1).

Example:
  sigmoid(0)    # 0.5
  sigmoid(100)  # ~1.0`,
		},
		"relu": {
			Fn: fnRelu,
			HelpText: `relu(x) - Rectified Linear Unit: max(0, x)

Parameters:
  x - number

Returns:
  Float, 0 for negative inputs.

Example:
  relu(-1)  # 0.0
  relu(5)   # 5.0`,
		},
		"gelu": {
			Fn: fnGelu,
			HelpText: `gelu(x) - Gaussian Error Linear Unit

Computes 0.5 * x * (1 + erf(x / sqrt(2))). Used in BERT, GPT-2, T5.

Parameters:
  x - number

Returns:
  Float.

Example:
  gelu(1.0)  # ~0.8413`,
		},
		"silu": {
			Fn: fnSilu,
			HelpText: `silu(x) - Sigmoid Linear Unit (Swish): x * sigmoid(x)

Used in LLaMA, Gemma, Mistral.

Parameters:
  x - number

Returns:
  Float.

Example:
  silu(2.0)  # ~1.7616`,
		},

		// --- Vector operations ---

		"vec_add": {
			Fn: fnVecAdd,
			HelpText: `vec_add(a, b) - Element-wise addition of two vectors

Parameters:
  a - list of numbers
  b - list of numbers (same length as a)

Returns:
  New list where result[i] = a[i] + b[i].

Example:
  vec_add([1, 2, 3], [4, 5, 6])  # [5.0, 7.0, 9.0]`,
		},
		"vec_sub": {
			Fn: fnVecSub,
			HelpText: `vec_sub(a, b) - Element-wise subtraction of two vectors

Parameters:
  a - list of numbers
  b - list of numbers (same length as a)

Returns:
  New list where result[i] = a[i] - b[i].

Example:
  vec_sub([5, 3, 1], [1, 2, 3])  # [4.0, 1.0, -2.0]`,
		},
		"vec_mul": {
			Fn: fnVecMul,
			HelpText: `vec_mul(a, b) - Element-wise multiplication of two vectors

Parameters:
  a - list of numbers
  b - list of numbers (same length as a)

Returns:
  New list where result[i] = a[i] * b[i].

Example:
  vec_mul([2, 3], [4, 5])  # [8.0, 15.0]`,
		},
		"vec_scale": {
			Fn: fnVecScale,
			HelpText: `vec_scale(a, s) - Multiply every element of a vector by a scalar

Parameters:
  a - list of numbers
  s - scalar number

Returns:
  New list where result[i] = a[i] * s.

Example:
  vec_scale([1, 2, 3], 2.0)  # [2.0, 4.0, 6.0]`,
		},
		"vec_apply": {
			Fn: fnVecApply,
			HelpText: `vec_apply(x, fn_name) - Apply a named activation function element-wise

Parameters:
  x       - list of numbers
  fn_name - one of: "sigmoid", "relu", "gelu", "silu"

Returns:
  New list with the activation applied to each element.

Example:
  vec_apply([-1, 0, 1, 2], "relu")      # [0.0, 0.0, 1.0, 2.0]
  vec_apply([-1, 0, 1, 2], "sigmoid")   # [~0.27, 0.5, ~0.73, ~0.88]`,
		},

		// --- LLM inference primitives ---

		"rms_norm": {
			Fn: fnRmsNorm,
			HelpText: `rms_norm(x, weight, eps=1e-5) - RMS normalization

For each row: divide by root-mean-square, then multiply element-wise by weight.
Used by LLaMA, Mistral, Phi, Qwen, and most modern transformers.
Called twice per transformer layer (before attention and before FFN).

Parameters:
  x      - matrix (seq_len, dim) as list of lists
  weight - vector (dim,) as list
  eps    - optional epsilon for numerical stability (default 1e-5)

Returns:
  Matrix of same shape as x.

Example:
  x = [[0.5, -0.3, 0.8], [0.1, 0.2, -0.4]]
  w = [1.0, 1.0, 1.0]
  rms_norm(x, w)       # default eps=1e-5
  rms_norm(x, w, 1e-6) # custom eps`,
		},
		"rope": {
			Fn: fnRope,
			HelpText: `rope(x, start_pos=0) - Rotary Position Embeddings

Applies position-dependent rotation to pairs of dimensions. Standard positional
encoding for LLaMA, Mistral, Qwen, Phi, and most modern transformers.
Called 2x per head per layer (Q and K).

Parameters:
  x          - matrix (seq_len, d_k), d_k must be even
  start_pos  - optional starting position offset (default 0)

Returns:
  Matrix of same shape as x.

Example:
  q = [[1.0, 0.0, 0.0, 1.0]]
  rope(q)         # position 0
  rope(q, 8)      # starting at position 8`,
		},
		"silu_gate": {
			Fn: fnSiluGate,
			HelpText: `silu_gate(gate, up) - Fused SiLU activation + element-wise multiply (SwiGLU)

Computes silu(gate) * up element-wise. The core of SwiGLU used in LLaMA,
Mistral, and most modern FFNs. Fusing avoids an intermediate allocation.

Parameters:
  gate - matrix (seq_len, d_ff)
  up   - matrix (seq_len, d_ff), same shape as gate

Returns:
  Matrix of same shape.

Example:
  gate = [[1.0, -1.0], [0.0, 2.0]]
  up   = [[1.0,  1.0], [1.0, 1.0]]
  silu_gate(gate, up)`,
		},
		"attention": {
			Fn: fnAttention,
			HelpText: `attention(q, k, v, causal=True) - Scaled dot-product attention

Computes softmax(Q @ K^T / sqrt(d_k)) @ V with optional causal masking.
The single most expensive operation in transformer inference.

Parameters:
  q      - query matrix (q_len, d_k)
  k      - key matrix (kv_len, d_k)
  v      - value matrix (kv_len, d_k)
  causal - optional boolean (default True). When True and q_len > 1,
           masks out positions where key index > query index.

Returns:
  Matrix (q_len, d_k).

Example:
  q = [[1.0, 0.0]]
  k = [[1.0, 0.0], [0.0, 1.0]]
  v = [[1.0, 0.0], [0.0, 1.0]]
  attention(q, k, v)             # causal=True (default)
  attention(q, k, v, False)      # no masking`,
		},
		"linear": {
			Fn: fnLinear,
			HelpText: `linear(x, weight, bias=None) - Fused matrix multiply + optional bias add

Computes x @ weight.T + bias. Weight is stored as (out_features, in_features)
following PyTorch convention. Every transformer layer has 7+ linear calls.

Parameters:
  x      - input matrix (seq_len, in_features)
  weight - weight matrix (out_features, in_features)
  bias   - optional bias vector (out_features)

Returns:
  Matrix (seq_len, out_features).

Example:
  x = [[1.0, 2.0]]
  w = [[1.0, 0.0], [0.0, 1.0]]
  linear(x, w)             # [[1.0, 2.0]]
  linear(x, w, [10.0, 20.0])  # [[11.0, 22.0]]`,
		},
		"linear_row": {
			Fn: fnLinearRow,
			HelpText: `linear_row(x, weight, bias=None) - Compute only the last row of linear()

Same as linear() but returns only the last row as a vector. Used for the output
projection during generation where only the last token's logits matter.
Saves (seq_len - 1) * out_features operations.

Parameters:
  x      - input matrix (seq_len, in_features)
  weight - weight matrix (out_features, in_features)
  bias   - optional bias vector (out_features)

Returns:
  Vector (out_features,) - the last row of the full output.

Example:
  x = [[1.0, 2.0], [3.0, 4.0]]
  w = [[1.0, 0.0], [0.0, 1.0]]
  linear_row(x, w)  # [3.0, 4.0]`,
		},
		"linear_q8": {
			Fn: fnLinearQ8,
			HelpText: `linear_q8(x, raw, groups_per_row) - Quantized Q8_0 matmul (no dequantize)

Computes x @ weight.T where weight is stored as raw Q8_0 blocks.
Avoids the overhead of dequantizing weights to float first.
Each Q8_0 block: 2-byte f16 scale + 32 int8 values = 34 bytes.
Weight shape: (out_features, in_features) where in_features = groups_per_row * 32.

Parameters:
  x              - input matrix (seq_len, in_features)
  raw            - string of raw Q8_0 block bytes (out_features * groups_per_row * 34 bytes)
  groups_per_row - number of Q8_0 groups per weight row (in_features / 32)

Returns:
  Matrix (seq_len, out_features).

Example:
  # weight [100, 576] => groups_per_row = 18, raw = 100 * 18 * 34 bytes
  linear_q8(x, raw, 18)`,
		},
		"linear_row_q8": {
			Fn: fnLinearRowQ8,
			HelpText: `linear_row_q8(x, raw, groups_per_row) - Last-row-only quantized Q8_0 matmul

Same as linear_q8 but computes only the last row of x, returning a vector.
Used for the output projection during generation.

Parameters:
  x              - input matrix (seq_len, in_features)
  raw            - string of raw Q8_0 block bytes
  groups_per_row - number of Q8_0 groups per weight row (in_features / 32)

Returns:
  Vector (out_features,).

Example:
  linear_row_q8(x, raw, 18)`,
		},
		"top_k": {
			Fn: fnTopK,
			HelpText: `top_k(logits, k) - Find the k highest-scoring elements using partial sort

Returns list of (index, value) tuples sorted descending. Uses O(n) partial
sort instead of O(n log n) full sort.

Parameters:
  logits - list of numbers
  k      - number of top elements (clamped to list length)

Returns:
  List of [index, value] pairs sorted by value descending.

Example:
  top_k([0.1, 0.5, 0.3, 0.9, 0.7], 3)
  # [[3, 0.9], [4, 0.7], [1, 0.5]]`,
		},
		"dequantize_q8": {
			Fn: fnDequantizeQ8,
			HelpText: `dequantize_q8(data, scales, group_size) - Dequantize int8 data with per-group scales

Each value is dequantized as: float = int8 * scale[group_index].
Compatible with the Q8_0 format used by llama.cpp/GGUF.

Parameters:
  data       - list of integers in int8 range [-128, 127]
  scales     - list of floats, one per group
  group_size - number of elements per scale group (typically 64)

Returns:
  List of dequantized floats.

Example:
  dequantize_q8([10, -5, 20, 15], [0.1, 0.2], 2)
  # [1.0, -0.5, 4.0, 3.0]`,
		},
		"dequantize_q8_0": {
			Fn: fnDequantizeQ8_0,
			HelpText: `dequantize_q8_0(raw, n_groups) - Native GGUF Q8_0 block dequantization

Takes raw Q8_0 block data (as returned by fs.read_bytes) and the number of groups.
Each Q8_0 block is 34 bytes: 2-byte f16 scale + 32 int8 values.
Returns n_groups * 32 dequantized floats. No Python-side loop needed.

Parameters:
  raw      - string of raw Q8_0 block bytes (from fs.read_bytes)
  n_groups - number of Q8_0 groups (each group = 34 bytes = 32 elements)

Returns:
  List of dequantized floats (length = n_groups * 32).

Example:
  raw = fs.read_bytes("model.gguf", offset, n_groups * 34)
  dequantize_q8_0(raw, n_groups)`,
		},

		// --- Matrix utilities ---

		"concat_rows": {
			Fn: fnConcatRows,
			HelpText: `concat_rows(a, b) - Concatenate two matrices along the row axis

Both matrices must have the same number of columns.

Parameters:
  a - matrix (m, n)
  b - matrix (p, n)

Returns:
  Matrix (m + p, n).

Example:
  concat_rows([[1, 2]], [[3, 4], [5, 6]])
  # [[1, 2], [3, 4], [5, 6]]`,
		},
		"slice_rows": {
			Fn: fnSliceRows,
			HelpText: `slice_rows(x, start, end) - Extract rows [start, end) from a matrix

Parameters:
  x     - matrix (m, n)
  start - start row index (clamped to 0)
  end   - end row index (exclusive, clamped to m)

Returns:
  Matrix with rows [start, end).

Example:
  slice_rows([[1,2],[3,4],[5,6],[7,8]], 1, 3)
  # [[3, 4], [5, 6]]`,
		},
		"flatten": {
			Fn: fnFlatten,
			HelpText: `flatten(x) - Flatten a 2D matrix into a 1D list

Parameters:
  x - matrix (m, n)

Returns:
  List of length m * n with elements in row-major order.

Example:
  flatten([[1, 2], [3, 4]])  # [1.0, 2.0, 3.0, 4.0]`,
		},
	},
	map[string]object.Object{
		"VERSION": &object.String{Value: "1.0.0"},
	},
	"LLM inference primitives for transformer model execution",
)
