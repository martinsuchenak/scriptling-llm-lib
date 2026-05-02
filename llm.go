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

		"rms_norm": {
			Fn: fnRmsNorm,
			HelpText: `rms_norm(x, weight, eps=1e-5) - RMS normalization`,
		},
		"rope": {
			Fn: fnRope,
			HelpText: `rope(x, start_pos=0) - Rotary Position Embeddings`,
		},
		"silu_gate": {
			Fn: fnSiluGate,
			HelpText: `silu_gate(gate, up) - Fused SiLU activation + element-wise multiply`,
		},
		"attention": {
			Fn: fnAttention,
			HelpText: `attention(q, k, v, causal=True) - Scaled dot-product attention`,
		},
		"linear": {
			Fn: fnLinear,
			HelpText: `linear(x, weight, bias=None) - Fused matrix multiply + optional bias add`,
		},
		"linear_row": {
			Fn: fnLinearRow,
			HelpText: `linear_row(x, weight, bias=None) - Last-row-only linear`,
		},
		"linear_q8": {
			Fn: fnLinearQ8Fast,
			HelpText: `linear_q8(x, raw, groups_per_row) - Quantized Q8_0 matmul (fast)`,
		},
		"linear_row_q8": {
			Fn: fnLinearRowQ8Fast,
			HelpText: `linear_row_q8(x, raw, groups_per_row) - Last-row-only Q8_0 matmul (fast)`,
		},
		"linear_q4": {
			Fn: fnLinearQ4Fast,
			HelpText: `linear_q4(x, raw, groups_per_row) - Quantized Q4_0 matmul (fast)

Parameters:
  x              - input matrix (seq_len, in_features)
  raw            - string of raw Q4_0 block bytes (out_features * groups_per_row * 18 bytes)
  groups_per_row - number of Q4_0 groups per weight row (in_features / 32)

Returns:
  Matrix (seq_len, out_features).`,
		},
		"linear_row_q4": {
			Fn: fnLinearRowQ4Fast,
			HelpText: `linear_row_q4(x, raw, groups_per_row) - Last-row-only Q4_0 matmul (fast)

Parameters:
  x              - input matrix (seq_len, in_features)
  raw            - string of raw Q4_0 block bytes
  groups_per_row - number of Q4_0 groups per weight row (in_features / 32)

Returns:
  Vector (out_features,).`,
		},
		"top_k": {
			Fn: fnTopK,
			HelpText: `top_k(logits, k) - Find the k highest-scoring elements`,
		},
		"dequantize_q8": {
			Fn: fnDequantizeQ8,
			HelpText: `dequantize_q8(data, scales, group_size) - Dequantize int8 with per-group scales`,
		},
		"dequantize_q8_0": {
			Fn: fnDequantizeQ8_0,
			HelpText: `dequantize_q8_0(raw, n_groups) - Native GGUF Q8_0 block dequantization`,
		},
		"dequantize_q4_0": {
			Fn: fnDequantizeQ4_0,
			HelpText: `dequantize_q4_0(raw, n_groups) - Native GGUF Q4_0 block dequantization`,
		},
		"sample": {
			Fn: fnSample,
			HelpText: `sample(logits, strategy, temperature=1.0, top_k=50) - Native sampling

Strategies: "greedy", "temperature", "top_k", "top_p"

Parameters:
  logits      - 1D list or FloatArray of logit values
  strategy    - sampling strategy string
  temperature - temperature for non-greedy (default 1.0)
  top_k       - k for top_k strategy (default 50)

Keyword args: temperature, top_k, top_p

Returns:
  Integer token index.

Example:
  sample(logits, "greedy")
  sample(logits, "top_k", 0.8, 50)
  sample(logits, "top_p", top_p=0.9)`,
		},
		"split_heads": {
			Fn: fnSplitHeads,
			HelpText: `split_heads(x, n_heads) - Split last dimension into n_heads heads

Parameters:
  x       - 2D matrix (seq_len, d_model)
  n_heads - number of attention heads

Returns:
  List of n_heads 2D FloatArrays, each (seq_len, d_k).`,
		},
		"merge_heads": {
			Fn: fnMergeHeads,
			HelpText: `merge_heads(heads) - Concatenate head outputs back to d_model

Parameters:
  heads - list of n_heads 2D FloatArrays, each (seq_len, d_k)

Returns:
  2D FloatArray (seq_len, n_heads * d_k).`,
		},
		"repeat_kv": {
			Fn: fnRepeatKV,
			HelpText: `repeat_kv(heads, n_rep) - Repeat KV heads for GQA

Parameters:
  heads - list of KV head FloatArrays
  n_rep - number of times to repeat each head

Returns:
  Expanded list of FloatArrays.`,
		},
		"concat_rows": {
			Fn: fnConcatRows,
			HelpText: `concat_rows(a, b) - Concatenate two matrices along row axis`,
		},
		"slice_rows": {
			Fn: fnSliceRows,
			HelpText: `slice_rows(x, start, end) - Extract rows [start, end)`,
		},
		"flatten": {
			Fn: fnFlatten,
			HelpText: `flatten(x) - Flatten 2D matrix to 1D`,
		},
		"reshape": {
			Fn: fnReshape,
			HelpText: `reshape(data, rows, cols) - Reshape 1D list to 2D FloatArray

Parameters:
  data - 1D list of numbers
  rows - number of rows
  cols - number of columns

Returns:
  2D FloatArray (rows, cols). data length must equal rows*cols.`,
		},
		"zeros": {
			Fn: fnZeros,
			HelpText: `zeros(n) or zeros(rows, cols) - Create zero-filled FloatArray

Parameters:
  n    - length of 1D array
  or
  rows - number of rows
  cols - number of columns

Returns:
  1D or 2D FloatArray filled with zeros.`,
		},
		"arange": {
			Fn: fnArange,
			HelpText: `arange(stop) or arange(start, stop[, step]) - Create range FloatArray`,
		},
		"quantize_q8": {
			Fn: fnQuantizeQ8,
			HelpText: `quantize_q8(data, rows, cols) - Quantize flat float data to Q8_0 raw bytes`,
		},
		"quantize_q8_rows": {
			Fn: fnQuantizeQ8Rows,
			HelpText: `quantize_q8_rows(matrix, cols) - Quantize list-of-lists to Q8_0 raw bytes

Parameters:
  matrix - 2D list of floats [[row0], [row1], ...]
  cols   - number of columns per row (must be divisible by 32)

Returns:
  String of raw Q8_0 block bytes.`,
		},
	},
	map[string]object.Object{
		"VERSION": &object.String{Value: "1.1.0"},
	},
	"LLM inference primitives for transformer model execution",
)
