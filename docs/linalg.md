# linalg.py API Reference

Pure-Python linear algebra operations for transformer inference. All computation uses only the `math` standard-library module. No type annotations, no external dependencies, no walrus operator, no async, and no generators — compatible with the Scriptling runtime.

## Data Representation

| Concept | Python type | Example |
|---------|-------------|---------|
| **Vector** | `list[float]` | `[0.1, 0.2, 0.3]` |
| **Matrix** | `list[list[float]]` | `[[1.0, 2.0], [3.0, 4.0]]` |

Matrices are row-major: `matrix[i]` is row `i`, `matrix[i][j]` is element at row `i`, column `j`.

Shape notation below uses `(M, K)` to mean `M` rows of length `K`.

---

## Activation Functions

### `apply_relu(a)`

ReLU activation. Clamps every element to `[0, +inf)`.

```
apply_relu(a) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(M, K)` | Input matrix |
| **return** | `(M, K)` | `max(0.0, x)` applied element-wise |

---

### `apply_silu(a)`

SiLU (Swish) activation. Computes `x / (1 + exp(-x))` per element.

```
apply_silu(a) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(M, K)` | Input matrix |
| **return** | `(M, K)` | SiLU-activated matrix |

---

### `apply_gelu(a)`

Approximate GELU activation using the tanh approximation:

`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`

Uses the custom `tanh()` function defined in this module.

```
apply_gelu(a) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(M, K)` | Input matrix |
| **return** | `(M, K)` | GELU-activated matrix |

---

## Arithmetic Operations

### `vec_add(a, b)`

Element-wise vector addition.

```
vec_add(a, b) -> Vector
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(K,)` | First vector |
| `b` | `(K,)` | Second vector (must match length of `a`) |
| **return** | `(K,)` | Element-wise sum |

---

### `mat_add(a, b)`

Element-wise matrix addition.

```
mat_add(a, b) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(M, K)` | First matrix |
| `b` | `(M, K)` | Second matrix (must match shape of `a`) |
| **return** | `(M, K)` | Element-wise sum |

Implemented with `zip` for efficiency.

---

### `mat_mul_scalar(a, s)`

Scalar multiplication. Multiplies every element by `s`.

```
mat_mul_scalar(a, s) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(M, K)` | Input matrix |
| `s` | scalar | Multiplier |
| **return** | `(M, K)` | Scaled matrix |

---

### `add_bias(a, bias)`

Adds a bias vector to every row of a matrix.

```
add_bias(a, bias) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(M, K)` | Input matrix |
| `bias` | `(K,)` | Bias vector (length must equal row width) |
| **return** | `(M, K)` | Matrix with bias added to each row |

---

## Matrix Multiplication

### `matmul(a, b)`

General matrix-matrix multiply. Computes `C = A @ B`.

```
matmul(a, b) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(M, K)` | Left operand |
| `b` | `(K, N)` | Right operand |
| **return** | `(M, N)` | Product matrix |

Uses a triple-nested loop with direct indexed access (`a_row[k] * b[k][j]`).

---

### `matmul_t(a, bt)`

Matrix multiply with a pre-transposed right operand. Computes `C = A @ B` where `bt = transpose(B)`.

```
matmul_t(a, bt) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(M, K)` | Left operand |
| `bt` | `(N, K)` | Transposed right operand (each row is a column of the original `B`) |
| **return** | `(M, N)` | Product matrix |

The inner loop calls `_dot(a_row, bt[j])` instead of strided access into `b[k][j]`. This is faster because `bt[j]` is a contiguous row, so the dot product iterates two flat lists in lockstep via `zip`.

---

### `matmul_last_row(a, b)`

Computes only the last row of `A @ B`. Equivalent to `matmul(a, b)[-1]` but avoids computing the other rows.

```
matmul_last_row(a, b) -> Vector
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(M, K)` | Left operand (only the last row is used) |
| `b` | `(K, N)` | Right operand |
| **return** | `(N,)` | Last row of the product |

Useful during output projection when generating one token at a time and only the final token's logits are needed.

---

### `matmul_last_row_t(a, bt)`

Computes only the last row of `A @ B` with pre-transposed `bt = transpose(B)`. Combines the optimizations of `matmul_t` and `matmul_last_row`.

```
matmul_last_row_t(a, bt) -> Vector
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(M, K)` | Left operand (only the last row is used) |
| `bt` | `(N, K)` | Transposed right operand |
| **return** | `(N,)` | Last row of the product |

---

### `vec_matmul(v, m)`

Vector-matrix multiply. Computes `v @ M`.

```
vec_matmul(v, m) -> Vector
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `v` | `(K,)` | Input vector |
| `m` | `(K, N)` | Matrix |
| **return** | `(N,)` | Product vector |

---

### `matvec_mul(m, v)`

Matrix-vector multiply. Computes `M @ v`.

```
matvec_mul(m, v) -> Vector
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `m` | `(M, K)` | Matrix |
| `v` | `(K,)` | Input vector |
| **return** | `(M,)` | Product vector |

---

### `transpose(a)`

Matrix transpose.

```
transpose(a) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(M, K)` | Input matrix |
| **return** | `(K, M)` | Transposed matrix |

Returns `[]` for an empty input.

---

## Choosing a Matmul Variant

| Function | Right operand | Output | When to use |
|----------|---------------|--------|-------------|
| `matmul` | Raw `(K, N)` | Full `(M, N)` matrix | General-purpose, when `B` is not pre-transposed |
| `matmul_t` | Pre-transposed `(N, K)` | Full `(M, N)` matrix | Hot path where `B` can be transposed once and reused (e.g. weight matrices loaded at init time) |
| `matmul_last_row` | Raw `(K, N)` | Single `(N,)` vector | Output projection during token generation — only the last token's row matters |
| `matmul_last_row_t` | Pre-transposed `(N, K)` | Single `(N,)` vector | Same as `matmul_last_row` but with the speed benefit of a pre-transposed weight matrix |

Pre-transposing weight matrices at load time and using the `_t` variants avoids column-stride access in the inner loop, yielding a significant speedup in the pure-Python interpreter.

---

## Normalization

### `layer_norm(x, weight, bias, eps=1e-5)`

Layer normalization (GPT-2 style). Subtracts the mean, divides by standard deviation, then applies an element-wise affine transform.

```
layer_norm(x, weight, bias, eps=1e-5) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `x` | `(M, K)` | Input matrix |
| `weight` | `(K,)` | Scale (gamma) vector |
| `bias` | `(K,)` | Shift (beta) vector |
| `eps` | scalar | Small constant for numerical stability (default `1e-5`) |
| **return** | `(M, K)` | Normalized matrix |

Each row is normalized independently. Computes `mean` and `variance` across the `K` dimension, then applies `(v - mean) / sqrt(var + eps) * weight + bias`.

---

### `rms_norm(x, weight, eps=1e-5)`

Root Mean Square normalization (Llama style). Divides by the RMS value and applies an element-wise scale. No mean subtraction or bias term.

```
rms_norm(x, weight, eps=1e-5) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `x` | `(M, K)` | Input matrix |
| `weight` | `(K,)` | Scale (gamma) vector |
| `eps` | scalar | Small constant for numerical stability (default `1e-5`) |
| **return** | `(M, K)` | Normalized matrix |

Each row is normalized independently. Computes `ss = mean(x^2)`, then applies `v / sqrt(ss + eps) * weight`. Simpler and slightly faster than `layer_norm`.

---

## Softmax

### `softmax(x)`

Numerically stable softmax for a single vector. Subtracts the maximum value before exponentiation to prevent overflow.

```
softmax(x) -> Vector
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `x` | `(K,)` | Input vector |
| **return** | `(K,)` | Probability distribution (sums to 1.0) |

---

### `softmax_rows(x)`

Row-wise softmax for a matrix. Applies `softmax` independently to each row.

```
softmax_rows(x) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `x` | `(M, K)` | Input matrix |
| **return** | `(M, K)` | Matrix where each row is a probability distribution |

---

### `softmax_masked(x, mask)`

Softmax with a binary mask. Positions where `mask[i][j] == 0.0` are zeroed out before normalization. Used for causal attention to prevent attending to future tokens.

```
softmax_masked(x, mask) -> Matrix
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `x` | `(M, K)` | Input matrix (attention scores) |
| `mask` | `(M, K)` | Binary mask (`1.0` = keep, `0.0` = ignore) |
| **return** | `(M, K)` | Masked probability distribution per row |

If a row is entirely masked (`sum_exps == 0`), returns a row of zeros to avoid division by zero.

---

## Internal Helpers

### `tanh(x)`

Hyperbolic tangent, built from `math.exp`. Required because the Scriptling runtime does not expose `math.tanh`.

```
tanh(x) -> float
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `x` | scalar | Input value |
| **return** | scalar | `tanh(x)` |

Clamps at `x >= 20.0` (returns `1.0`) and `x <= -20.0` (returns `-1.0`) to avoid overflow in `exp()`.

---

### `_dot(a, b)`

Dot product of two vectors. Uses `zip` iteration to avoid index-lookup overhead.

```
_dot(a, b) -> float
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `a` | `(K,)` | First vector |
| `b` | `(K,)` | Second vector |
| **return** | scalar | Sum of element-wise products |

Private function. Used internally by `matmul_t` and `matmul_last_row_t`.
