package main

import (
	"fmt"
	"log"

	"github.com/martinsuchenak/scriptling-llm-lib"
	"github.com/paularlott/scriptling"
)

func main() {
	p := scriptling.New()
	p.RegisterLibrary(scriptlingllmlib.Library)

	script := `
import llm

# Activation functions
print("sigmoid(0) =", llm.sigmoid(0))
print("relu(-1) =", llm.relu(-1))
print("gelu(1) =", llm.gelu(1))
print("silu(2) =", llm.silu(2))

# Vector operations
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]
print("vec_add =", llm.vec_add(a, b))
print("vec_sub =", llm.vec_sub(b, a))
print("vec_mul =", llm.vec_mul(a, b))
print("vec_scale =", llm.vec_scale(a, 2.0))
print("vec_apply relu =", llm.vec_apply([-1.0, 0.0, 1.0], "relu"))

# Inference helpers
logits = [0.1, 0.5, 0.3, 0.9, 0.7]
print("argmax =", llm.argmax(logits))
print("argmin =", llm.argmin(logits))
print("clip =", llm.clip([-2.0, 0.5, 3.0], -1.0, 2.0))
print("topk =", llm.topk(logits, 3))
print("top_k =", llm.top_k(logits, 3))

# LLM Primitives
x = [[0.5, -0.3, 0.8], [0.1, 0.2, -0.4]]
w = [1.0, 1.0, 1.0]
print("rms_norm =", llm.rms_norm(x, w))

# RoPE
q = [[1.0, 0.0, 0.0, 1.0]]
print("rope =", llm.rope(q))

# Attention
q_mat = [[1.0, 0.0]]
k_mat = [[1.0, 0.0], [0.0, 1.0]]
v_mat = [[1.0, 0.0], [0.0, 1.0]]
print("attention =", llm.attention(q_mat, k_mat, v_mat))

# Linear
x_in = [[1.0, 2.0]]
weight = [[1.0, 0.0], [0.0, 1.0]]
bias = [10.0, 20.0]
print("linear =", llm.linear(x_in, weight, bias))
print("linear_row =", llm.linear_row(x_in, weight, bias))

# Matrix ops
m = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
print("slice_rows =", llm.slice_rows(m, 0, 2))
print("concat_rows =", llm.concat_rows(m, [[7.0, 8.0]]))
print("flatten =", llm.flatten(m))

# Dequantize
data = [10, -5, 20, 15]
scales = [0.1, 0.2]
print("dequantize_q8 =", llm.dequantize_q8(data, scales, 2))

# SwiGLU
gate = [[1.0, -1.0], [0.0, 2.0]]
up = [[1.0, 1.0], [1.0, 1.0]]
print("silu_gate =", llm.silu_gate(gate, up))

# Version
print("version =", llm.VERSION)
`

	_, err := p.Eval(script)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("\n--- Example complete ---")
}
