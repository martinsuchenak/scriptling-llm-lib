import llm

# Activation functions
print("=== Activation Functions ===")
print("sigmoid(0) =", llm.sigmoid(0))
print("relu(-1) =", llm.relu(-1))
print("gelu(1) =", llm.gelu(1))
print("silu(2) =", llm.silu(2))

# Vector operations
print("\n=== Vector Operations ===")
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]
print("vec_add([1,2,3], [4,5,6]) =", llm.vec_add(a, b))
print("vec_scale([1,2,3], 2) =", llm.vec_scale(a, 2.0))
print("vec_apply([-1,0,1,2], relu) =", llm.vec_apply([-1.0, 0.0, 1.0, 2.0], "relu"))

# Inference helpers
print("\n=== Inference Helpers ===")
logits = [0.1, 0.5, 0.3, 0.9, 0.7]
print("logits =", logits)
print("argmax =", llm.argmax(logits))
print("top_k(3) =", llm.top_k(logits, 3))
print("clip =", llm.clip([-2.0, 0.5, 3.0], -1.0, 2.0))

# Transformer layer simulation
print("\n=== Mini Transformer Layer ===")
d_model = 4
seq_len = 2

x = [[0.5, -0.3, 0.8, 0.1], [0.2, 0.4, -0.1, 0.6]]
weight = [1.0, 1.0, 1.0, 1.0]

normed = llm.rms_norm(x, weight)
print("rms_norm =", normed)

q = llm.rope(normed, start_pos=0)
k = llm.rope(normed, start_pos=0)
v = normed
attn = llm.attention(q, k, v, causal=True)
print("attention =", attn)

gate = [[1.0, -0.5, 0.8, 0.3], [-0.2, 0.7, 0.1, -0.4]]
up = [[0.5, 0.9, -0.3, 0.6], [0.8, -0.1, 0.4, 0.2]]
hidden = llm.silu_gate(gate, up)
print("silu_gate =", hidden)

flattened = llm.flatten(hidden)
print("flattened =", flattened)

# Dequantize
print("\n=== Dequantization ===")
data = [10, -5, 20, 15, -100, 50]
scales = [0.1, 0.2, 0.05]
print("dequantize_q8 =", llm.dequantize_q8(data, scales, 2))

print("\nDone!")
