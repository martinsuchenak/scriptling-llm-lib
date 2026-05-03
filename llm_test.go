package scriptlingllmlib

import (
	"testing"

	"github.com/paularlott/scriptling/object"
)

func TestLibraryRegistration(t *testing.T) {
	if Library.Name() != "llm" {
		t.Errorf("Library.Name() = %s, want llm", Library.Name())
	}

	funcs := Library.Functions()
	funcCount := len(funcs)
	if funcCount != 55 {
		t.Errorf("Library has %d functions, want 55", funcCount)
	}

	required := []string{
		"argmax", "argmin", "topk", "clip",
		"sigmoid", "relu", "gelu", "silu",
		"vec_add", "vec_sub", "vec_mul", "vec_scale", "vec_apply",
		"rms_norm", "rope", "silu_gate", "attention", "linear", "linear_row",
		"linear_q8", "linear_row_q8", "linear_q4", "linear_row_q4",
		"linear_q4_k", "linear_row_q4_k", "linear_q5", "linear_row_q5",
		"linear_q6_k", "linear_row_q6_k",
		"top_k", "dequantize_q8", "dequantize_q8_0", "dequantize_q4_0",
		"dequantize_q4_k", "dequantize_q5_0", "dequantize_q6_k",
		"sample", "split_heads", "merge_heads", "repeat_kv",
		"concat_rows", "slice_rows", "flatten", "reshape", "zeros", "arange",
		"quantize_q8", "quantize_q8_rows",
		"output_logits", "fused_qkv", "fused_ffn", "fused_block",
		"fused_rope_batch", "fused_attention", "generate",
	}
	for _, name := range required {
		if _, ok := funcs[name]; !ok {
			t.Errorf("missing function: %s", name)
		}
	}

	consts := Library.Constants()
	if v, ok := consts["VERSION"]; !ok {
		t.Error("missing VERSION constant")
	} else {
		if v.(*object.String).Value != "1.1.0" {
			t.Errorf("VERSION = %s, want 1.1.0", v.(*object.String).Value)
		}
	}
}
