package scriptlingllmlib

import (
	"fmt"
	"testing"
)

func TestTinyLlamaDiag(t *testing.T) {
	path := "models/tinyllama-1.1b-1t-openorca.Q8_0.gguf"
	model, err := LoadGGUF(path)
	if err != nil {
		t.Skip(err)
	}
	fmt.Printf("ChatTemplate: %q\n", model.Tokenizer.ChatTemplate)
	fmt.Printf("Arch: %v\n", model.Metadata["general.architecture"])
	fmt.Printf("Tokens[0]: %q\n", model.Tokenizer.Tokens[0])
	fmt.Printf("Tokens[1]: %q\n", model.Tokenizer.Tokens[1])
	fmt.Printf("Tokens[2]: %q\n", model.Tokenizer.Tokens[2])
	fmt.Printf("NumTokens: %d\n", len(model.Tokenizer.Tokens))
	fmt.Printf("Type: %s\n", model.Tokenizer.Type)
	if len(model.Tokenizer.Tokens) > 32000 {
		fmt.Printf("Tokens[32002]: %q\n", model.Tokenizer.Tokens[32002])
	}
	gguf, _ := LoadGGUF(path)
	gguf.Metadata["_path"] = path
	m, _ := buildInferenceModelF32(gguf, path)
	ids := m.Tokenizer.Encode("<|im_start|>user\nwhat is capital of france?<|im_end|>\n<|im_start|>assistant\n")
	fmt.Printf("ChatML encoded: %v (len=%d)\n", ids[:min(10, len(ids))], len(ids))
	raw := m.Tokenizer.Encode("what is capital of france?")
	fmt.Printf("Raw encoded: %v (len=%d)\n", raw, len(raw))
}
