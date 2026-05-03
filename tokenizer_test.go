package scriptlingllmlib

import (
	"testing"
)

func TestTokenizerSimpleVocab(t *testing.T) {
	vocab := map[string]int{
		"hello": 0,
		"world": 1,
		" ":     2,
		"!":     3,
		"<unk>": 4,
	}
	special := map[string]int{"<unk>": 4}

	tok := NewTokenizer(vocab, nil, special)

	decoded := tok.Decode([]int{0, 2, 1, 3})
	if decoded != "hello world!" {
		t.Errorf("decode = %q, want 'hello world!'", decoded)
	}

	decoded2 := tok.Decode([]int{0})
	if decoded2 != "hello" {
		t.Errorf("decode [0] = %q, want 'hello'", decoded2)
	}
}

func TestTokenizerBPESentencepiece(t *testing.T) {
	vocab := map[string]int{
		" h":      0,
		" e":      1,
		" l":      2,
		" lo":     3,
		" hel":    4,
		" hello":  5,
		" w":      6,
		" wo":     7,
		" wor":    8,
		" world":  9,
		"<unk>":   10,
	}
	special := map[string]int{"<unk>": 10}
	merges := [][2]string{
		{" h", " e"},
		{" he", " l"},
		{" hel", " lo"},
		{" w", " o"},
		{" wo", " r"},
	}

	tok := NewTokenizer(vocab, merges, special)

	if !tok.IsSP {
		t.Fatal("expected IsSP=true")
	}

	ids := tok.Encode("hello")
	if len(ids) == 0 {
		t.Fatal("expected non-empty encoding")
	}

	last := ids[len(ids)-1]
	if last == special["<unk>"] {
		t.Error("last token should not be <unk>, BPE merges may not be working")
	}

	ids = tok.Encode("world")
	if len(ids) == 0 {
		t.Fatal("expected non-empty encoding for 'world'")
	}
}

func TestTokenizerBPEGPT2(t *testing.T) {
	vocab := map[string]int{
		"Ġ":      0,
		"Ġh":     1,
		"Ġe":     2,
		"Ġl":     3,
		"Ġlo":    4,
		"Ġhel":   5,
		"Ġhello": 6,
		"<unk>":  7,
	}
	special := map[string]int{"<unk>": 7}
	merges := [][2]string{
		{"Ġ", "h"},
		{"Ġh", "e"},
		{"Ġhe", "l"},
		{"Ġhel", "lo"},
	}

	tok := NewTokenizer(vocab, merges, special)

	if !tok.IsGPT2Byte {
		t.Fatal("expected IsGPT2Byte=true")
	}
}

func TestTokenizerEmptyInput(t *testing.T) {
	vocab := map[string]int{"a": 0}
	special := map[string]int{"<unk>": 1}
	tok := NewTokenizer(vocab, nil, special)

	ids := tok.Encode("")
	if ids != nil {
		t.Errorf("expected nil for empty input, got %v", ids)
	}
}

func TestTokenizerSpecialTokens(t *testing.T) {
	vocab := map[string]int{
		"hello":        0,
		"<|im_start|>": 1,
		"<|im_end|>":   2,
		"<unk>":        3,
	}
	special := map[string]int{
		"<|im_start|>": 1,
		"<|im_end|>":   2,
		"<unk>":        3,
	}

	tok := NewTokenizer(vocab, nil, special)

	if tok.EOSID != 2 {
		t.Errorf("EOSID = %d, want 2 (im_end overrides </s>)", tok.EOSID)
	}

	ids := tok.Encode("hello<|im_start|>hello")
	if len(ids) != 3 {
		t.Fatalf("expected 3 tokens, got %d: %v", len(ids), ids)
	}
	if ids[1] != 1 {
		t.Errorf("middle token = %d, want 1 (<|im_start|>)", ids[1])
	}
}

func TestTokenizerDecodeSkipSpecial(t *testing.T) {
	vocab := map[string]int{
		"hello": 0,
		"world": 1,
		"<s>":   2,
		"</s>":  3,
	}
	special := map[string]int{"<s>": 2, "</s>": 3}

	tok := NewTokenizer(vocab, nil, special)

	decoded := tok.Decode([]int{2, 0, 1, 3})
	if decoded != "helloworld" {
		t.Errorf("decode = %q, want 'helloworld'", decoded)
	}
}

func TestTokenizerSPSpaces(t *testing.T) {
	vocab := map[string]int{
		"▁hello": 0,
		"▁world": 1,
	}
	special := map[string]int{}

	tok := NewTokenizer(vocab, nil, special)

	decoded := tok.Decode([]int{0, 1})
	if decoded != "hello world" {
		t.Errorf("decode = %q, want 'hello world'", decoded)
	}
}

func TestTokenizerRoundtrip(t *testing.T) {
	vocab := map[string]int{
		" hello": 0,
		" world": 1,
		"<unk>":  2,
	}
	special := map[string]int{"<unk>": 2}

	tok := NewTokenizer(vocab, nil, special)

	if !tok.IsSP {
		t.Fatal("expected IsSP=true for space-prefixed vocab")
	}

	ids := tok.Encode("hello")
	if len(ids) != 1 || ids[0] != 0 {
		t.Errorf("encode 'hello' = %v, want [0]", ids)
	}
}

func TestTokenizerDataStructure(t *testing.T) {
	td := &TokenizerData{
		Tokens:  []string{"<pad>", "<s>", "</s>", "hello", "world"},
		Scores:  []float64{0, 0, 0, -1.0, -1.5},
		Special: map[string]int{"<pad>": 0, "<s>": 1, "</s>": 2},
		Vocab:   map[string]int{"<pad>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4},
		Type:    "bpe",
	}
	if len(td.Tokens) != len(td.Scores) {
		t.Error("Tokens and Scores must have same length")
	}
	if td.Type != "bpe" {
		t.Errorf("Type = %q, want 'bpe'", td.Type)
	}
}

func TestTokenizerDataVocabConsistency(t *testing.T) {
	tokens := []string{"<pad>", "a", "b", "c"}
	scores := []float64{0, -1, -2, -3}
	vocab := make(map[string]int, len(tokens))
	for i, tok := range tokens {
		vocab[tok] = i
	}

	td := &TokenizerData{
		Tokens: tokens,
		Scores: scores,
		Vocab:  vocab,
	}

	for i, tok := range td.Tokens {
		if td.Vocab[tok] != i {
			t.Errorf("vocab[%q] = %d, want %d", tok, td.Vocab[tok], i)
		}
	}
}
