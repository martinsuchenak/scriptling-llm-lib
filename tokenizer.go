package scriptlingllmlib

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

type Tokenizer struct {
	TokenToID   map[string]int
	IDToToken   map[int]string
	MergeRank   map[string]int
	Special     map[string]int
	EOSID       int
	IsSP        bool
	IsSPUnicode bool
	IsGPT2Byte  bool
	ByteEncoder map[int]rune
	ByteDecoder map[rune]int

	sortedSpecial []string
	pattern       *regexp.Regexp
}

func NewTokenizer(vocab map[string]int, merges [][2]string, special map[string]int) *Tokenizer {
	t := &Tokenizer{
		TokenToID: vocab,
		IDToToken: make(map[int]string, len(vocab)),
		MergeRank: make(map[string]int, len(merges)),
		Special:   special,
	}

	for k, v := range vocab {
		t.IDToToken[v] = k
	}

	for rank, pair := range merges {
		key := pair[0] + "|" + pair[1]
		t.MergeRank[key] = rank
	}

	t.EOSID = special["</s>"]
	if id, ok := special["<|im_end|>"]; ok {
		t.EOSID = id
	}

	t.sortedSpecial = make([]string, 0, len(special))
	for k := range special {
		t.sortedSpecial = append(t.sortedSpecial, k)
	}
	sort.Slice(t.sortedSpecial, func(i, j int) bool {
		return len(t.sortedSpecial[i]) > len(t.sortedSpecial[j])
	})

	t.IsSP = false
	t.IsGPT2Byte = false
	t.IsSPUnicode = false
	hasSP := false
	hasGPT2 := false
	for k := range vocab {
		if len(k) > 1 && k[0] == ' ' {
			t.IsSP = true
			break
		}
		if strings.HasPrefix(k, "▁") {
			hasSP = true
		}
		if strings.Contains(k, "Ġ") {
			hasGPT2 = true
		}
	}
	if !t.IsSP && hasSP {
		t.IsSP = true
		t.IsSPUnicode = true
	}
	if !t.IsSP && hasGPT2 {
		t.IsGPT2Byte = true
	}

	t.pattern = regexp.MustCompile(`'(?:s|t|re|ve|m|ll|d)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+`)

	t.buildByteEncoder()

	return t
}

func (t *Tokenizer) buildByteEncoder() {
	t.ByteEncoder = make(map[int]rune, 256)
	t.ByteDecoder = make(map[rune]int, 256)

	bs := make([]int, 0, 256)
	for b := 33; b < 127; b++ {
		bs = append(bs, b)
	}
	for b := 161; b < 173; b++ {
		bs = append(bs, b)
	}
	for b := 174; b < 256; b++ {
		bs = append(bs, b)
	}

	cs := make([]int, len(bs))
	copy(cs, bs)

	n := 0
	for b := 0; b < 256; b++ {
		found := false
		for _, v := range bs {
			if v == b {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}

	for i := range bs {
		t.ByteEncoder[bs[i]] = rune(cs[i])
		t.ByteDecoder[rune(cs[i])] = bs[i]
	}
}

func (t *Tokenizer) Encode(text string) []int {
	if text == "" {
		return nil
	}

	if len(t.sortedSpecial) > 0 {
		return t.encodeWithSpecial(text)
	}

	if t.IsSP {
		return t.encodeSentencepiece(text)
	}

	return t.encodeGPT2(text)
}

func (t *Tokenizer) encodeWithSpecial(text string) []int {
	var ids []int
	remaining := text

	for len(remaining) > 0 {
		bestPos := -1
		bestTok := ""
		for _, st := range t.sortedSpecial {
			pos := strings.Index(remaining, st)
			if pos != -1 && (bestPos == -1 || pos < bestPos) {
				bestPos = pos
				bestTok = st
			}
		}

		if bestTok == "" {
			ids = append(ids, t.encodeText(remaining)...)
			break
		}

		if bestPos > 0 {
			ids = append(ids, t.encodeText(remaining[:bestPos])...)
		}
		ids = append(ids, t.Special[bestTok])
		remaining = remaining[bestPos+len(bestTok):]
	}

	return ids
}

func (t *Tokenizer) encodeText(text string) []int {
	if t.IsSP {
		return t.encodeSentencepiece(text)
	}
	return t.encodeGPT2(text)
}

func (t *Tokenizer) encodeSentencepiece(text string) []int {
	text = " " + text
	if t.IsSPUnicode {
		text = strings.ReplaceAll(text, " ", "▁")
	}
	if id, ok := t.TokenToID[text]; ok {
		return []int{id}
	}

	if len(t.MergeRank) > 0 {
		symbols := make([]string, utf8.RuneCountInString(text))
		i := 0
		for _, r := range text {
			symbols[i] = string(r)
			i++
		}
		symbols = t.bpe(symbols)

		ids := make([]int, len(symbols))
		for i, s := range symbols {
			if id, ok := t.TokenToID[s]; ok {
				ids[i] = id
			} else {
				ids[i] = t.Special["<unk>"]
			}
		}
		return ids
	}

	return t.encodeSentencepieceGreedy(text)
}

func (t *Tokenizer) encodeSentencepieceGreedy(text string) []int {
	var ids []int
	pos := 0
	runes := []rune(text)
	for pos < len(runes) {
		bestLen := 0
		bestID := -1
		for end := min(len(runes), pos+64); end > pos; end-- {
			candidate := string(runes[pos:end])
			if id, ok := t.TokenToID[candidate]; ok {
				bestLen = end - pos
				bestID = id
				break
			}
		}
		if bestLen > 0 {
			ids = append(ids, bestID)
			pos += bestLen
		} else {
			ch := runes[pos]
			hexTok := fmt.Sprintf("<0x%02X>", ch)
			if id, ok := t.TokenToID[hexTok]; ok {
				ids = append(ids, id)
			} else if id, ok := t.Special["<unk>"]; ok {
				ids = append(ids, id)
			}
			pos++
		}
	}
	return ids
}

func (t *Tokenizer) encodeGPT2(text string) []int {
	words := t.pattern.FindAllString(text, -1)
	var ids []int

	for _, word := range words {
		var wordEncoded string
		if t.IsGPT2Byte {
			var buf strings.Builder
			for _, b := range []byte(word) {
				buf.WriteRune(t.ByteEncoder[int(b)])
			}
			wordEncoded = buf.String()
		} else {
			wordEncoded = word
		}

		if id, ok := t.TokenToID[wordEncoded]; ok {
			ids = append(ids, id)
			continue
		}

		symbols := make([]string, utf8.RuneCountInString(wordEncoded))
		i := 0
		for _, r := range wordEncoded {
			symbols[i] = string(r)
			i++
		}
		symbols = t.bpe(symbols)

		for _, s := range symbols {
			if id, ok := t.TokenToID[s]; ok {
				ids = append(ids, id)
			} else {
				ids = append(ids, t.Special["<unk>"])
			}
		}
	}

	return ids
}

func (t *Tokenizer) bpe(symbols []string) []string {
	for len(symbols) >= 2 {
		bestRank := math.MaxInt32
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			key := symbols[i] + "|" + symbols[i+1]
			if rank, ok := t.MergeRank[key]; ok {
				if rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}

		if bestIdx == -1 {
			break
		}

		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, symbols[bestIdx]+symbols[bestIdx+1])
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}

	return symbols
}

func (t *Tokenizer) Decode(ids []int) string {
	var parts []string
	for _, id := range ids {
		tok, ok := t.IDToToken[id]
		if !ok {
			tok = "<unk>"
		}
		if _, isSpecial := t.Special[tok]; isSpecial {
			continue
		}
		parts = append(parts, tok)
	}

	text := strings.Join(parts, "")

	if t.IsGPT2Byte {
		var buf strings.Builder
		for _, ch := range text {
			if b, ok := t.ByteDecoder[ch]; ok {
				buf.WriteRune(rune(b))
			} else {
				buf.WriteRune(ch)
			}
		}
		text = buf.String()
	}

	text = strings.ReplaceAll(text, "\u2581", " ")
	text = strings.TrimSpace(text)

	return text
}
