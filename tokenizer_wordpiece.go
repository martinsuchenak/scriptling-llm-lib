package scriptlingllmlib

import (
	"strings"
	"unicode"
)

// wordPiece is the BERT WordPiece tokenizer used by encoder embedding models
// (all-MiniLM, BGE, E5, …). It lowercases (these are uncased models), splits on
// whitespace and punctuation, then greedily matches the longest subword in the
// vocab, using "##" for continuations. [CLS]/[SEP] wrap the sequence.
type wordPiece struct {
	vocab         map[string]int
	unk, cls, sep int
	// metaspace selects the vocab convention. Some BERT GGUFs store the WordPiece
	// vocab with the SentencePiece "▁" marker on word-initial pieces and bare
	// continuations (metaspace=true); others use bare word-initial pieces and
	// "##" continuations (classic WordPiece, metaspace=false).
	metaspace bool
}

const metaspaceMark = "▁" // ▁

func newWordPiece(vocab map[string]int, cls, sep, unk int) *wordPiece {
	w := &wordPiece{vocab: vocab, cls: cls, sep: sep, unk: unk}
	for _, probe := range []string{metaspaceMark + "the", metaspaceMark + "a", metaspaceMark + "of"} {
		if _, ok := vocab[probe]; ok {
			w.metaspace = true
			break
		}
	}
	return w
}

// encode returns the token ids for text, including the leading [CLS] and
// trailing [SEP].
func (w *wordPiece) encode(text string) []int {
	ids := make([]int, 0, 16)
	ids = append(ids, w.cls)
	for _, tok := range basicTokenize(text) {
		ids = append(ids, w.wordpiece(tok)...)
	}
	ids = append(ids, w.sep)
	return ids
}

// basicTokenize lowercases and splits on whitespace, with each punctuation
// character becoming its own token (matching BERT's BasicTokenizer).
func basicTokenize(text string) []string {
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range strings.ToLower(text) {
		switch {
		case unicode.IsSpace(r):
			flush()
		case isBertPunct(r):
			flush()
			out = append(out, string(r))
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return out
}

// wordpiece greedily splits one whitespace-delimited token into subword ids.
func (w *wordPiece) wordpiece(word string) []int {
	rs := []rune(word)
	if len(rs) > 100 {
		return []int{w.unk}
	}
	ids := make([]int, 0, 4)
	start := 0
	for start < len(rs) {
		end := len(rs)
		cur := -1
		for start < end {
			sub := string(rs[start:end])
			if w.metaspace {
				if start == 0 {
					sub = metaspaceMark + sub // word-initial piece
				}
			} else if start > 0 {
				sub = "##" + sub // WordPiece continuation
			}
			if id, ok := w.vocab[sub]; ok {
				cur = id
				break
			}
			end--
		}
		if cur < 0 {
			return []int{w.unk} // any unmatchable piece -> whole word is [UNK]
		}
		ids = append(ids, cur)
		start = end
	}
	return ids
}

// isBertPunct matches BERT's _is_punctuation: the ASCII punctuation ranges plus
// any Unicode punctuation category.
func isBertPunct(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}
