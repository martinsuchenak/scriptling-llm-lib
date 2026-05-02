# Tokenizer API Reference

Two tokenizer implementations are provided in `tokenizer.py`. Both read vocabulary from a `dict` mapping token strings to integer IDs. Special tokens follow the convention: `<pad>` = 0, `<s>` = 1, `</s>` = 2, `<unk>` = 3.

Both classes are designed to run inside the Scriptling sandbox -- no type annotations, no external dependencies, and only RE2-compatible regex patterns.

---

## SimpleTokenizer

Whitespace-split word-level tokenizer intended for tiny test models. Each word is looked up directly in the vocabulary. Unknown words map to `<unk>`.

### Constructor

```python
SimpleTokenizer(vocab, special)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `vocab` | `dict[str, int]` | Mapping from token string to integer ID |
| `special` | `dict[str, int]` | Mapping from special token name to ID (e.g. `{"<unk>": 3, "</s>": 2}`) |

### Methods

#### `encode(text)`

Splits `text` on spaces, looks up each word in the vocabulary, and returns the list of integer IDs. Words not found in the vocabulary are mapped to the `<unk>` ID.

```python
tok = SimpleTokenizer(vocab={"hello": 4, "world": 5}, special={"<unk>": 3})
tok.encode("hello world")
# [4, 5]

tok.encode("hello unknown_word")
# [4, 3]
```

#### `decode(ids)`

Maps each integer ID back to its token string and joins them with spaces.

```python
tok.decode([4, 5])
# "hello world"

tok.decode([4, 999])
# "hello <unk>"
```

### Attributes

| Attribute | Description |
|-----------|-------------|
| `token_to_id` | The vocab dict passed at construction |
| `id_to_token` | Reverse lookup: int ID to token string |
| `special_tokens` | The special tokens dict passed at construction |
| `unk_id` | Integer ID for `<unk>` (default 0) |
| `eos_id` | Integer ID for `</s>` (default 2) |

---

## BPETokenizer

Byte Pair Encoding subword tokenizer supporting both GPT-2 and Llama/sentencepiece conventions. It auto-detects which style to use based on the vocabulary contents.

### BPE Algorithm

Byte Pair Encoding is a subword tokenization method that iteratively merges the most frequent adjacent pair of symbols. The merge rules are learned during training and stored as an ordered list of pairs. At encoding time, the algorithm applies these merges greedily in priority order:

1. Start with the input split into individual characters.
2. Scan all adjacent symbol pairs and find the one with the lowest merge rank (highest priority).
3. If no mergeable pair is found, stop.
4. Combine the winning pair into a single symbol.
5. Repeat from step 2.

This produces variable-length subword tokens that balance vocabulary size with the ability to encode any input.

Merge pairs are stored as `"left|right"` string keys rather than tuple keys, for Scriptling compatibility.

### Constructor

```python
BPETokenizer(vocab, merges, special)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `vocab` | `dict[str, int]` | Mapping from token string to integer ID |
| `merges` | `list[list[str]]` | Ordered list of merge pairs, e.g. `[["h", "e"], ["he", "l"], ...]`. Order defines priority: earlier pairs have lower rank and are applied first. |
| `special` | `dict[str, int]` | Mapping from special token name to ID |

The constructor builds an internal `merge_rank` dict mapping `"left|right"` keys to their integer rank. It also auto-detects sentencepiece style by checking whether any vocabulary key starts with a space and has length greater than one.

### Methods

#### `encode(text)`

Encodes `text` into a list of integer IDs. Dispatches to `_encode_sentencepiece` or `_encode_gpt2` based on the auto-detected tokenizer style. Returns an empty list for empty input.

#### `decode(ids)`

Maps integer IDs back to token strings, joins them, and handles special tokens and character conventions:

- Tokens enclosed in angle brackets (e.g. `<s>`, `</s>`, `<pad>`) are treated as special tokens and excluded from the output.
- The Unicode character `\u2581` (solid block, used as a word-boundary marker in some tokenizers) is replaced with a space.
- Leading and trailing whitespace is stripped.

```python
tok = BPETokenizer(vocab={"<unk>": 3, " h": 10, "ello": 11, " hello": 12},
                   merges=[[" ", "h"], ["h", "ello"]],
                   special={"<unk>": 3, "</s>": 2})
tok.encode("hello")
# [12]  or [10, 11] depending on merges applied

tok.decode([12])
# "hello"
```

### Internal Methods

#### `_encode_sentencepiece(text)`

Sentencepiece encoding (Llama style):

1. Prepends a space to the input text.
2. Checks if the full text exists in the vocabulary directly. If so, returns its ID.
3. Otherwise splits the text into individual characters.
4. Applies BPE merges via `_bpe`.
5. Maps each resulting symbol to its token ID (unknown symbols become `<unk>`).

#### `_encode_gpt2(text)`

GPT-2 style encoding:

1. Pre-tokenizes the text using a regex pattern that splits on contractions, letter runs, digit runs, punctuation, and whitespace.
2. For each pre-tokenized word, checks for a direct vocabulary match.
3. If no direct match, splits the word into characters and applies BPE via `_bpe`.
4. Maps each resulting symbol to its token ID.

The regex pattern used is RE2-compatible (no lookahead or lookbehind):

```
's|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+
```

#### `_bpe(symbols)`

Core BPE merge loop. Takes a list of character strings and repeatedly applies the highest-priority merge until no more merges are possible:

1. Scan all adjacent pairs `(symbols[i], symbols[i+1])`.
2. For each pair, look up `"left|right"` in `merge_rank`. The pair with the lowest rank value wins.
3. If no pair is found in `merge_rank`, stop.
4. Replace the winning pair at position `best_idx` with the concatenated string `symbols[best_idx] + symbols[best_idx+1]`.
5. Repeat.

Returns the list of merged symbols.

### Attributes

| Attribute | Description |
|-----------|-------------|
| `token_to_id` | The vocab dict passed at construction |
| `id_to_token` | Reverse lookup: int ID to token string |
| `merges` | The raw merge list passed at construction |
| `merge_rank` | `dict[str, int]` mapping `"left\|right"` to rank (0 = highest priority) |
| `special_tokens` | The special tokens dict passed at construction |
| `unk_id` | Integer ID for `<unk>` (default 0) |
| `eos_id` | Integer ID for `</s>` (default 2) |
| `is_sentencepiece` | `True` if vocabulary contains space-prefixed tokens (Llama style) |

---

## Vocabulary Format

Both tokenizers expect the same vocabulary format: a `dict` mapping token strings to integer IDs.

```python
vocab = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    " the": 4,
    " is": 5,
    "hello": 6,
}
```

For `BPETokenizer`, the `merges` parameter is an ordered list of two-element lists:

```python
merges = [
    [" ", "t"],      # rank 0: merge " " + "t" -> " t"
    [" t", "h"],     # rank 1: merge " t" + "h" -> " th"
    [" th", "e"],    # rank 2: merge " th" + "e" -> " the"
]
```

Lower rank values indicate higher priority. When multiple merge pairs are present in the symbol sequence, the pair with the lowest rank is merged first.

## BPE Encoding Example

Given the vocabulary and merges above, encoding `"the"` with sentencepiece style proceeds as follows:

**Input:** `"the"` (prepended space becomes `" the"`)

**Step 1 -- character split:** `[" ", "t", "h", "e"]`

**Step 2 -- first merge scan:** Adjacent pairs are `" |t"`, `"t|h"`, `"h|e"`. Only `" |t"` is in `merge_rank` (rank 0). Merge to produce `[" t", "h", "e"]`.

**Step 3 -- second merge scan:** Adjacent pairs are `" t|h"`, `"h|e"`. Pair `" t|h"` has rank 1. Merge to produce `[" th", "e"]`.

**Step 4 -- third merge scan:** Adjacent pairs are `" th|e"`. Pair `" th|e"` has rank 2. Merge to produce `[" the"]`.

**Step 5 -- no more pairs:** Single symbol remaining. Look up `" the"` in vocabulary to get ID 4.

**Result:** `[4]`
