package scriptlingllmlib

import (
	"fmt"
	"math"
	"sync"
)

// InferenceBert is a BERT-style bidirectional encoder, used by sentence-embedding
// models. It handles two GGUF architectures:
//   - "bert" (all-MiniLM, BGE, E5, GTE): learned position embeddings, separate
//     q/k/v with biases, plain GELU MLP.
//   - "nomic-bert" (nomic-embed-text): RoPE instead of positions, a fused QKV
//     matrix with no biases, and a gated GeGLU FFN.
//
// There is no causal mask or KV cache: the whole sequence is encoded in one pass
// and pooled into a single vector. The weight matrices reuse the quantized matmul
// kernels.
type InferenceBert struct {
	Dim, NHeads, NLayers, FFN, MaxPos int
	Eps                               float32
	pooling                           string // "mean" (default) or "cls"

	useRope     bool
	ropeNeox    bool
	ropeFreqs   []float32
	ropeHalfDim int
	actMode     int // FFN activation: 0 = GELU (plain BERT), 2 = SiLU (gated/SwiGLU)

	TokenEmb  []float32 // [vocab][dim] dense
	PosEmb    []float32 // [maxpos][dim]; nil when useRope
	TypeEmb   []float32 // [2][dim] (type 0 used for single-sentence input)
	VocabSize int

	EmbNormW, EmbNormB []float32
	Layers             []bertLayer
	Tok                *wordPiece
}

type bertLayer struct {
	Wqkv                 interface{} // fused QKV (nomic); nil => separate Wq/Wk/Wv
	Wq, Wk, Wv, Wo       interface{}
	Bq, Bk, Bv, Bo       []float32 // nil where the model has no biases
	AttnNormW, AttnNormB []float32
	Wgate                interface{} // gated FFN (nomic); nil => plain GELU MLP
	Wup, Wdown           interface{}
	Bup, Bdown           []float32
	OutNormW, OutNormB   []float32
}

var bertCache = struct {
	mu sync.Mutex
	m  map[string]*InferenceBert
}{m: map[string]*InferenceBert{}}

func getOrLoadBert(path string) (*InferenceBert, error) {
	bertCache.mu.Lock()
	defer bertCache.mu.Unlock()
	if b, ok := bertCache.m[path]; ok {
		return b, nil
	}
	gguf, err := LoadGGUF(path)
	if err != nil {
		return nil, err
	}
	gguf.Metadata["_path"] = path
	b, err := buildBertModel(gguf)
	gguf.ReleaseFileData()
	if err != nil {
		return nil, err
	}
	bertCache.m[path] = b
	return b, nil
}

func buildBertModel(g *GGUFModel) (*InferenceBert, error) {
	md := g.Metadata
	arch := metaString(md, "general.architecture", "bert")
	p := arch + "." // "bert." or "nomic-bert."
	dim := metaInt(md, p+"embedding_length", 0)
	nLayers := metaInt(md, p+"block_count", 0)
	if dim == 0 || nLayers == 0 {
		return nil, fmt.Errorf("%s: missing embedding_length/block_count", arch)
	}
	m := &InferenceBert{
		Dim:     dim,
		NHeads:  metaInt(md, p+"attention.head_count", 12),
		NLayers: nLayers,
		FFN:     metaInt(md, p+"feed_forward_length", 4*dim),
		MaxPos:  metaInt(md, p+"context_length", 512),
		Eps:     float32(metaFloat(md, p+"attention.layer_norm_epsilon", 1e-12)),
		pooling: "mean",
	}
	if metaInt(md, p+"pooling_type", 1) == 2 {
		m.pooling = "cls"
	}
	if freqBase := metaFloat(md, p+"rope.freq_base", 0); freqBase > 0 {
		m.useRope = true
		m.ropeNeox = true // nomic-bert uses NEOX (rotate-half) rotary
		headDim := dim / m.NHeads
		ropeDim := metaInt(md, p+"rope.dimension_count", headDim)
		m.ropeHalfDim = ropeDim / 2
		m.ropeFreqs = make([]float32, m.ropeHalfDim)
		for i := 0; i < m.ropeHalfDim; i++ {
			m.ropeFreqs[i] = float32(1.0 / math.Pow(freqBase, 2.0*float64(i)/float64(ropeDim)))
		}
	}

	var err error
	emb, vocab, _, err := g.loadTensor2DF32("token_embedding.weight")
	if err != nil {
		return nil, fmt.Errorf("%s: token_embedding: %w", arch, err)
	}
	m.TokenEmb, m.VocabSize = emb, vocab
	if !m.useRope {
		if m.PosEmb, _, _, err = g.loadTensor2DF32("position_embd.weight"); err != nil {
			return nil, fmt.Errorf("%s: position_embd: %w", arch, err)
		}
	}
	m.TypeEmb, _, _, _ = g.loadTensor2DF32("token_types.weight")
	m.EmbNormW, _ = g.loadTensor1DF32("token_embd_norm.weight")
	m.EmbNormB, _ = g.loadTensor1DF32("token_embd_norm.bias")

	m.Layers = make([]bertLayer, nLayers)
	for i := 0; i < nLayers; i++ {
		w := fmt.Sprintf("blocks.%d.", i) // mapped weight names
		b := fmt.Sprintf("blk.%d.", i)    // raw bias/norm/fused names
		L := &m.Layers[i]

		if qkv, qerr := g.loadWeightF32Direct(b + "attn_qkv.weight"); qerr == nil && qkv != nil {
			L.Wqkv = qkv
		} else {
			if L.Wq, err = g.loadWeightF32Direct(w + "attn.w_q.weight"); err != nil {
				return nil, fmt.Errorf("%s: layer %d q: %w", arch, i, err)
			}
			L.Wk, _ = g.loadWeightF32Direct(w + "attn.w_k.weight")
			L.Wv, _ = g.loadWeightF32Direct(w + "attn.w_v.weight")
			L.Bq, _ = g.loadTensor1DF32(b + "attn_q.bias")
			L.Bk, _ = g.loadTensor1DF32(b + "attn_k.bias")
			L.Bv, _ = g.loadTensor1DF32(b + "attn_v.bias")
		}
		L.Wo, _ = g.loadWeightF32Direct(w + "attn.w_o.weight")
		L.Bo, _ = g.loadTensor1DF32(b + "attn_output.bias")
		L.AttnNormW, _ = g.loadTensor1DF32(b + "attn_output_norm.weight")
		L.AttnNormB, _ = g.loadTensor1DF32(b + "attn_output_norm.bias")

		L.Wgate, _ = g.loadWeightF32Direct(w + "ffn.w_gate.weight") // nil if absent
		L.Wup, _ = g.loadWeightF32Direct(w + "ffn.w_up.weight")
		L.Wdown, _ = g.loadWeightF32Direct(w + "ffn.w_down.weight")
		L.Bup, _ = g.loadTensor1DF32(b + "ffn_up.bias")
		L.Bdown, _ = g.loadTensor1DF32(b + "ffn_down.bias")
		L.OutNormW, _ = g.loadTensor1DF32(b + "layer_output_norm.weight")
		L.OutNormB, _ = g.loadTensor1DF32(b + "layer_output_norm.bias")
	}

	// A gated FFN (Wgate present) is SwiGLU (SiLU activation); a plain up→down FFN
	// is GELU. Verified against llama.cpp: all-MiniLM (plain/GELU) matches at
	// cosine 0.999, nomic (gated/SiLU) at 0.996.
	if m.Layers[0].Wgate != nil {
		m.actMode = 2 // SiLU
	}

	if g.Tokenizer == nil || len(g.Tokenizer.Vocab) == 0 {
		return nil, fmt.Errorf("%s: missing tokenizer vocab", arch)
	}
	m.Tok = newWordPiece(
		g.Tokenizer.Vocab,
		metaInt(md, "tokenizer.ggml.cls_token_id", 101),
		metaInt(md, "tokenizer.ggml.seperator_token_id", 102),
		metaInt(md, "tokenizer.ggml.unknown_token_id", 100),
	)
	return m, nil
}

// embed encodes a single text and returns the pooled (optionally L2-normalized)
// vector. It is a thin wrapper over the batched forward path.
func (m *InferenceBert) embed(text, pooling string, normalize bool) []float32 {
	if pooling == "" {
		pooling = m.pooling
	}
	x, offsets, lengths := m.forward([][]int{m.Tok.encode(text)})
	return m.pool(x, offsets[0], lengths[0], pooling, normalize)
}

// embedBatch encodes many texts in a single forward pass and returns one pooled
// vector per text, in input order. Packing the sequences into shared matmuls
// reads each weight matrix once for the whole batch (instead of once per text),
// which is the dominant cost for larger encoders — so batched throughput is far
// higher than calling embed in a loop. Like embed it mutates no shared state, so
// concurrent calls on the same model are safe.
func (m *InferenceBert) embedBatch(texts []string, pooling string, normalize bool) [][]float32 {
	if pooling == "" {
		pooling = m.pooling
	}
	if len(texts) == 0 {
		return nil
	}
	seqs := make([][]int, len(texts))
	for i, t := range texts {
		seqs[i] = m.Tok.encode(t)
	}
	x, offsets, lengths := m.forward(seqs)
	out := make([][]float32, len(texts))
	for s := range texts {
		out[s] = m.pool(x, offsets[s], lengths[s], pooling, normalize)
	}
	return out
}

// forward runs the encoder over a batch of token-id sequences and returns the
// final hidden states for every token, flattened as [totalTokens][dim], along
// with each sequence's start offset (in tokens) and length.
//
// All sequences are concatenated into one [T][dim] activation so every linear
// layer is a single matmul over T rows — each weight is streamed from memory once
// per batch, and T large enough also crosses the parallel-matmul threshold so one
// call saturates the cores. Attention and RoPE are applied per sequence (block
// diagonal) so tokens never attend across sequence boundaries; everything else
// (embeddings, LayerNorm, residuals, FFN activation) is row-wise and batches
// transparently.
func (m *InferenceBert) forward(seqs [][]int) (x []float32, offsets, lengths []int) {
	dim := m.Dim
	headDim := dim / m.NHeads
	nSeq := len(seqs)

	offsets = make([]int, nSeq)
	lengths = make([]int, nSeq)
	T := 0
	for s, ids := range seqs {
		offsets[s] = T
		lengths[s] = len(ids)
		T += len(ids)
	}
	if T == 0 {
		return nil, offsets, lengths
	}

	// Embeddings: token + (position if no RoPE) + token-type(0). Position is the
	// index within each sequence, so it resets at every sequence boundary.
	x = make([]float32, T*dim)
	for s, ids := range seqs {
		base := offsets[s]
		for i, id := range ids {
			if id >= m.VocabSize {
				id = 0
			}
			to, te := (base+i)*dim, id*dim
			for d := 0; d < dim; d++ {
				x[to+d] = m.TokenEmb[te+d]
				if m.TypeEmb != nil {
					x[to+d] += m.TypeEmb[d]
				}
			}
			if !m.useRope && m.PosEmb != nil {
				pos := i
				if pos >= m.MaxPos {
					pos = m.MaxPos - 1
				}
				pe := pos * dim
				for d := 0; d < dim; d++ {
					x[to+d] += m.PosEmb[pe+d]
				}
			}
		}
	}
	bertLayerNorm(x, T, dim, m.EmbNormW, m.EmbNormB, m.Eps)

	q := make([]float32, T*dim)
	k := make([]float32, T*dim)
	v := make([]float32, T*dim)
	qkv := make([]float32, T*3*dim)
	att := make([]float32, T*dim)
	o := make([]float32, T*dim)
	up := make([]float32, T*m.FFN)
	gate := make([]float32, T*m.FFN)
	down := make([]float32, T*dim)

	for li := range m.Layers {
		L := &m.Layers[li]
		if L.Wqkv != nil {
			bertLinear(L.Wqkv, x, T, dim, 3*dim, nil, qkv)
			for i := 0; i < T; i++ {
				src, dq := i*3*dim, i*dim
				copy(q[dq:dq+dim], qkv[src:src+dim])
				copy(k[dq:dq+dim], qkv[src+dim:src+2*dim])
				copy(v[dq:dq+dim], qkv[src+2*dim:src+3*dim])
			}
		} else {
			bertLinear(L.Wq, x, T, dim, dim, L.Bq, q)
			bertLinear(L.Wk, x, T, dim, dim, L.Bk, k)
			bertLinear(L.Wv, x, T, dim, dim, L.Bv, v)
		}
		// RoPE and attention are per sequence (block diagonal): slice each
		// sequence's rows so positions reset and tokens stay within their sequence.
		for s := 0; s < nSeq; s++ {
			n := lengths[s]
			off := offsets[s] * dim
			if m.useRope {
				bertApplyRope(q[off:], n, m.NHeads, headDim, m.ropeFreqs, m.ropeHalfDim, m.ropeNeox)
				bertApplyRope(k[off:], n, m.NHeads, headDim, m.ropeFreqs, m.ropeHalfDim, m.ropeNeox)
			}
			bertAttention(q[off:], k[off:], v[off:], n, m.NHeads, headDim, att[off:])
		}
		bertLinear(L.Wo, att, T, dim, dim, L.Bo, o)
		for i := range x {
			x[i] += o[i]
		}
		bertLayerNorm(x, T, dim, L.AttnNormW, L.AttnNormB, m.Eps)

		if L.Wgate != nil {
			// Gated GeGLU FFN: act(x·Wgate) ⊙ (x·Wup), then ·Wdown.
			bertLinear(L.Wgate, x, T, dim, m.FFN, nil, gate)
			bertLinear(L.Wup, x, T, dim, m.FFN, nil, up)
			for i := range up {
				up[i] = bertAct(gate[i], m.actMode) * up[i]
			}
		} else {
			bertLinear(L.Wup, x, T, dim, m.FFN, L.Bup, up)
			for i := range up {
				up[i] = bertAct(up[i], m.actMode)
			}
		}
		bertLinear(L.Wdown, up, T, m.FFN, dim, L.Bdown, down)
		for i := range x {
			x[i] += down[i]
		}
		bertLayerNorm(x, T, dim, L.OutNormW, L.OutNormB, m.Eps)
	}
	return x, offsets, lengths
}

// pool reduces one sequence's hidden states (rows [off, off+n) of x) to a single
// vector via mean or CLS pooling, optionally L2-normalized.
func (m *InferenceBert) pool(x []float32, off, n int, pooling string, normalize bool) []float32 {
	dim := m.Dim
	out := make([]float32, dim)
	base := off * dim
	if pooling == "last" || pooling == "cls" {
		copy(out, x[base:base+dim]) // [CLS] is the sequence's first token
	} else {
		for i := 0; i < n; i++ {
			ro := base + i*dim
			for d := 0; d < dim; d++ {
				out[d] += x[ro+d]
			}
		}
		inv := 1.0 / float32(n)
		for d := range out {
			out[d] *= inv
		}
	}
	if normalize {
		var ss float64
		for _, val := range out {
			ss += float64(val) * float64(val)
		}
		if ss > 0 {
			inv := float32(1.0 / math.Sqrt(ss))
			for d := range out {
				out[d] *= inv
			}
		}
	}
	return out
}

// bertLinear computes dst = x @ W^T + bias for a [rows][inDim] input and a
// [outDim][inDim] weight (QuantWeight or dense float).
func bertLinear(w interface{}, x []float32, rows, inDim, outDim int, bias, dst []float32) []float32 {
	var res []float32
	switch ww := w.(type) {
	case *QuantWeight:
		res, _, _ = modelMatmulQuantIntoF32(ww, x, rows, inDim, dst)
	case []float32:
		res = dst[:rows*outDim]
		for r := 0; r < rows; r++ {
			xo := r * inDim
			for oF := 0; oF < outDim; oF++ {
				wo := oF * inDim
				var s float32
				for i := 0; i < inDim; i++ {
					s += x[xo+i] * ww[wo+i]
				}
				res[r*outDim+oF] = s
			}
		}
	default:
		return nil
	}
	if bias != nil {
		for r := 0; r < rows; r++ {
			ro := r * outDim
			for oF := 0; oF < outDim; oF++ {
				res[ro+oF] += bias[oF]
			}
		}
	}
	return res
}

// bertAct applies the FFN activation: mode 2 = SiLU (gated SwiGLU FFN, e.g.
// nomic-bert), anything else = GELU (plain BERT FFN). Selected per model.
func bertAct(x float32, mode int) float32 {
	if mode == 2 {
		return x / (1 + float32(math.Exp(float64(-x))))
	}
	return float32(gelu(float64(x)))
}

// bertLayerNorm applies a per-row LayerNorm (mean-centered, weight + bias) in place.
func bertLayerNorm(x []float32, rows, dim int, w, b []float32, eps float32) {
	for r := 0; r < rows; r++ {
		off := r * dim
		var mean float32
		for i := 0; i < dim; i++ {
			mean += x[off+i]
		}
		mean /= float32(dim)
		var variance float32
		for i := 0; i < dim; i++ {
			d := x[off+i] - mean
			variance += d * d
		}
		variance /= float32(dim)
		inv := float32(1.0 / math.Sqrt(float64(variance)+float64(eps)))
		for i := 0; i < dim; i++ {
			x[off+i] = (x[off+i]-mean)*inv*w[i] + b[i]
		}
	}
}

// bertApplyRope rotates each head's first 2*halfDim lanes in place. neox=true
// pairs lane j with j+halfDim (rotate-half); neox=false pairs 2j with 2j+1.
func bertApplyRope(data []float32, n, nHeads, headDim int, freqs []float32, halfDim int, neox bool) {
	dim := nHeads * headDim
	for i := 0; i < n; i++ {
		pos := float32(i)
		for h := 0; h < nHeads; h++ {
			off := i*dim + h*headDim
			for j := 0; j < halfDim; j++ {
				angle := freqs[j] * pos
				c := float32(math.Cos(float64(angle)))
				s := float32(math.Sin(float64(angle)))
				var a, b int
				if neox {
					a, b = off+j, off+j+halfDim
				} else {
					a, b = off+2*j, off+2*j+1
				}
				x0, x1 := data[a], data[b]
				data[a] = x0*c - x1*s
				data[b] = x0*s + x1*c
			}
		}
	}
}

// bertAttention is full (bidirectional) multi-head attention. q/k/v and out are
// [seqLen][dim].
func bertAttention(q, k, v []float32, seqLen, nHeads, headDim int, out []float32) {
	dim := nHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	scores := make([]float32, seqLen)
	for h := 0; h < nHeads; h++ {
		ho := h * headDim
		for i := 0; i < seqLen; i++ {
			qi := i*dim + ho
			maxs := float32(math.Inf(-1))
			for j := 0; j < seqLen; j++ {
				kj := j*dim + ho
				var s float32
				for d := 0; d < headDim; d++ {
					s += q[qi+d] * k[kj+d]
				}
				s *= scale
				scores[j] = s
				if s > maxs {
					maxs = s
				}
			}
			var sum float32
			for j := 0; j < seqLen; j++ {
				e := float32(math.Exp(float64(scores[j] - maxs)))
				scores[j] = e
				sum += e
			}
			inv := 1.0 / sum
			oi := i*dim + ho
			for d := 0; d < headDim; d++ {
				var acc float32
				for j := 0; j < seqLen; j++ {
					acc += scores[j] * v[j*dim+ho+d]
				}
				out[oi+d] = acc * inv
			}
		}
	}
}
