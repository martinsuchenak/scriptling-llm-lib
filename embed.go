package scriptlingllmlib

import (
	"fmt"
	"math"
)

// Pooling strategies for Embed: how per-token hidden states are reduced to one
// vector.
const (
	PoolingMean = "mean" // average the final hidden state over all tokens
	PoolingLast = "last" // take the final hidden state of the last token
)

// EmbedOptions configures an embedding request. Only Model and Text are required.
type EmbedOptions struct {
	Model string // path to the .gguf model (required)
	Text  string // text to embed (required)

	Pooling   string // PoolingMean (default) | PoolingLast
	Normalize bool   // L2-normalize the result (recommended for cosine similarity)
}

// Embed computes a dense vector embedding of opts.Text using the model's final
// hidden states. It loads the model through the same concurrency-safe global
// cache as Generate and runs on a private clone, so it is safe to call from
// multiple goroutines. The returned vector has length DModel.
func Embed(opts EmbedOptions) ([]float32, error) {
	shared, err := globalModelCacheF32.getOrLoad(opts.Model)
	if err != nil {
		return nil, err
	}
	m := shared.clone()
	ids := m.Tokenizer.Encode(opts.Text)
	if len(ids) == 0 {
		return nil, fmt.Errorf("embed: no tokens produced for input")
	}
	pooling := opts.Pooling
	if pooling == "" {
		pooling = PoolingMean
	}
	return m.embedTokens(ids, pooling, opts.Normalize), nil
}

// embedTokens runs the transformer over tokenIDs and pools the final-normed
// hidden states into a single vector.
func (m *InferenceModelF32) embedTokens(tokenIDs []int, pooling string, normalize bool) []float32 {
	enterInference()
	defer exitInference()

	m.initKVCaches()
	xData, seqLen, dModel := m.runBlocks(tokenIDs, 0)

	eps := float32(m.Config.NormEps)
	// normPos applies the model's final RMSNorm to position p into dst.
	normPos := func(p int, dst []float32) {
		off := p * dModel
		var ss float32
		for j := 0; j < dModel; j++ {
			v := xData[off+j]
			ss += v * v
		}
		inv := float32(1.0 / math.Sqrt(float64(ss)/float64(dModel)+float64(eps)))
		for j := 0; j < dModel; j++ {
			dst[j] = xData[off+j] * inv * m.FinalNormW[j]
		}
	}

	out := make([]float32, dModel)
	if pooling == PoolingLast {
		normPos(seqLen-1, out)
	} else {
		tmp := make([]float32, dModel)
		for p := 0; p < seqLen; p++ {
			normPos(p, tmp)
			for j := 0; j < dModel; j++ {
				out[j] += tmp[j]
			}
		}
		inv := float32(1.0 / float64(seqLen))
		for j := 0; j < dModel; j++ {
			out[j] *= inv
		}
	}

	if normalize {
		var ss float32
		for _, v := range out {
			ss += v * v
		}
		if ss > 0 {
			inv := float32(1.0 / math.Sqrt(float64(ss)))
			for j := range out {
				out[j] *= inv
			}
		}
	}
	return out
}
