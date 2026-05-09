package scriptlingllmlib

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

type TransformerBlock struct {
	AttnNormW []float64
	WQ        interface{}
	WK        interface{}
	WV        interface{}
	WO        interface{}
	QNormW    []float64
	KNormW    []float64
	FFNNormW  []float64
	WGate     interface{}
	WUp       interface{}
	WDown     interface{}
}

type KVCache struct {
	K [][]float64
	V [][]float64
}

type sessionEntry struct {
	kvCaches []KVCache
	kvPos    int
}

type modelCache struct {
	mu       sync.Mutex
	models   map[string]*InferenceModel
	sessions map[string]map[string]*sessionEntry
}

var globalModelCache = &modelCache{
	models:   make(map[string]*InferenceModel),
	sessions: make(map[string]map[string]*sessionEntry),
}

func (c *modelCache) getOrLoad(path string) (*InferenceModel, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if m, ok := c.models[path]; ok {
		return m, nil
	}

	gguf, err := LoadGGUF(path)
	if err != nil {
		return nil, err
	}

	model, err := buildInferenceModel(gguf, path)
	if err != nil {
		return nil, err
	}

	c.models[path] = model
	return model, nil
}

type InferenceModel struct {
	Config      ModelConfig
	Arch        string
	TokenEmb    [][]float64
	Blocks      []TransformerBlock
	FinalNormW  []float64
	OutputW     interface{}
	OutputWQ8   *QuantWeight
	KVCaches    []KVCache
	Tokenizer   *Tokenizer
	ChatTpl     string
	nHeads      int
	nKVHeads    int
	dK          int
	nRep        int
	ropeFreqs   []float64
	ropeHalfDim int
}

func copyKVCaches(caches []KVCache) []KVCache {
	result := make([]KVCache, len(caches))
	for i, c := range caches {
		result[i] = KVCache{
			K: make([][]float64, len(c.K)),
			V: make([][]float64, len(c.V)),
		}
		for h := range c.K {
			result[i].K[h] = append([]float64(nil), c.K[h]...)
			result[i].V[h] = append([]float64(nil), c.V[h]...)
		}
	}
	return result
}

func loadOptional1D(gguf *GGUFModel, name string) []float64 {
	iface, err := gguf.LoadTensor(name)
	if err != nil {
		return nil
	}
	f, ok := iface.([]float64)
	if !ok {
		return nil
	}
	return f
}

func (c *modelCache) clearModels() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.models = make(map[string]*InferenceModel)
	c.sessions = make(map[string]map[string]*sessionEntry)
}

func (c *modelCache) getSession(modelPath, sessionID string) *sessionEntry {
	sessions, ok := c.sessions[modelPath]
	if !ok {
		return nil
	}
	return sessions[sessionID]
}

func (c *modelCache) saveSession(modelPath, sessionID string, caches []KVCache, kvPos int) {
	if _, ok := c.sessions[modelPath]; !ok {
		c.sessions[modelPath] = make(map[string]*sessionEntry)
	}
	c.sessions[modelPath][sessionID] = &sessionEntry{
		kvCaches: copyKVCaches(caches),
		kvPos:    kvPos,
	}
}

func (c *modelCache) clearSession(modelPath, sessionID string) {
	if sessions, ok := c.sessions[modelPath]; ok {
		delete(sessions, sessionID)
		if len(sessions) == 0 {
			delete(c.sessions, modelPath)
		}
	}
}

func buildInferenceModel(gguf *GGUFModel, path string) (*InferenceModel, error) {
	cfg := gguf.Config

	gguf.Metadata["_path"] = path

	tokenEmbIface, err := gguf.LoadTensor("token_embedding.weight")
	if err != nil {
		return nil, fmt.Errorf("model: loading token_embedding: %w", err)
	}
	tokenEmb, ok := tokenEmbIface.([][]float64)
	if !ok {
		return nil, fmt.Errorf("model: token_embedding must be 2D float matrix")
	}

	finalNormIface, err := gguf.LoadTensor("final_norm.weight")
	if err != nil {
		return nil, fmt.Errorf("model: loading final_norm: %w", err)
	}
	finalNormW, ok := finalNormIface.([]float64)
	if !ok {
		return nil, fmt.Errorf("model: final_norm must be 1D")
	}

	outputWIface, err := gguf.LoadTensor("output.weight")
	if err != nil {
		outputWIface = tokenEmb
	}

	var outputWQ8 *QuantWeight
	var outputW interface{}

	switch w := outputWIface.(type) {
	case *QuantWeight:
		outputW = w
		outputWQ8 = w
	case [][]float64:
		qw := quantizeQ8Rows(w)
		if qw != nil {
			outputW = qw
			outputWQ8 = qw
		} else {
			outputW = w
		}
	default:
		outputW = outputWIface
	}

	blocks := make([]TransformerBlock, cfg.NLayers)
	for i := 0; i < cfg.NLayers; i++ {
		prefix := fmt.Sprintf("blocks.%d.", i)

		attnNormIface, err := gguf.LoadTensor(prefix + "attn_norm.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn_norm.weight", err)
		}
		attnNormW, ok := attnNormIface.([]float64)
		if !ok {
			return nil, fmt.Errorf("model: %s must be 1D", prefix+"attn_norm.weight")
		}

		wq, err := gguf.LoadTensor(prefix + "attn.w_q.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_q.weight", err)
		}
		wk, err := gguf.LoadTensor(prefix + "attn.w_k.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_k.weight", err)
		}
		wv, err := gguf.LoadTensor(prefix + "attn.w_v.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_v.weight", err)
		}
		wo, err := gguf.LoadTensor(prefix + "attn.w_o.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_o.weight", err)
		}

		ffnNormIface, err := gguf.LoadTensor(prefix + "ffn_norm.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn_norm.weight", err)
		}
		ffnNormW, ok := ffnNormIface.([]float64)
		if !ok {
			return nil, fmt.Errorf("model: %s must be 1D", prefix+"ffn_norm.weight")
		}

		wGate, err := gguf.LoadTensor(prefix + "ffn.w_gate.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn.w_gate.weight", err)
		}
		wUp, err := gguf.LoadTensor(prefix + "ffn.w_up.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn.w_up.weight", err)
		}
		wDown, err := gguf.LoadTensor(prefix + "ffn.w_down.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn.w_down.weight", err)
		}

		blocks[i] = TransformerBlock{
			AttnNormW: attnNormW,
			WQ:        wq,
			WK:        wk,
			WV:        wv,
			WO:        wo,
			QNormW:    loadOptional1D(gguf, prefix+"attn_q_norm.weight"),
			KNormW:    loadOptional1D(gguf, prefix+"attn_k_norm.weight"),
			FFNNormW:  ffnNormW,
			WGate:     wGate,
			WUp:       wUp,
			WDown:     wDown,
		}
	}

	nHeads := cfg.NHeads
	nKVHeads := cfg.NKVHeads
	if nKVHeads == 0 {
		nKVHeads = nHeads
	}
	dK := cfg.DModel / nHeads
	nRep := nHeads / nKVHeads

	ropeDim := cfg.RopeDim
	if ropeDim == 0 {
		ropeDim = dK
	}
	ropeHalfDim := ropeDim / 2
	ropeFreqs := make([]float64, ropeHalfDim)
	for i := 0; i < ropeHalfDim; i++ {
		ropeFreqs[i] = 1.0 / math.Pow(cfg.RopeFreqBase, 2.0*float64(i)/float64(ropeDim))
	}

	td := gguf.Tokenizer
	var tokenizer *Tokenizer
	if td.Type == "bpe" && len(td.Merges) > 0 {
		tokenizer = NewTokenizer(td.Vocab, td.Merges, td.Special)
	} else {
		tokenizer = NewTokenizer(td.Vocab, nil, td.Special)
	}

	arch := ""
	if v, ok := gguf.Metadata["general.architecture"]; ok {
		arch, _ = v.(string)
	}

	return &InferenceModel{
		Config:      cfg,
		Arch:        arch,
		TokenEmb:    tokenEmb,
		Blocks:      blocks,
		FinalNormW:  finalNormW,
		OutputW:     outputW,
		OutputWQ8:   outputWQ8,
		Tokenizer:   tokenizer,
		ChatTpl:     td.ChatTemplate,
		nHeads:      nHeads,
		nKVHeads:    nKVHeads,
		dK:          dK,
		nRep:        nRep,
		ropeFreqs:   ropeFreqs,
		ropeHalfDim: ropeHalfDim,
	}, nil
}

func (m *InferenceModel) initKVCaches() {
	m.KVCaches = make([]KVCache, m.Config.NLayers)
	for i := range m.KVCaches {
		m.KVCaches[i] = KVCache{
			K: make([][]float64, m.nKVHeads),
			V: make([][]float64, m.nKVHeads),
		}
	}
}

func (m *InferenceModel) Forward(tokenIDs []int, startPos int) []float64 {
	seqLen := len(tokenIDs)
	dModel := m.Config.DModel

	xData := make([]float64, seqLen*dModel)
	for i, tid := range tokenIDs {
		if tid < len(m.TokenEmb) {
			copy(xData[i*dModel:], m.TokenEmb[tid])
		}
	}

	for i := 0; i < m.Config.NLayers; i++ {
		xData = m.forwardBlock(i, xData, seqLen, dModel, startPos)
	}

	logits := m.outputLogits(xData, seqLen, dModel)

	return logits
}

func (m *InferenceModel) forwardBlock(blockIdx int, xData []float64, seqLen, dModel, startPos int) []float64 {
	block := m.Blocks[blockIdx]
	cache := m.KVCaches[blockIdx]
	eps := m.Config.NormEps

	normed := make([]float64, len(xData))
	rmsNormFlat(xData, block.AttnNormW, eps, seqLen, dModel, normed)

	var qData, kData, vData []float64
	var qRows, qCols, kRows, kCols, vRows, vCols int

	qData, qRows, qCols = modelMatmul(block.WQ, normed, seqLen, dModel)
	kData, kRows, kCols = modelMatmul(block.WK, normed, seqLen, dModel)
	vData, vRows, vCols = modelMatmul(block.WV, normed, seqLen, dModel)

	qHeads := splitHeadsData(qData, qRows, qCols, m.nHeads)
	kHeads := splitHeadsData(kData, kRows, kCols, m.nKVHeads)
	vHeads := splitHeadsData(vData, vRows, vCols, m.nKVHeads)

	if block.QNormW != nil {
		for h := 0; h < m.nHeads; h++ {
			rmsNormFlat(qHeads[h], block.QNormW, eps, qRows, m.dK, qHeads[h])
		}
	}
	if block.KNormW != nil {
		for h := 0; h < m.nKVHeads; h++ {
			rmsNormFlat(kHeads[h], block.KNormW, eps, kRows, m.dK, kHeads[h])
		}
	}

	for h := 0; h < m.nHeads; h++ {
		applyRopeInPlace(qHeads[h], qRows, m.dK, startPos, m.ropeFreqs, m.ropeHalfDim)
	}
	for h := 0; h < m.nKVHeads; h++ {
		applyRopeInPlace(kHeads[h], kRows, m.dK, startPos, m.ropeFreqs, m.ropeHalfDim)
	}

	for h := 0; h < m.nKVHeads; h++ {
		cache.K[h] = append(cache.K[h], kHeads[h]...)
		cache.V[h] = append(cache.V[h], vHeads[h]...)
	}

	for h := 0; h < m.nKVHeads; h++ {
		kHeads[h] = cache.K[h]
		vHeads[h] = cache.V[h]
	}

	if m.nRep > 1 {
		kHeads = repeatKVData(kHeads, m.nRep)
		vHeads = repeatKVData(vHeads, m.nRep)
	}

	kvLen := len(kHeads[0]) / m.dK
	cacheLen := kvLen - seqLen
	attnOut := make([]float64, seqLen*m.nHeads*m.dK)
	for h := 0; h < m.nHeads; h++ {
		fusedAttentionHead(qHeads[h], kHeads[h], vHeads[h], seqLen, m.dK, kvLen, true, cacheLen, attnOut, h*seqLen*m.dK)
	}

	merged := mergeHeadsData(attnOut, seqLen, m.nHeads, m.dK)

	oData, _, _ := modelMatmul(block.WO, merged, seqLen, m.nHeads*m.dK)

	residual := make([]float64, len(xData))
	for i := range xData {
		residual[i] = xData[i] + oData[i]
	}

	ffnNormed := make([]float64, len(residual))
	rmsNormFlat(residual, block.FFNNormW, eps, seqLen, dModel, ffnNormed)

	var gateData, upData []float64
	var gateRows, gateCols int
	gateData, gateRows, gateCols = modelMatmul(block.WGate, ffnNormed, seqLen, dModel)
	upData, _, _ = modelMatmul(block.WUp, ffnNormed, seqLen, dModel)

	hidden := gateRows * gateCols
	siluData := make([]float64, hidden)
	for i := 0; i < hidden; i++ {
		g := gateData[i]
		s := 1.0 / (1.0 + math.Exp(-g))
		siluData[i] = g * s * upData[i]
	}

	downData, _, _ := modelMatmul(block.WDown, siluData, gateRows, gateCols)

	result := make([]float64, len(residual))
	for i := range residual {
		result[i] = residual[i] + downData[i]
	}

	return result
}

func (m *InferenceModel) outputLogits(xData []float64, seqLen, dModel int) []float64 {
	lastOff := (seqLen - 1) * dModel
	eps := m.Config.NormEps

	var ss float64
	for j := 0; j < dModel; j++ {
		ss += xData[lastOff+j] * xData[lastOff+j]
	}
	inv := 1.0 / math.Sqrt(ss/float64(dModel)+eps)

	normed := make([]float64, dModel)
	for j := 0; j < dModel; j++ {
		normed[j] = xData[lastOff+j] * inv * m.FinalNormW[j]
	}

	switch w := m.OutputW.(type) {
	case *QuantWeight:
		return modelMatmulRowQuant(w, normed)
	case [][]float64:
		return modelMatmulRowFloat(w, normed)
	}

	return nil
}

func modelMatmul(w interface{}, xData []float64, xRows, xCols int) ([]float64, int, int) {
	switch wt := w.(type) {
	case *QuantWeight:
		return modelMatmulQuant(wt, xData, xRows, xCols)
	case [][]float64:
		return modelMatmulFloat2D(wt, xData, xRows, xCols)
	}
	return nil, 0, 0
}

func modelMatmulQuant(w *QuantWeight, xData []float64, xRows, xCols int) ([]float64, int, int) {
	rawBytes := w.Raw
	outFeatures := w.Rows

	switch w.QType {
	case "q8":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 34
		result := make([]float64, xRows*outFeatures)
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				xOff := xi * xCols
				rOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q8DotGroupX(rawBytes, rOff+g*34, xData, xOff+g*32)
				}
				result[xi*outFeatures+j] = sum
			}
		})
		return result, xRows, outFeatures
	case "q4":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 18
		result := make([]float64, xRows*outFeatures)
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				xOff := xi * xCols
				rOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q4DotGroupX(rawBytes, rOff+g*18, xData, xOff+g*32)
				}
				result[xi*outFeatures+j] = sum
			}
		})
		return result, xRows, outFeatures
	case "q4_1":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 20
		result := make([]float64, xRows*outFeatures)
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				xOff := xi * xCols
				rOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q4_1DotGroupX(rawBytes, rOff+g*20, xData, xOff+g*32)
				}
				result[xi*outFeatures+j] = sum
			}
		})
		return result, xRows, outFeatures
	case "q5":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 22
		result := make([]float64, xRows*outFeatures)
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				xOff := xi * xCols
				rOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q5DotGroupX(rawBytes, rOff+g*22, xData, xOff+g*32)
				}
				result[xi*outFeatures+j] = sum
			}
		})
		return result, xRows, outFeatures
	case "q4k":
		blocksPerRow := w.Groups
		rowBytes := blocksPerRow * 144
		result := make([]float64, xRows*outFeatures)
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				xOff := xi * xCols
				rOff := j * rowBytes
				var sum float64
				for b := 0; b < blocksPerRow; b++ {
					sum += q4kDotBlockFast(rawBytes, rOff+b*144, xData, xOff+b*256)
				}
				result[xi*outFeatures+j] = sum
			}
		})
		return result, xRows, outFeatures
	case "q6k":
		blocksPerRow := w.Groups
		rowBytes := blocksPerRow * 210
		result := make([]float64, xRows*outFeatures)
		parallelFor(xRows*outFeatures, func(start, end int) {
			for idx := start; idx < end; idx++ {
				xi := idx / outFeatures
				j := idx % outFeatures
				xOff := xi * xCols
				rOff := j * rowBytes
				var sum float64
				for b := 0; b < blocksPerRow; b++ {
					sum += q6kDotBlock(rawBytes, rOff+b*210, xData, xOff+b*256)
				}
				result[xi*outFeatures+j] = sum
			}
		})
		return result, xRows, outFeatures
	}

	return nil, 0, 0
}

func modelMatmulFloat2D(w [][]float64, xData []float64, xRows, xCols int) ([]float64, int, int) {
	wRows := len(w)
	if wRows == 0 {
		return nil, 0, 0
	}
	wCols := len(w[0])
	wFlat := make([]float64, 0, wRows*wCols)
	for _, r := range w {
		wFlat = append(wFlat, r...)
	}
	return fusedMatmulFloat(xData, xRows, xCols, wFlat, wRows, wCols)
}

func modelMatmulRowQuant(w *QuantWeight, normed []float64) []float64 {
	rawBytes := w.Raw
	outFeatures := w.Rows

	switch w.QType {
	case "q8":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 34
		result := make([]float64, outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q8DotGroupX(rawBytes, rOff+g*34, normed, g*32)
				}
				result[j] = sum
			}
		})
		return result
	case "q4":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 18
		result := make([]float64, outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q4DotGroupX(rawBytes, rOff+g*18, normed, g*32)
				}
				result[j] = sum
			}
		})
		return result
	case "q4_1":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 20
		result := make([]float64, outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q4_1DotGroupX(rawBytes, rOff+g*20, normed, g*32)
				}
				result[j] = sum
			}
		})
		return result
	case "q5":
		groupsPerRow := w.Groups
		rowBytes := groupsPerRow * 22
		result := make([]float64, outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rOff := j * rowBytes
				var sum float64
				for g := 0; g < groupsPerRow; g++ {
					sum += q5DotGroupX(rawBytes, rOff+g*22, normed, g*32)
				}
				result[j] = sum
			}
		})
		return result
	case "q4k":
		blocksPerRow := w.Groups
		rowBytes := blocksPerRow * 144
		result := make([]float64, outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rOff := j * rowBytes
				var sum float64
				for b := 0; b < blocksPerRow; b++ {
					sum += q4kDotBlockFast(rawBytes, rOff+b*144, normed, b*256)
				}
				result[j] = sum
			}
		})
		return result
	case "q6k":
		blocksPerRow := w.Groups
		rowBytes := blocksPerRow * 210
		result := make([]float64, outFeatures)
		parallelFor(outFeatures, func(start, end int) {
			for j := start; j < end; j++ {
				rOff := j * rowBytes
				var sum float64
				for b := 0; b < blocksPerRow; b++ {
					sum += q6kDotBlock(rawBytes, rOff+b*210, normed, b*256)
				}
				result[j] = sum
			}
		})
		return result
	}

	return nil
}

func modelMatmulRowFloat(w [][]float64, normed []float64) []float64 {
	wRows := len(w)
	wCols := len(w[0])
	wFlat := make([]float64, 0, wRows*wCols)
	for _, r := range w {
		wFlat = append(wFlat, r...)
	}
	logits := make([]float64, wRows)
	parallelFor(wRows, func(start, end int) {
		for j := start; j < end; j++ {
			wOff := j * wCols
			var sum float64
			for l := 0; l < wCols; l++ {
				sum += normed[l] * wFlat[wOff+l]
			}
			logits[j] = sum
		}
	})
	return logits
}

func (m *InferenceModel) Generate(prompt string, maxTokens int, strategy string, temperature float64, topK int, topP float64, repeatPenalty float64, repeatLastN int, systemPrompt string, templateName string, kvStartPos int) (string, int, int, int) {
	if kvStartPos == 0 {
		if templateName != "" {
			if tpl, ok := defaultTemplates[templateName]; ok {
				prompt = applyChatTemplate(tpl, prompt, systemPrompt, m.Arch)
			}
		} else if m.ChatTpl != "" {
			prompt = applyChatTemplate(m.ChatTpl, prompt, systemPrompt, m.Arch)
		}
	} else {
		prompt = "<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
	}

	tokenIDs := m.Tokenizer.Encode(prompt)
	nPrompt := len(tokenIDs)

	if kvStartPos == 0 {
		m.initKVCaches()
	}

	context := tokenIDs
	maxLen := m.Config.MaxSeqLen
	if len(context)+kvStartPos > maxLen {
		context = context[len(context)+kvStartPos-maxLen:]
	}

	logits := m.Forward(context, kvStartPos)

	recent := make([]int64, len(tokenIDs))
	for i, id := range tokenIDs {
		recent[i] = int64(id)
	}

	if repeatPenalty > 0 && len(recent) > 0 {
		applyRepeatPenalty(logits, recent, repeatPenalty, repeatLastN)
	}
	nextID := sampleLogits(logits, strategy, temperature, topK, topP)

	tokenIDs = append(tokenIDs, nextID)
	recent = append(recent, int64(nextID))

	if nextID == m.Tokenizer.EOSID {
		finalPos := kvStartPos + nPrompt
		return m.Tokenizer.Decode(tokenIDs[nPrompt:]), 1, nPrompt, finalPos
	}

	nGen := 1
	for step := 1; step < maxTokens; step++ {
		pos := kvStartPos + nPrompt + step - 1

		logits = m.Forward([]int{nextID}, pos)

		if repeatPenalty > 0 && len(recent) > 0 {
			applyRepeatPenalty(logits, recent, repeatPenalty, repeatLastN)
		}
		nextID = sampleLogits(logits, strategy, temperature, topK, topP)

		tokenIDs = append(tokenIDs, nextID)
		recent = append(recent, int64(nextID))
		nGen++

		if nextID == m.Tokenizer.EOSID {
			break
		}
	}

	finalPos := kvStartPos + nPrompt + nGen - 1
	return m.Tokenizer.Decode(tokenIDs[nPrompt:]), nGen, nPrompt, finalPos
}

func sampleLogits(logits []float64, strategy string, temperature float64, topK int, topP float64) int {
	n := len(logits)
	if n == 0 {
		return 0
	}

	switch strategy {
	case "greedy":
		bestIdx := 0
		bestVal := logits[0]
		for i := 1; i < n; i++ {
			if logits[i] > bestVal {
				bestVal = logits[i]
				bestIdx = i
			}
		}
		return bestIdx

	case "temperature":
		probs := softmaxGo(logits, temperature)
		return weightedSampleGo(probs)

	case "top_k":
		if topK <= 0 {
			topK = 50
		}
		if topK > n {
			topK = n
		}
		indexed := make([]indexedFloat, n)
		for i, v := range logits {
			indexed[i] = indexedFloat{index: i, value: v}
		}
		partialNthElement(indexed, topK)

		topKLogits := make([]float64, topK)
		topIndices := make([]int, topK)
		for i := 0; i < topK; i++ {
			topKLogits[i] = indexed[i].value / temperature
			topIndices[i] = indexed[i].index
		}

		probs := softmaxInPlace(topKLogits)
		offset := sampleRng.Float64()
		cum := 0.0
		for i, p := range probs {
			cum += p
			if offset <= cum {
				return topIndices[i]
			}
		}
		return topIndices[topK-1]

	case "top_p":
		scaled := make([]float64, n)
		for i, v := range logits {
			scaled[i] = v / temperature
		}
		probs := softmaxGo(scaled, 1.0)

		sorted := make([]idxProbEntry, n)
		for i, pr := range probs {
			sorted[i] = idxProbEntry{i, pr}
		}
		sortByProbDesc(sorted)

		cum := 0.0
		cutoff := 0
		for i, sp := range sorted {
			cum += sp.prob
			cutoff = i
			if cum >= topP {
				break
			}
		}
		cutoff++

		filteredProbs := make([]float64, cutoff)
		filteredIndices := make([]int, cutoff)
		for i := 0; i < cutoff; i++ {
			filteredProbs[i] = sorted[i].prob
			filteredIndices[i] = sorted[i].idx
		}

		var sum float64
		for _, pr := range filteredProbs {
			sum += pr
		}
		invSum := 1.0 / sum

		offset := sampleRng.Float64()
		cum = 0.0
		for i, pr := range filteredProbs {
			cum += pr * invSum
			if offset <= cum {
				return filteredIndices[i]
			}
		}
		return filteredIndices[cutoff-1]
	}

	return 0
}

func softmaxGo(logits []float64, temperature float64) []float64 {
	n := len(logits)
	result := make([]float64, n)
	maxVal := logits[0] / temperature
	for i := 1; i < n; i++ {
		v := logits[i] / temperature
		if v > maxVal {
			maxVal = v
		}
	}
	var sumExp float64
	for i, v := range logits {
		e := math.Exp(v/temperature - maxVal)
		result[i] = e
		sumExp += e
	}
	invSum := 1.0 / sumExp
	for i := range result {
		result[i] *= invSum
	}
	return result
}

func weightedSampleGo(probs []float64) int {
	offset := sampleRng.Float64()
	cum := 0.0
	for i, p := range probs {
		cum += p
		if offset <= cum {
			return i
		}
	}
	return len(probs) - 1
}

type idxProbEntry struct {
	idx  int
	prob float64
}

func sortByProbDesc(data []idxProbEntry) {
	for i := 1; i < len(data); i++ {
		for j := i; j > 0 && data[j].prob > data[j-1].prob; j-- {
			data[j], data[j-1] = data[j-1], data[j]
		}
	}
}

func fnGenerate(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 2, 4); err != nil {
		return err
	}

	modelPath, ok := args[0].(*object.String)
	if !ok {
		return errors.NewTypeError("STRING", args[0].Type().String())
	}

	prompt, ok := args[1].(*object.String)
	if !ok {
		return errors.NewTypeError("STRING", args[1].Type().String())
	}

	maxTokens := 100
	if len(args) >= 3 {
		v, err := args[2].AsInt()
		if err != nil {
			return errors.NewTypeError("INTEGER", args[2].Type().String())
		}
		maxTokens = int(v)
	}

	strategy := "greedy"
	if len(args) >= 4 {
		s, ok := args[3].(*object.String)
		if !ok {
			return errors.NewTypeError("STRING", args[3].Type().String())
		}
		strategy = s.Value
	}

	temperature := 1.0
	if kwargs.Has("temperature") {
		temperature = kwargs.MustGetFloat("temperature", 1.0)
	}

	topK := 50
	if kwargs.Has("top_k") {
		topK = int(kwargs.MustGetInt("top_k", 50))
	}

	topP := 0.9
	if kwargs.Has("top_p") {
		topP = kwargs.MustGetFloat("top_p", 0.9)
	}

	repeatPenalty := 1.15
	if kwargs.Has("repeat_penalty") {
		repeatPenalty = kwargs.MustGetFloat("repeat_penalty", 1.15)
	}

	repeatLastN := 64
	if kwargs.Has("repeat_last_n") {
		repeatLastN = int(kwargs.MustGetInt("repeat_last_n", 64))
	}

	systemPrompt := ""
	if kwargs.Has("system_prompt") {
		sp, ok := kwargs.Get("system_prompt").(*object.String)
		if ok {
			systemPrompt = sp.Value
		}
	}

	templateName := ""
	if kwargs.Has("template") {
		t, ok := kwargs.Get("template").(*object.String)
		if ok {
			templateName = t.Value
		}
	}

	showStats := false
	if kwargs.Has("stats") {
		s, ok := kwargs.Get("stats").(*object.Boolean)
		if ok {
			showStats = s.Value
		} else {
			v, err := kwargs.Get("stats").AsInt()
			if err == nil {
				showStats = v != 0
			}
		}
	}

	var sessionID string
	if kwargs.Has("session") {
		s, ok := kwargs.Get("session").(*object.String)
		if ok {
			sessionID = s.Value
		}
	}

	model, err := globalModelCacheF32.getOrLoad(modelPath.Value)
	if err != nil {
		return errors.NewError("generate: %s", err.Error())
	}

	kvStartPos := 0

	if sessionID != "" {
		globalModelCacheF32.mu.Lock()
		entry := globalModelCacheF32.getSession(modelPath.Value, sessionID)
		if entry != nil {
			model.KVCaches = entry.kvCaches
			kvStartPos = entry.kvPos
		} else {
			model.initKVCaches()
		}
		globalModelCacheF32.mu.Unlock()
	}

	tGenStart := time.Now()
	result, nGen, nPrompt, finalPos := model.Generate(
		prompt.Value, maxTokens, strategy, temperature,
		topK, topP, repeatPenalty, repeatLastN,
		systemPrompt, templateName, kvStartPos,
	)
	tGenEnd := time.Now()

	if sessionID != "" {
		globalModelCacheF32.mu.Lock()
		globalModelCacheF32.saveSession(modelPath.Value, sessionID, model.KVCaches, finalPos)
		globalModelCacheF32.mu.Unlock()
	}

	if showStats {
		totalTime := tGenEnd.Sub(tGenStart).Seconds()
		prefillSec := model.PrefillMs / 1000.0
		decodeSec := model.DecodeMs / 1000.0
		tps := float64(0)
		if decodeSec > 0 {
			tps = float64(nGen) / decodeSec
		}
		promptTps := float64(0)
		if prefillSec > 0 {
			promptTps = float64(nPrompt) / prefillSec
		}
		stats := fmt.Sprintf("\n--- stats ---\n  prompt tokens: %d\n  gen tokens:    %d\n  prefill:       %.2fs (%.1f t/s)\n  decode:        %.2fs (%.1f t/s)\n  total:         %.2fs", nPrompt, nGen, prefillSec, promptTps, decodeSec, tps, totalTime)
		return &object.String{Value: result + stats}
	}

	return &object.String{Value: result}
}

func fnClearSession(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.ExactArgs(args, 2); err != nil {
		return err
	}

	modelPath, ok := args[0].(*object.String)
	if !ok {
		return errors.NewTypeError("STRING", args[0].Type().String())
	}

	sessionID, ok := args[1].(*object.String)
	if !ok {
		return errors.NewTypeError("STRING", args[1].Type().String())
	}

	globalModelCacheF32.mu.Lock()
	globalModelCacheF32.clearSession(modelPath.Value, sessionID.Value)
	globalModelCacheF32.mu.Unlock()

	return object.NewBoolean(true)
}
