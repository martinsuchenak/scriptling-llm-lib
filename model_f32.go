package scriptlingllmlib

import (
	"fmt"
	"math"
	"sync"
	"time"
)

type modelCacheF32 struct {
	mu       sync.Mutex
	models   map[string]*InferenceModelF32
	sessions map[string]map[string]*sessionEntryF32
}

type sessionEntryF32 struct {
	kvCaches []KVCacheF32
	kvPos    int
}

var globalModelCacheF32 = &modelCacheF32{
	models:   make(map[string]*InferenceModelF32),
	sessions: make(map[string]map[string]*sessionEntryF32),
}

func (c *modelCacheF32) getOrLoad(path string) (*InferenceModelF32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if m, ok := c.models[path]; ok {
		return m, nil
	}

	gguf, err := LoadGGUF(path)
	if err != nil {
		return nil, err
	}

	model, err := buildInferenceModelF32(gguf, path)
	if err != nil {
		return nil, err
	}

	c.models[path] = model
	return model, nil
}

func (c *modelCacheF32) clearModels() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.models = make(map[string]*InferenceModelF32)
	c.sessions = make(map[string]map[string]*sessionEntryF32)
}

func (c *modelCacheF32) getSession(modelPath, sessionID string) *sessionEntryF32 {
	sessions, ok := c.sessions[modelPath]
	if !ok {
		return nil
	}
	return sessions[sessionID]
}

func (c *modelCacheF32) saveSession(modelPath, sessionID string, caches []KVCacheF32, kvPos int) {
	if _, ok := c.sessions[modelPath]; !ok {
		c.sessions[modelPath] = make(map[string]*sessionEntryF32)
	}
	if entry, ok := c.sessions[modelPath][sessionID]; ok {
		entry.kvPos = kvPos
	} else {
		c.sessions[modelPath][sessionID] = &sessionEntryF32{
			kvCaches: copyKVCachesF32(caches),
			kvPos:    kvPos,
		}
	}
}

func (c *modelCacheF32) clearSession(modelPath, sessionID string) {
	if sessions, ok := c.sessions[modelPath]; ok {
		delete(sessions, sessionID)
		if len(sessions) == 0 {
			delete(c.sessions, modelPath)
		}
	}
}

func copyKVCachesF32(caches []KVCacheF32) []KVCacheF32 {
	result := make([]KVCacheF32, len(caches))
	for i, c := range caches {
		result[i] = KVCacheF32{
			K: make([][]float32, len(c.K)),
			V: make([][]float32, len(c.V)),
		}
		for h := range c.K {
			result[i].K[h] = append([]float32(nil), c.K[h]...)
			result[i].V[h] = append([]float32(nil), c.V[h]...)
		}
	}
	return result
}

func buildInferenceModelF32(gguf *GGUFModel, path string) (*InferenceModelF32, error) {
	cfg := gguf.Config
	gguf.Metadata["_path"] = path

	tokenEmbF32, embRows, embCols, err := loadTensorF32(gguf, "token_embedding.weight")
	if err != nil {
		return nil, fmt.Errorf("model: loading token_embedding: %w", err)
	}

	finalNormF32, err := loadTensor1DF32(gguf, "final_norm.weight")
	if err != nil {
		return nil, fmt.Errorf("model: loading final_norm: %w", err)
	}

	var outputWIface interface{}
	outputWIface, err = gguf.LoadTensor("output.weight")
	if err != nil {
		outputWIface = nil
	}

	var outputW interface{}

	if outputWIface != nil {
		switch w := outputWIface.(type) {
		case *QuantWeight:
			outputW = w
		case [][]float64:
			qw := quantizeQ8Rows(w)
			if qw != nil {
				outputW = qw
			} else {
				outputW = flattenF64ToF32(w)
			}
		default:
			outputWIface = nil
		}
	}

	if outputWIface == nil {
		emb2D := make([][]float64, embRows)
		for i := 0; i < embRows; i++ {
			row := make([]float64, embCols)
			for j := 0; j < embCols; j++ {
				row[j] = float64(tokenEmbF32[i*embCols+j])
			}
			emb2D[i] = row
		}
		qw := quantizeQ8Rows(emb2D)
		if qw != nil {
			outputW = qw
		} else {
			outputW = tokenEmbF32
		}
	}

	blocks := make([]TransformerBlockF32, cfg.NLayers)
	for i := 0; i < cfg.NLayers; i++ {
		prefix := fmt.Sprintf("blocks.%d.", i)

		attnNormF32, err := loadTensor1DF32(gguf, prefix+"attn_norm.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn_norm.weight", err)
		}

		wq, err := loadWeightF32(gguf, prefix+"attn.w_q.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_q.weight", err)
		}
		wk, err := loadWeightF32(gguf, prefix+"attn.w_k.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_k.weight", err)
		}
		wv, err := loadWeightF32(gguf, prefix+"attn.w_v.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_v.weight", err)
		}
		wo, err := loadWeightF32(gguf, prefix+"attn.w_o.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_o.weight", err)
		}

		ffnNormF32, err := loadTensor1DF32(gguf, prefix+"ffn_norm.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn_norm.weight", err)
		}

		wGate, err := loadWeightF32(gguf, prefix+"ffn.w_gate.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn.w_gate.weight", err)
		}
		wUp, err := loadWeightF32(gguf, prefix+"ffn.w_up.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn.w_up.weight", err)
		}
		wDown, err := loadWeightF32(gguf, prefix+"ffn.w_down.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn.w_down.weight", err)
		}

		blocks[i] = TransformerBlockF32{
			AttnNormW: attnNormF32,
			WQ:        wq,
			WK:        wk,
			WV:        wv,
			WO:        wo,
			QNormW:    loadOptional1DF32(gguf, prefix+"attn_q_norm.weight"),
			KNormW:    loadOptional1DF32(gguf, prefix+"attn_k_norm.weight"),
			FFNNormW:  ffnNormF32,
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
	ropeFreqs := make([]float32, ropeHalfDim)
	for i := 0; i < ropeHalfDim; i++ {
		ropeFreqs[i] = float32(1.0 / math.Pow(cfg.RopeFreqBase, 2.0*float64(i)/float64(ropeDim)))
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

	return &InferenceModelF32{
		Config:      cfg,
		Arch:        arch,
		TokenEmb:    tokenEmbF32,
		EmbRows:     embRows,
		EmbCols:     embCols,
		Blocks:      blocks,
		FinalNormW:  finalNormF32,
		OutputW:     outputW,
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

func coalesceOutputWeight(f32Weight []float32, qw *QuantWeight) interface{} {
	if qw != nil {
		return qw
	}
	if f32Weight != nil {
		return f32Weight
	}
	return nil
}

func flattenF64ToF32(m [][]float64) []float32 {
	if len(m) == 0 {
		return nil
	}
	rows := len(m)
	cols := len(m[0])
	flat := make([]float32, rows*cols)
	for i, row := range m {
		for j, v := range row {
			flat[i*cols+j] = float32(v)
		}
	}
	return flat
}

func loadWeightF32(gguf *GGUFModel, name string) (interface{}, error) {
	iface, err := gguf.LoadTensor(name)
	if err != nil {
		return nil, err
	}
	switch w := iface.(type) {
	case *QuantWeight:
		return w, nil
	case [][]float64:
		return flattenF64ToF32(w), nil
	}
	return nil, nil
}

func loadTensor1DF32(gguf *GGUFModel, name string) ([]float32, error) {
	iface, err := gguf.LoadTensor(name)
	if err != nil {
		return nil, err
	}
	switch w := iface.(type) {
	case []float64:
		return f64ToF32(w), nil
	}
	return nil, fmt.Errorf("model: %s must be 1D", name)
}

func loadOptional1DF32(gguf *GGUFModel, name string) []float32 {
	iface, err := gguf.LoadTensor(name)
	if err != nil {
		return nil
	}
	if f, ok := iface.([]float64); ok {
		return f64ToF32(f)
	}
	return nil
}

func loadTensorF32(gguf *GGUFModel, name string) ([]float32, int, int, error) {
	iface, err := gguf.LoadTensor(name)
	if err != nil {
		return nil, 0, 0, err
	}
	switch w := iface.(type) {
	case [][]float64:
		rows := len(w)
		cols := 0
		if rows > 0 {
			cols = len(w[0])
		}
		return flattenF64ToF32(w), rows, cols, nil
	}
	return nil, 0, 0, fmt.Errorf("model: %s must be 2D", name)
}

func (m *InferenceModelF32) initKVCaches() {
	m.KVCaches = make([]KVCacheF32, m.Config.NLayers)
	for i := range m.KVCaches {
		m.KVCaches[i] = KVCacheF32{
			K: make([][]float32, m.nKVHeads),
			V: make([][]float32, m.nKVHeads),
		}
	}
	dModel := m.Config.DModel
	dFF := m.Config.DFF
	nHeads := m.nHeads
	dK := m.dK
	b := &m.bufs
	b.normed = make([]float32, dModel)
	b.ffnNormed = make([]float32, dModel)
	b.outputNorm = make([]float32, dModel)
	b.qData = make([]float32, nHeads*dK)
	b.kData = make([]float32, m.nKVHeads*dK)
	b.vData = make([]float32, m.nKVHeads*dK)
	b.attnOut = make([]float32, nHeads*dK)
	b.merged = make([]float32, nHeads*dK)
	b.oData = make([]float32, dModel)
	b.gateData = make([]float32, dFF)
	b.upData = make([]float32, dFF)
	b.siluData = make([]float32, dFF)
	b.downData = make([]float32, dModel)
	b.logits = make([]float32, m.Config.VocabSize)
}

func (m *InferenceModelF32) Forward(tokenIDs []int, startPos int) []float32 {
	seqLen := len(tokenIDs)
	dModel := m.Config.DModel

	m.xDataBuf = growSlice(m.xDataBuf, seqLen*dModel)
	xData := m.xDataBuf[:seqLen*dModel]
	for i, tid := range tokenIDs {
		if tid < m.EmbRows {
			copy(xData[i*dModel:], m.TokenEmb[tid*m.EmbCols:tid*m.EmbCols+dModel])
		}
	}

	for i := 0; i < m.Config.NLayers; i++ {
		xData = m.forwardBlock(i, xData, seqLen, dModel, startPos)
	}

	logits := m.outputLogits(xData, seqLen, dModel)
	return logits
}

func (m *InferenceModelF32) forwardBlock(blockIdx int, xData []float32, seqLen, dModel, startPos int) []float32 {
	block := m.Blocks[blockIdx]
	cache := &m.KVCaches[blockIdx]
	eps := float32(m.Config.NormEps)
	b := &m.bufs

	b.normed = growSlice(b.normed, seqLen*dModel)
	rmsNormFlatF32(xData, block.AttnNormW, eps, seqLen, dModel, b.normed)

	qData, qRows, qCols := modelMatmulIntoF32(block.WQ, b.normed, seqLen, dModel, b.qData)
	b.qData = qData
	kData, kRows, kCols := modelMatmulIntoF32(block.WK, b.normed, seqLen, dModel, b.kData)
	b.kData = kData
	vData, vRows, vCols := modelMatmulIntoF32(block.WV, b.normed, seqLen, dModel, b.vData)
	b.vData = vData

	qHeads := splitHeadsDataF32(qData, qRows, qCols, m.nHeads)
	kHeads := splitHeadsDataF32(kData, kRows, kCols, m.nKVHeads)
	vHeads := splitHeadsDataF32(vData, vRows, vCols, m.nKVHeads)

	if block.QNormW != nil {
		for h := 0; h < m.nHeads; h++ {
			rmsNormFlatF32(qHeads[h], block.QNormW, eps, qRows, m.dK, qHeads[h])
		}
	}
	if block.KNormW != nil {
		for h := 0; h < m.nKVHeads; h++ {
			rmsNormFlatF32(kHeads[h], block.KNormW, eps, kRows, m.dK, kHeads[h])
		}
	}

	for h := 0; h < m.nHeads; h++ {
		applyRopeInPlaceF32(qHeads[h], qRows, m.dK, startPos, m.ropeFreqs, m.ropeHalfDim)
	}
	for h := 0; h < m.nKVHeads; h++ {
		applyRopeInPlaceF32(kHeads[h], kRows, m.dK, startPos, m.ropeFreqs, m.ropeHalfDim)
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
		kHeads = repeatKVDataF32(kHeads, m.nRep)
		vHeads = repeatKVDataF32(vHeads, m.nRep)
	}

	kvLen := len(kHeads[0]) / m.dK
	cacheLen := kvLen - seqLen
	attnSize := seqLen * m.nHeads * m.dK
	b.attnOut = growSlice(b.attnOut, attnSize)
	for h := 0; h < m.nHeads; h++ {
		fusedAttentionHeadF32(qHeads[h], kHeads[h], vHeads[h], seqLen, m.dK, kvLen, true, cacheLen, b.attnOut, h*seqLen*m.dK)
	}

	merged := mergeHeadsDataF32(b.attnOut, seqLen, m.nHeads, m.dK)
	oData, _, _ := modelMatmulIntoF32(block.WO, merged, seqLen, m.nHeads*m.dK, b.oData)
	b.oData = oData

	for i := range xData {
		xData[i] += oData[i]
	}

	b.ffnNormed = growSlice(b.ffnNormed, seqLen*dModel)
	rmsNormFlatF32(xData, block.FFNNormW, eps, seqLen, dModel, b.ffnNormed)

	gateData, gateRows, gateCols := modelMatmulIntoF32(block.WGate, b.ffnNormed, seqLen, dModel, b.gateData)
	b.gateData = gateData
	upData, _, _ := modelMatmulIntoF32(block.WUp, b.ffnNormed, seqLen, dModel, b.upData)
	b.upData = upData

	hidden := gateRows * gateCols
	b.siluData = growSlice(b.siluData, hidden)
	for i := 0; i < hidden; i++ {
		g := float64(gateData[i])
		s := float32(1.0 / (1.0 + math.Exp(-g)))
		b.siluData[i] = gateData[i] * s * upData[i]
	}

	downData, _, _ := modelMatmulIntoF32(block.WDown, b.siluData, gateRows, gateCols, b.downData)
	b.downData = downData

	for i := range downData {
		xData[i] += downData[i]
	}

	return xData
}

func (m *InferenceModelF32) outputLogits(xData []float32, seqLen, dModel int) []float32 {
	lastOff := (seqLen - 1) * dModel
	eps := float32(m.Config.NormEps)

	var ss float32
	for j := 0; j < dModel; j++ {
		v := xData[lastOff+j]
		ss += v * v
	}
	inv := 1.0 / float32(math.Sqrt(float64(ss)/float64(dModel)+float64(eps)))

	b := &m.bufs
	b.outputNorm = growSlice(b.outputNorm, dModel)
	for j := 0; j < dModel; j++ {
		b.outputNorm[j] = xData[lastOff+j] * inv * m.FinalNormW[j]
	}

	switch w := m.OutputW.(type) {
	case *QuantWeight:
		return modelMatmulRowQuantIntoF32(w, b.outputNorm, b.logits)
	case []float32:
		return modelMatmulRowFloatF32(w, b.outputNorm)
	}

	return nil
}

func (m *InferenceModelF32) Generate(prompt string, maxTokens int, strategy string, temperature float64, topK int, topP float64, repeatPenalty float64, repeatLastN int, systemPrompt string, templateName string, kvStartPos int) (string, int, int, int) {
	tGenStart := time.Now()
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

	logitsF32 := m.Forward(context, kvStartPos)
	tPrefillEnd := time.Now()
	m.PrefillMs = tPrefillEnd.Sub(tGenStart).Seconds() * 1000

	var recent []int64
	var nextID int
	if strategy == "greedy" {
		nextID = argmaxF32(logitsF32)
		recent = make([]int64, len(tokenIDs))
		for i, id := range tokenIDs {
			recent[i] = int64(id)
		}
	} else {
		logits := f32ToF64(logitsF32)
		recent = make([]int64, len(tokenIDs))
		for i, id := range tokenIDs {
			recent[i] = int64(id)
		}
		if repeatPenalty > 0 && len(recent) > 0 {
			applyRepeatPenalty(logits, recent, repeatPenalty, repeatLastN)
		}
		nextID = sampleLogits(logits, strategy, temperature, topK, topP)
	}

	tokenIDs = append(tokenIDs, nextID)
	recent = append(recent, int64(nextID))

	if nextID == m.Tokenizer.EOSID {
		finalPos := kvStartPos + nPrompt
		return m.Tokenizer.Decode(tokenIDs[nPrompt:]), 1, nPrompt, finalPos
	}

	nGen := 1
	for step := 1; step < maxTokens; step++ {
		pos := kvStartPos + nPrompt + step - 1

		logitsF32 = m.Forward([]int{nextID}, pos)

		if strategy == "greedy" {
			nextID = argmaxF32(logitsF32)
		} else {
			logits := f32ToF64(logitsF32)
			if repeatPenalty > 0 && len(recent) > 0 {
				applyRepeatPenalty(logits, recent, repeatPenalty, repeatLastN)
			}
			nextID = sampleLogits(logits, strategy, temperature, topK, topP)
		}

		tokenIDs = append(tokenIDs, nextID)
		recent = append(recent, int64(nextID))
		nGen++

		if nextID == m.Tokenizer.EOSID {
			break
		}
	}

	finalPos := kvStartPos + nPrompt + nGen - 1
	tDecodeEnd := time.Now()
	m.DecodeMs = tDecodeEnd.Sub(tGenStart).Seconds()*1000 - m.PrefillMs
	return m.Tokenizer.Decode(tokenIDs[nPrompt:]), nGen, nPrompt, finalPos
}
