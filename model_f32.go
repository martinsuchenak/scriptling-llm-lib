package scriptlingllmlib

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// validUTF8PrefixLen returns the length of the longest prefix of s that ends on
// a UTF-8 rune boundary, i.e. excluding any trailing bytes of an incomplete
// multi-byte rune. Byte-level BPE can split one rune across several tokens, so
// the streaming path holds back a partial trailing rune until it completes.
func validUTF8PrefixLen(s string) int {
	n := len(s)
	for i := 1; i <= utf8.UTFMax && i <= n; i++ {
		if utf8.RuneStart(s[n-i]) {
			if utf8.ValidString(s[n-i:]) {
				return n
			}
			return n - i
		}
	}
	return n
}

type modelCacheF32 struct {
	mu       sync.Mutex
	models   map[string]*InferenceModelF32
	sessions map[string]map[string]*sessionEntryF32
}

type sessionEntryF32 struct {
	model *InferenceModelF32 // clone holding this session's KV cache + scratch
	pos   int                // KV position (tokens seen so far)
	mu    sync.Mutex         // serializes turns of this session
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

	gguf.ReleaseFileData()

	c.models[path] = model
	return model, nil
}

func (c *modelCacheF32) clearModels() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.models = make(map[string]*InferenceModelF32)
	c.sessions = make(map[string]map[string]*sessionEntryF32)
}

// getOrCreateSession returns the session's persistent clone, creating it (with a
// fresh KV cache that shares the model's weights) on first use.
func (c *modelCacheF32) getOrCreateSession(modelPath, sessionID string, shared *InferenceModelF32) *sessionEntryF32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	sessions := c.sessions[modelPath]
	if sessions == nil {
		sessions = make(map[string]*sessionEntryF32)
		c.sessions[modelPath] = sessions
	}
	if e := sessions[sessionID]; e != nil {
		return e
	}
	e := &sessionEntryF32{model: shared.clone()}
	sessions[sessionID] = e
	return e
}

func (c *modelCacheF32) clearSession(modelPath, sessionID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if sessions, ok := c.sessions[modelPath]; ok {
		delete(sessions, sessionID)
		if len(sessions) == 0 {
			delete(c.sessions, modelPath)
		}
	}
}

// runGenerate runs one generation against the shared (read-only) model. Without
// a session it uses a throwaway clone; with a session it uses that session's
// persistent clone and serializes the session's turns. Safe to call concurrently
// for different (or empty) session IDs. ctx cancels generation between decode
// steps; pass context.Background() for no cancellation.
func runGenerate(ctx context.Context, onToken func(string), shared *InferenceModelF32, modelPath, prompt string, maxTokens int, strategy string, temperature float64, topK int, topP, repeatPenalty float64, repeatLastN int, systemPrompt, templateName, sessionID string) (result string, nGen, nPrompt int, prefillMs, decodeMs float64) {
	if sessionID == "" {
		m := shared.clone()
		m.ctx = ctx
		m.onToken = onToken
		result, nGen, nPrompt, _ = m.Generate(prompt, maxTokens, strategy, temperature, topK, topP, repeatPenalty, repeatLastN, systemPrompt, templateName, 0)
		return result, nGen, nPrompt, m.PrefillMs, m.DecodeMs
	}
	entry := globalModelCacheF32.getOrCreateSession(modelPath, sessionID, shared)
	entry.mu.Lock()
	defer entry.mu.Unlock()
	entry.model.ctx = ctx
	entry.model.onToken = onToken
	defer func() { entry.model.ctx = nil; entry.model.onToken = nil }()
	var finalPos int
	result, nGen, nPrompt, finalPos = entry.model.Generate(prompt, maxTokens, strategy, temperature, topK, topP, repeatPenalty, repeatLastN, systemPrompt, templateName, entry.pos)
	entry.pos = finalPos
	return result, nGen, nPrompt, entry.model.PrefillMs, entry.model.DecodeMs
}

func buildInferenceModelF32(gguf *GGUFModel, path string) (*InferenceModelF32, error) {
	cfg := gguf.Config
	gguf.Metadata["_path"] = path

	tokenEmbF32, embRows, embCols, err := gguf.loadTensor2DF32("token_embedding.weight")
	if err != nil {
		return nil, fmt.Errorf("model: loading token_embedding: %w", err)
	}

	finalNormF32, err := gguf.loadTensor1DF32("final_norm.weight")
	if err != nil {
		return nil, fmt.Errorf("model: loading final_norm: %w", err)
	}

	var outputW interface{}
	outputIface, err := gguf.loadWeightF32Direct("output.weight")
	if err != nil || outputIface == nil {
		if embCols%32 == 0 {
			qw := quantizeQ8RowsF32(tokenEmbF32, embRows, embCols)
			if qw != nil {
				outputW = qw
			} else {
				outputW = tokenEmbF32
			}
		} else {
			outputW = tokenEmbF32
		}
	} else {
		outputW = outputIface
	}

	blocks := make([]TransformerBlockF32, cfg.NLayers)
	for i := 0; i < cfg.NLayers; i++ {
		prefix := fmt.Sprintf("blocks.%d.", i)

		attnNormF32, err := gguf.loadTensor1DF32(prefix + "attn_norm.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn_norm.weight", err)
		}

		wq, err := gguf.loadWeightF32Direct(prefix + "attn.w_q.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_q.weight", err)
		}
		wk, err := gguf.loadWeightF32Direct(prefix + "attn.w_k.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_k.weight", err)
		}
		wv, err := gguf.loadWeightF32Direct(prefix + "attn.w_v.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_v.weight", err)
		}
		wo, err := gguf.loadWeightF32Direct(prefix + "attn.w_o.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"attn.w_o.weight", err)
		}

		ffnNormF32, err := gguf.loadTensor1DF32(prefix + "ffn_norm.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn_norm.weight", err)
		}

		wGate, err := gguf.loadWeightF32Direct(prefix + "ffn.w_gate.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn.w_gate.weight", err)
		}
		wUp, err := gguf.loadWeightF32Direct(prefix + "ffn.w_up.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn.w_up.weight", err)
		}
		wDown, err := gguf.loadWeightF32Direct(prefix + "ffn.w_down.weight")
		if err != nil {
			return nil, fmt.Errorf("model: loading %s: %w", prefix+"ffn.w_down.weight", err)
		}

		qNormW, _ := gguf.loadTensor1DF32(prefix + "attn_q_norm.weight")
		kNormW, _ := gguf.loadTensor1DF32(prefix + "attn_k_norm.weight")

		var qBias, kBias, vBias []float32
		blkPrefix := fmt.Sprintf("blk.%d.", i)
		qBias, _ = gguf.loadTensor1DF32(blkPrefix + "attn_q.bias")
		kBias, _ = gguf.loadTensor1DF32(blkPrefix + "attn_k.bias")
		vBias, _ = gguf.loadTensor1DF32(blkPrefix + "attn_v.bias")
		if qBias == nil {
			qBias, _ = gguf.loadTensor1DF32(prefix + "attn.q.bias")
		}
		if kBias == nil {
			kBias, _ = gguf.loadTensor1DF32(prefix + "attn.k.bias")
		}
		if vBias == nil {
			vBias, _ = gguf.loadTensor1DF32(prefix + "attn.v.bias")
		}

		blocks[i] = TransformerBlockF32{
			AttnNormW: attnNormF32,
			WQ:        wq,
			WK:        wk,
			WV:        wv,
			WO:        wo,
			QBias:     qBias,
			KBias:     kBias,
			VBias:     vBias,
			QNormW:    qNormW,
			KNormW:    kNormW,
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

// clone returns an inference instance that shares this model's read-only weights
// (the big tensors, tokenizer, rope tables — no copy) but has its own mutable
// state: KV cache, scratch buffers, and timing stats. Each concurrent request
// runs on its own clone, so requests never corrupt one another. Cheap: only the
// per-request scratch (~hundreds of KB) is allocated, not the weights.
func (m *InferenceModelF32) clone() *InferenceModelF32 {
	c := *m // shallow copy shares all weight slices by reference
	c.KVCaches = nil
	c.bufs = blockBuffers{}
	c.xDataBuf = nil
	c.PrefillMs = 0
	c.DecodeMs = 0
	c.initKVCaches()
	return &c
}

// runBlocks embeds the tokens and runs them through every transformer block,
// returning the per-position hidden states (seqLen*dModel) prior to the output
// projection. Shared by Forward (which projects to logits) and the embedding path
// (which pools the hidden states).
func (m *InferenceModelF32) runBlocks(tokenIDs []int, startPos int) (xData []float32, seqLen, dModel int) {
	seqLen = len(tokenIDs)
	dModel = m.Config.DModel

	m.xDataBuf = growSlice(m.xDataBuf, seqLen*dModel)
	xData = m.xDataBuf[:seqLen*dModel]
	for i, tid := range tokenIDs {
		if tid < m.EmbRows {
			copy(xData[i*dModel:], m.TokenEmb[tid*m.EmbCols:tid*m.EmbCols+dModel])
		}
	}

	for i := 0; i < m.Config.NLayers; i++ {
		xData = m.forwardBlock(i, xData, seqLen, dModel, startPos)
	}
	return xData, seqLen, dModel
}

func (m *InferenceModelF32) Forward(tokenIDs []int, startPos int) []float32 {
	xData, seqLen, dModel := m.runBlocks(tokenIDs, startPos)
	return m.outputLogits(xData, seqLen, dModel)
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

	if block.QBias != nil {
		for s := 0; s < qRows; s++ {
			off := s * qCols
			for j := 0; j < len(block.QBias) && j < qCols; j++ {
				qData[off+j] += block.QBias[j]
			}
		}
	}
	if block.KBias != nil {
		for s := 0; s < kRows; s++ {
			off := s * kCols
			for j := 0; j < len(block.KBias) && j < kCols; j++ {
				kData[off+j] += block.KBias[j]
			}
		}
	}
	if block.VBias != nil {
		for s := 0; s < vRows; s++ {
			off := s * vCols
			for j := 0; j < len(block.VBias) && j < vCols; j++ {
				vData[off+j] += block.VBias[j]
			}
		}
	}

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
	enterInference()
	defer exitInference()
	ctx := m.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	tGenStart := time.Now()
	if kvStartPos == 0 {
		if templateName != "" {
			if tpl, ok := defaultTemplates[templateName]; ok {
				prompt = applyChatTemplate(tpl, prompt, systemPrompt, m.Arch)
			}
		} else if m.ChatTpl != "" {
			prompt = applyChatTemplate(m.ChatTpl, prompt, systemPrompt, m.Arch)
		} else {
			tpl := defaultTemplates["chatml"]
			prompt = applyChatTemplate(tpl, prompt, systemPrompt, m.Arch)
		}
	} else {
		if m.ChatTpl != "" {
			prompt = applyContinuation(m.ChatTpl, prompt, m.Arch)
		} else {
			prompt = "<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
		}
	}

	tokenIDs := m.Tokenizer.Encode(prompt)
	nPrompt := len(tokenIDs)

	// Streaming emitter: on each call, decode the generated tokens so far and
	// emit the new, UTF-8-safe suffix as a delta. final=true flushes any held
	// trailing bytes. Leading whitespace is trimmed to match Decode's output.
	emitted := 0
	leadingDone := false
	emit := func(final bool) {
		if m.onToken == nil {
			return
		}
		full := m.Tokenizer.decodeRaw(tokenIDs[nPrompt:])
		end := len(full)
		if !final && end > emitted {
			end = emitted + validUTF8PrefixLen(full[emitted:])
		}
		if end <= emitted {
			return
		}
		chunk := full[emitted:end]
		emitted = end
		if !leadingDone {
			chunk = strings.TrimLeft(chunk, " ")
			if chunk != "" {
				leadingDone = true
			}
		}
		if chunk != "" {
			m.onToken(chunk)
		}
	}

	if kvStartPos == 0 {
		m.initKVCaches()
	}

	// Cancelled before any work: leave the KV cache untouched.
	if ctx.Err() != nil {
		return "", 0, nPrompt, kvStartPos
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
		emit(true)
		output := m.Tokenizer.Decode(tokenIDs[nPrompt:])
		if m.Arch == "qwen3" {
			output = stripThinkTags(output)
		}
		return output, 1, nPrompt, finalPos
	}
	emit(false)

	nGen := 1
	for step := 1; step < maxTokens; step++ {
		if ctx.Err() != nil {
			break
		}
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
		emit(false)
	}
	emit(true)

	finalPos := kvStartPos + nPrompt + nGen - 1
	tDecodeEnd := time.Now()
	m.DecodeMs = tDecodeEnd.Sub(tGenStart).Seconds()*1000 - m.PrefillMs
	output := m.Tokenizer.Decode(tokenIDs[nPrompt:])
	if m.Arch == "qwen3" {
		output = stripThinkTags(output)
	}
	return output, nGen, nPrompt, finalPos
}

func stripThinkTags(s string) string {
	for {
		start := strings.Index(s, "<think")
		if start == -1 {
			break
		}
		end := strings.Index(s[start:], "</think")
		if end == -1 {
			s = s[:start]
			break
		}
		s = s[:start] + s[start+end+len("</think"):]
	}
	return strings.TrimSpace(s)
}
