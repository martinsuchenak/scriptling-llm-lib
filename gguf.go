package scriptlingllmlib

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
)

const ggufMagic = 0x46554747

// kquantPacked enables the native packed Q4_K/Q5_K/Q6_K kernels instead of
// dequantizing to dense float32. Opt-in (SLLM_KQUANT_PACKED) until the packed
// SIMD kernels make it the faster default; today it trades speed for ~4-6x less
// memory, which is what lets large k-quant models fit on RAM-limited hosts.
var kquantPacked = os.Getenv("SLLM_KQUANT_PACKED") != ""

var ggufTypeSizes = map[int]int{
	0: 4,
	1: 2,
	2: 18,
	3: 20,
	6: 22,
	7: 24,
	8: 34,
	9: 40,
}

type tensorInfo struct {
	Name      string
	Dims      []uint64
	Type      uint32
	Offset    uint64
	RawOffset int64
}

type GGUFModel struct {
	Version    uint32
	Metadata   map[string]interface{}
	Tensors    map[string]*tensorInfo
	Config     ModelConfig
	Tokenizer  *TokenizerData
	File       *os.File
	DataOffset int64
	fileData   []byte
	unmap      func() error // non-nil when fileData is a memory map
}

type ModelConfig struct {
	VocabSize    int
	DModel       int
	NHeads       int
	NKVHeads     int
	NLayers      int
	MaxSeqLen    int
	DFF          int
	NormEps      float64
	RopeFreqBase float64
	RopeDim      int
}

type TokenizerData struct {
	Tokens       []string
	Scores       []float64
	Merges       [][2]string
	Special      map[string]int
	ChatTemplate string
	Vocab        map[string]int
	Type         string
}

type QuantWeight struct {
	QType  string
	Raw    []byte
	Groups int
	Rows   int
	Cols   int
}

type ggufReader struct {
	data []byte
	pos  int
	bad  bool // set once any read runs past the end of data
}

func newGGUFReader(data []byte) *ggufReader {
	return &ggufReader{data: data}
}

// ensure reports whether n more bytes are available at the cursor. Once a read
// has gone out of bounds the reader is "bad" and every subsequent read is a
// no-op returning a zero value, so callers can parse optimistically and check
// r.bad once at the end instead of after every read.
func (r *ggufReader) ensure(n int) bool {
	if r.bad || n < 0 || r.pos+n > len(r.data) {
		r.bad = true
		return false
	}
	return true
}

func (r *ggufReader) readUint8() uint8 {
	if !r.ensure(1) {
		return 0
	}
	v := r.data[r.pos]
	r.pos++
	return v
}

func (r *ggufReader) readInt8() int8 {
	return int8(r.readUint8())
}

func (r *ggufReader) readUint16() uint16 {
	if !r.ensure(2) {
		return 0
	}
	v := binary.LittleEndian.Uint16(r.data[r.pos:])
	r.pos += 2
	return v
}

func (r *ggufReader) readInt16() int16 {
	return int16(r.readUint16())
}

func (r *ggufReader) readUint32() uint32 {
	if !r.ensure(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(r.data[r.pos:])
	r.pos += 4
	return v
}

func (r *ggufReader) readInt32() int32 {
	return int32(r.readUint32())
}

func (r *ggufReader) readUint64() uint64 {
	if !r.ensure(8) {
		return 0
	}
	v := binary.LittleEndian.Uint64(r.data[r.pos:])
	r.pos += 8
	return v
}

func (r *ggufReader) readInt64() int64 {
	return int64(r.readUint64())
}

func (r *ggufReader) readFloat32() float32 {
	if !r.ensure(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(r.data[r.pos:])
	r.pos += 4
	return math.Float32frombits(v)
}

func (r *ggufReader) readFloat64() float64 {
	if !r.ensure(8) {
		return 0
	}
	v := binary.LittleEndian.Uint64(r.data[r.pos:])
	r.pos += 8
	return math.Float64frombits(v)
}

func (r *ggufReader) readString() string {
	length := r.readUint64()
	if r.bad || length == 0 {
		return ""
	}
	// A string can be no longer than the whole file; this also guards the
	// uint64->int conversion against overflow on a hostile length.
	if length > uint64(len(r.data)) {
		r.bad = true
		return ""
	}
	n := int(length)
	if !r.ensure(n) {
		return ""
	}
	s := string(r.data[r.pos : r.pos+n])
	r.pos += n
	return s
}

func (r *ggufReader) readValue(vtype uint32) interface{} {
	switch vtype {
	case 0:
		return r.readUint8()
	case 1:
		return r.readInt8()
	case 2:
		return r.readUint16()
	case 3:
		return r.readInt16()
	case 4:
		return r.readUint32()
	case 5:
		return r.readInt32()
	case 6:
		return float64(r.readFloat32())
	case 7:
		return r.readUint8() != 0
	case 8:
		return r.readString()
	case 10:
		return r.readUint64()
	case 11:
		return r.readInt64()
	case 12:
		return r.readFloat64()
	case 9:
		arrType := r.readUint32()
		arrLen := r.readUint64()
		if r.bad {
			return nil
		}
		// Each element is at least one byte, so an array can't have more
		// elements than the file has bytes. Cap the prealloc so a hostile
		// length can't force a giant up-front allocation; the slice still
		// grows to fit a genuine (bounded) array via append.
		if arrLen > uint64(len(r.data)) {
			r.bad = true
			return nil
		}
		capHint := arrLen
		if capHint > 4096 {
			capHint = 4096
		}
		result := make([]interface{}, 0, capHint)
		for i := uint64(0); i < arrLen; i++ {
			v := r.readValue(arrType)
			if r.bad {
				return nil
			}
			result = append(result, v)
		}
		return result
	default:
		// Unknown value type: we can't know its width, so fail rather than
		// desync the cursor.
		r.bad = true
		return nil
	}
}

func mapTensorName(name string) string {
	static := map[string]string{
		"token_embd.weight":  "token_embedding.weight",
		"output_norm.weight": "final_norm.weight",
		"output.weight":      "output.weight",
	}

	if mapped, ok := static[name]; ok {
		return mapped
	}

	patterns := []struct {
		prefix string
		suffix string
		tpl    string
	}{
		{"blk.", ".attn_norm.weight", "blocks.%d.attn_norm.weight"},
		{"blk.", ".attn_q.weight", "blocks.%d.attn.w_q.weight"},
		{"blk.", ".attn_k.weight", "blocks.%d.attn.w_k.weight"},
		{"blk.", ".attn_v.weight", "blocks.%d.attn.w_v.weight"},
		{"blk.", ".attn_output.weight", "blocks.%d.attn.w_o.weight"},
		{"blk.", ".attn_q_norm.weight", "blocks.%d.attn_q_norm.weight"},
		{"blk.", ".attn_k_norm.weight", "blocks.%d.attn_k_norm.weight"},
		{"blk.", ".ffn_norm.weight", "blocks.%d.ffn_norm.weight"},
		{"blk.", ".ffn_gate.weight", "blocks.%d.ffn.w_gate.weight"},
		{"blk.", ".ffn_down.weight", "blocks.%d.ffn.w_down.weight"},
		{"blk.", ".ffn_up.weight", "blocks.%d.ffn.w_up.weight"},
	}

	for _, p := range patterns {
		if len(name) > len(p.prefix)+len(p.suffix) &&
			name[:len(p.prefix)] == p.prefix &&
			name[len(name)-len(p.suffix):] == p.suffix {
			idxStr := name[len(p.prefix) : len(name)-len(p.suffix)]
			idx := 0
			for _, c := range idxStr {
				if c >= '0' && c <= '9' {
					idx = idx*10 + int(c-'0')
				} else {
					goto nextPattern
				}
			}
			return fmt.Sprintf(p.tpl, idx)
		}
	nextPattern:
	}

	return name
}

func metaInt(m map[string]interface{}, key string, defaultVal int) int {
	v, ok := m[key]
	if !ok {
		return defaultVal
	}
	switch val := v.(type) {
	case uint8:
		return int(val)
	case int8:
		return int(val)
	case uint16:
		return int(val)
	case int16:
		return int(val)
	case uint32:
		return int(val)
	case int32:
		return int(val)
	case uint64:
		return int(val)
	case int64:
		return int(val)
	case float64:
		return int(val)
	}
	return defaultVal
}

func metaFloat(m map[string]interface{}, key string, defaultVal float64) float64 {
	v, ok := m[key]
	if !ok {
		return defaultVal
	}
	switch val := v.(type) {
	case float64:
		return val
	case uint32:
		return float64(val)
	case int32:
		return float64(val)
	case uint64:
		return float64(val)
	case int64:
		return float64(val)
	}
	return defaultVal
}

func metaString(m map[string]interface{}, key string, defaultVal string) string {
	v, ok := m[key]
	if !ok {
		return defaultVal
	}
	if s, ok := v.(string); ok {
		return s
	}
	return defaultVal
}

func metaStringSlice(m map[string]interface{}, key string) []string {
	v, ok := m[key]
	if !ok {
		return nil
	}
	if arr, ok := v.([]interface{}); ok {
		result := make([]string, len(arr))
		for i, item := range arr {
			if s, ok := item.(string); ok {
				result[i] = s
			}
		}
		return result
	}
	return nil
}

func metaFloatSlice(m map[string]interface{}, key string) []float64 {
	v, ok := m[key]
	if !ok {
		return nil
	}
	if arr, ok := v.([]interface{}); ok {
		result := make([]float64, len(arr))
		for i, item := range arr {
			switch val := item.(type) {
			case float64:
				result[i] = val
			case uint32:
				result[i] = float64(val)
			case int32:
				result[i] = float64(val)
			case uint64:
				result[i] = float64(val)
			case int64:
				result[i] = float64(val)
			}
		}
		return result
	}
	return nil
}

func dequantizeF32(data []byte, offset int, nElements int) []float64 {
	result := make([]float64, nElements)
	for i := 0; i < nElements; i++ {
		bits := binary.LittleEndian.Uint32(data[offset+i*4:])
		result[i] = float64(math.Float32frombits(bits))
	}
	return result
}

func dequantizeF16(data []byte, offset int, nElements int) []float64 {
	result := make([]float64, nElements)
	for i := 0; i < nElements; i++ {
		bits := binary.LittleEndian.Uint16(data[offset+i*2:])
		result[i] = float16ToFloat64(bits)
	}
	return result
}

func dequantizeQ8_0Native(data []byte, offset int, nElements int) []float64 {
	nGroups := nElements / 32
	result := make([]float64, nElements)
	for g := 0; g < nGroups; g++ {
		base := offset + g*34
		scale := float16ToFloat64(binary.LittleEndian.Uint16(data[base:]))
		for i := 0; i < 32; i++ {
			result[g*32+i] = float64(int8(data[base+2+i])) * scale
		}
	}
	return result
}

func dequantizeQ4_0Native(data []byte, offset int, nElements int) []float64 {
	nGroups := nElements / 32
	result := make([]float64, nElements)
	for g := 0; g < nGroups; g++ {
		base := offset + g*18
		scale := float16ToFloat64(binary.LittleEndian.Uint16(data[base:]))
		for i := 0; i < 16; i++ {
			b := data[base+2+i]
			qLow := float64(int8(b&0x0F) - 8)
			qHigh := float64(int8((b>>4)&0x0F) - 8)
			result[g*32+i] = qLow * scale
			result[g*32+16+i] = qHigh * scale
		}
	}
	return result
}

func dequantizeQ4_1Native(data []byte, offset int, nElements int) []float64 {
	nGroups := nElements / 32
	result := make([]float64, nElements)
	for g := 0; g < nGroups; g++ {
		base := offset + g*20
		d := float16ToFloat64(binary.LittleEndian.Uint16(data[base:]))
		m := float16ToFloat64(binary.LittleEndian.Uint16(data[base+2:]))
		for i := 0; i < 16; i++ {
			b := data[base+4+i]
			qLow := float64(b & 0x0F)
			qHigh := float64((b >> 4) & 0x0F)
			result[g*32+i] = qLow*d + m
			result[g*32+16+i] = qHigh*d + m
		}
	}
	return result
}

func dequantizeQ5_0Native(data []byte, offset int, nElements int) []float64 {
	nGroups := nElements / 32
	result := make([]float64, nElements)
	for g := 0; g < nGroups; g++ {
		base := offset + g*22
		d := float16ToFloat64(binary.LittleEndian.Uint16(data[base:]))
		qh := binary.LittleEndian.Uint32(data[base+2:])
		for j := 0; j < 16; j++ {
			q := data[base+6+j]
			xh0 := int32((qh>>uint(j))<<4) & 0x10
			xh1 := int32(qh>>uint(j+12)) & 0x10
			x0 := (int32(q&0x0F) | xh0) - 16
			x1 := (int32(q>>4) | xh1) - 16
			result[g*32+j] = float64(x0) * d
			result[g*32+j+16] = float64(x1) * d
		}
	}
	return result
}

func LoadGGUF(path string) (*GGUFModel, error) {
	// Prefer a read-only memory map (cheap, file-backed); fall back to a full
	// read where mmap is unavailable (e.g. Windows) or fails.
	fileData, unmap, err := mmapFile(path)
	if err != nil {
		fileData, err = os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("gguf: failed to read file: %w", err)
		}
		unmap = nil
	}

	g, perr := parseGGUF(fileData)
	if perr != nil {
		if unmap != nil {
			_ = unmap()
		}
		return nil, perr
	}

	if unmap != nil {
		g.unmap = unmap
		// Safety net: unmap if the caller forgets ReleaseFileData.
		runtime.SetFinalizer(g, (*GGUFModel).ReleaseFileData)
	}
	return g, nil
}

// parseGGUF parses an in-memory GGUF image. It treats the bytes as untrusted:
// every read is bounds-checked and every attacker-controlled count/length is
// capped against the image size, so malformed input yields an error, never a
// panic or unbounded allocation.
func parseGGUF(fileData []byte) (*GGUFModel, error) {
	r := newGGUFReader(fileData)

	magic := r.readUint32()
	if magic != ggufMagic {
		return nil, fmt.Errorf("gguf: not a GGUF file (magic=0x%08X)", magic)
	}

	version := r.readUint32()
	nTensors := r.readUint64()
	nMetadata := r.readUint64()

	// Each metadata entry and tensor descriptor occupies several bytes, so the
	// file can't contain more of them than it has bytes. This caps the counts
	// (which are attacker-controlled) before they drive any allocation.
	if r.bad || nTensors > uint64(len(fileData)) || nMetadata > uint64(len(fileData)) {
		return nil, fmt.Errorf("gguf: malformed header (tensors=%d, metadata=%d)", nTensors, nMetadata)
	}

	metadata := make(map[string]interface{}, 0)
	for i := uint64(0); i < nMetadata; i++ {
		key := r.readString()
		vtype := r.readUint32()
		val := r.readValue(vtype)
		if r.bad {
			return nil, fmt.Errorf("gguf: truncated or malformed metadata")
		}
		metadata[key] = val
	}

	tensorInfos := make([]*tensorInfo, 0, nTensors)
	for i := uint64(0); i < nTensors; i++ {
		name := r.readString()
		nDims := r.readUint32()
		if r.bad || uint64(nDims) > uint64(len(fileData)) {
			return nil, fmt.Errorf("gguf: truncated or malformed tensor info")
		}
		dims := make([]uint64, nDims)
		for d := uint32(0); d < nDims; d++ {
			dims[d] = r.readUint64()
		}
		ggufType := r.readUint32()
		tOffset := r.readUint64()
		if r.bad {
			return nil, fmt.Errorf("gguf: truncated or malformed tensor info")
		}
		mapped := mapTensorName(name)
		tensorInfos = append(tensorInfos, &tensorInfo{
			Name:   mapped,
			Dims:   dims,
			Type:   ggufType,
			Offset: tOffset,
		})
	}

	alignment := uint64(metaInt(metadata, "general.alignment", 32))
	dataStart := uint64(r.pos)
	if dataStart%alignment != 0 {
		dataStart = dataStart + (alignment - (dataStart % alignment))
	}

	arch := metaString(metadata, "general.architecture", "llama")

	vocabSize := metaInt(metadata, arch+".vocab_size", metaInt(metadata, "tokenizer.ggml.n_tokens", 0))
	nLayers := metaInt(metadata, arch+".block_count", 0)
	nHeads := metaInt(metadata, arch+".attention.head_count", 0)
	nKVHeads := metaInt(metadata, arch+".attention.head_count_kv", nHeads)
	dim := metaInt(metadata, arch+".embedding_length", 0)
	hiddenDim := metaInt(metadata, arch+".feed_forward_length", 0)
	maxSeqLen := metaInt(metadata, arch+".context_length", 512)
	normEps := metaFloat(metadata, arch+".attention.layer_norm_rms_epsilon", 1e-5)
	ropeFreqBase := metaFloat(metadata, arch+".rope.freq_base", 10000.0)
	ropeDim := metaInt(metadata, arch+".rope.dimension_count", 0)
	if ropeDim == 0 && nHeads > 0 {
		ropeDim = dim / nHeads
	}

	config := ModelConfig{
		VocabSize:    vocabSize,
		DModel:       dim,
		NHeads:       nHeads,
		NKVHeads:     nKVHeads,
		NLayers:      nLayers,
		MaxSeqLen:    maxSeqLen,
		DFF:          hiddenDim,
		NormEps:      normEps,
		RopeFreqBase: ropeFreqBase,
		RopeDim:      ropeDim,
	}

	tensorMap := make(map[string]*tensorInfo, nTensors)
	for i := uint64(0); i < nTensors; i++ {
		ti := tensorInfos[i]
		ti.RawOffset = int64(dataStart + ti.Offset)
		tensorMap[ti.Name] = ti
	}

	tokens := metaStringSlice(metadata, "tokenizer.ggml.tokens")
	scores := metaFloatSlice(metadata, "tokenizer.ggml.scores")
	merges := metaStringSlice(metadata, "tokenizer.ggml.merges")
	chatTemplate := metaString(metadata, "tokenizer.chat_template", "")

	var tokenizerData *TokenizerData
	if len(tokens) > 0 {
		vocab := make(map[string]int, len(tokens))
		special := make(map[string]int)
		for i, tok := range tokens {
			vocab[tok] = i
			if len(tok) > 2 && tok[0] == '<' && tok[len(tok)-1] == '>' {
				special[tok] = i
			}
		}

		var mergeList [][2]string
		for _, m := range merges {
			parts := splitTwo(m, " ")
			if len(parts) == 2 {
				mergeList = append(mergeList, [2]string{parts[0], parts[1]})
			}
		}

		tokType := "simple"
		if len(mergeList) > 0 {
			tokType = "bpe"
		}

		tokenizerData = &TokenizerData{
			Tokens:       tokens,
			Scores:       scores,
			Merges:       mergeList,
			Special:      special,
			ChatTemplate: chatTemplate,
			Vocab:        vocab,
			Type:         tokType,
		}
	} else {
		tokenizerData = &TokenizerData{
			Tokens: nil,
			Scores: nil,
			Merges: nil,
			Special: map[string]int{
				"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
			},
			Vocab:        map[string]int{},
			Type:         "simple",
			ChatTemplate: "",
		}
	}

	gguf := &GGUFModel{
		Version:    version,
		Metadata:   metadata,
		Tensors:    tensorMap,
		Config:     config,
		Tokenizer:  tokenizerData,
		DataOffset: int64(dataStart),
		fileData:   fileData,
	}

	return gguf, nil
}

// ReleaseFileData frees the file image. If it was memory-mapped the mapping is
// unmapped; otherwise the byte slice is simply dropped for the GC. Safe to call
// more than once. After this, LoadTensor re-reads the file on demand.
func (g *GGUFModel) ReleaseFileData() {
	if g.unmap != nil {
		_ = g.unmap()
		g.unmap = nil
		runtime.SetFinalizer(g, nil)
	}
	g.fileData = nil
}

func (g *GGUFModel) LoadTensor(name string) (interface{}, error) {
	ti, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}

	if len(ti.Dims) == 0 {
		return []float64{}, nil
	}

	nElements := uint64(1)
	for _, d := range ti.Dims {
		nElements *= d
	}

	if nElements == 0 {
		if len(ti.Dims) == 1 {
			return []float64{}, nil
		}
		return [][]float64{{}}, nil
	}

	fileData := g.fileData
	if fileData == nil {
		var err error
		fileData, err = os.ReadFile(g.Metadata["_path"].(string))
		if err != nil {
			return nil, fmt.Errorf("gguf: failed to re-read file: %w", err)
		}
	}

	if len(ti.Dims) == 1 {
		return g.dequantize1D(fileData, ti, int(nElements)), nil
	}

	actualRows := int(ti.Dims[1])
	actualCols := int(ti.Dims[0])

	if name == "token_embedding.weight" {
		flat := g.dequantize1D(fileData, ti, int(nElements))
		return reshape2D(flat, actualRows, actualCols), nil
	}

	return g.loadQuantized2D(fileData, ti, actualRows, actualCols, int(nElements)), nil
}

func (g *GGUFModel) loadTensor1DF32(name string) ([]float32, error) {
	ti, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	if len(ti.Dims) == 0 || len(ti.Dims) > 1 {
		return nil, fmt.Errorf("gguf: tensor %q is not 1D", name)
	}
	nElements := int(ti.Dims[0])
	if nElements == 0 {
		return nil, nil
	}
	offset := int(ti.RawOffset)
	fileData := g.fileData
	switch ti.Type {
	case 0:
		result := make([]float32, nElements)
		for i := 0; i < nElements; i++ {
			result[i] = math.Float32frombits(binary.LittleEndian.Uint32(fileData[offset+i*4:]))
		}
		return result, nil
	case 1:
		result := make([]float32, nElements)
		for i := 0; i < nElements; i++ {
			result[i] = readF16F32(fileData, offset+i*2)
		}
		return result, nil
	default:
		if !tensorTypeSupported(ti.Type) {
			return nil, fmt.Errorf("gguf: tensor %q has unsupported type %d", name, ti.Type)
		}
		f64 := g.dequantize1D(fileData, ti, nElements)
		return f64ToF32(f64), nil
	}
}

func (g *GGUFModel) loadTensor2DF32(name string) ([]float32, int, int, error) {
	ti, ok := g.Tensors[name]
	if !ok {
		return nil, 0, 0, fmt.Errorf("gguf: tensor %q not found", name)
	}
	if len(ti.Dims) != 2 {
		return nil, 0, 0, fmt.Errorf("gguf: tensor %q is not 2D", name)
	}
	rows := int(ti.Dims[1])
	cols := int(ti.Dims[0])
	nElements := rows * cols
	if nElements == 0 {
		return nil, 0, 0, nil
	}
	offset := int(ti.RawOffset)
	fileData := g.fileData
	switch ti.Type {
	case 0:
		result := make([]float32, nElements)
		for i := 0; i < nElements; i++ {
			result[i] = math.Float32frombits(binary.LittleEndian.Uint32(fileData[offset+i*4:]))
		}
		return result, rows, cols, nil
	case 1:
		result := make([]float32, nElements)
		for i := 0; i < nElements; i++ {
			result[i] = readF16F32(fileData, offset+i*2)
		}
		return result, rows, cols, nil
	default:
		if !tensorTypeSupported(ti.Type) {
			return nil, 0, 0, fmt.Errorf("gguf: tensor %q has unsupported type %d", name, ti.Type)
		}
		f64 := g.dequantize1D(fileData, ti, nElements)
		return f64ToF32(f64), rows, cols, nil
	}
}

func (g *GGUFModel) loadWeightF32Direct(name string) (interface{}, error) {
	ti, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	if len(ti.Dims) != 2 {
		return nil, fmt.Errorf("gguf: tensor %q is not 2D", name)
	}
	rows := int(ti.Dims[1])
	cols := int(ti.Dims[0])
	nElements := rows * cols
	if nElements == 0 {
		return nil, nil
	}
	offset := int(ti.RawOffset)
	fileData := g.fileData

	switch ti.Type {
	case 8:
		groupSize := 32
		groupsPerRow := cols / groupSize
		nGroups := nElements / groupSize
		totalBytes := nGroups * 34
		raw := make([]byte, totalBytes)
		copy(raw, fileData[offset:offset+totalBytes])
		return &QuantWeight{QType: "q8", Raw: raw, Groups: groupsPerRow, Rows: rows, Cols: cols}, nil
	case 2:
		groupSize := 32
		groupsPerRow := cols / groupSize
		nGroups := nElements / groupSize
		totalBytes := nGroups * 18
		raw := make([]byte, totalBytes)
		copy(raw, fileData[offset:offset+totalBytes])
		return &QuantWeight{QType: "q4", Raw: raw, Groups: groupsPerRow, Rows: rows, Cols: cols}, nil
	case 3:
		groupSize := 32
		groupsPerRow := cols / groupSize
		nGroups := nElements / groupSize
		totalBytes := nGroups * 20
		raw := make([]byte, totalBytes)
		copy(raw, fileData[offset:offset+totalBytes])
		return &QuantWeight{QType: "q4_1", Raw: raw, Groups: groupsPerRow, Rows: rows, Cols: cols}, nil
	case 6:
		groupSize := 32
		groupsPerRow := cols / groupSize
		nGroups := nElements / groupSize
		totalBytes := nGroups * 22
		raw := make([]byte, totalBytes)
		copy(raw, fileData[offset:offset+totalBytes])
		return &QuantWeight{QType: "q5", Raw: raw, Groups: groupsPerRow, Rows: rows, Cols: cols}, nil
	case 12, 13, 14:
		// Q4_K/Q5_K/Q6_K: keep the super-blocks packed and dequantize-and-dot on
		// the fly (kquant_kernels.go). ~4-6x fewer bytes/token than dense float
		// and full per-element precision (no Q8 requant). Needs cols a multiple of
		// the 256-element super-block (always true for k-quant tensors).
		//
		// Opt-in for now: the scalar kernel cuts memory ~4-6x (lets large k-quant
		// models fit on RAM-limited hosts) but is slower than the SIMD float matmul
		// until the packed SIMD kernels land, so dense float stays the default.
		if kquantPacked && cols%256 == 0 {
			var qt string
			var blockBytes int
			switch ti.Type {
			case 12:
				qt, blockBytes = "q4k", q4kBlockBytes
			case 13:
				qt, blockBytes = "q5k", q5kBlockBytes
			case 14:
				qt, blockBytes = "q6k", q6kBlockBytes
			}
			nSB := nElements / 256
			totalBytes := nSB * blockBytes
			rawW := make([]byte, totalBytes)
			copy(rawW, fileData[offset:offset+totalBytes])
			return &QuantWeight{QType: qt, Raw: rawW, Groups: cols / 256, Rows: rows, Cols: cols}, nil
		}
		fallthrough
	case 7, 10, 11:
		// Q5_1 fallback and Q2_K/Q3_K (no packed kernel yet): dense float32.
		flat := g.dequantize1D(fileData, ti, nElements)
		return f64ToF32(flat), nil
	}

	if !tensorTypeSupported(ti.Type) {
		return nil, fmt.Errorf("gguf: tensor %q has unsupported type %d", name, ti.Type)
	}
	f32, _, _, err := g.loadTensor2DF32(name)
	if err != nil {
		return nil, err
	}
	return f32, nil
}

func (g *GGUFModel) dequantize1D(fileData []byte, ti *tensorInfo, nElements int) []float64 {
	offset := int(ti.RawOffset)
	switch ti.Type {
	case 0:
		return dequantizeF32(fileData, offset, nElements)
	case 1:
		return dequantizeF16(fileData, offset, nElements)
	case 2:
		return dequantizeQ4_0Native(fileData, offset, nElements)
	case 3:
		return dequantizeQ4_1Native(fileData, offset, nElements)
	case 6:
		return dequantizeQ5_0Native(fileData, offset, nElements)
	case 7:
		return dequantizeQ5_1Native(fileData, offset, nElements)
	case 8:
		return dequantizeQ8_0Native(fileData, offset, nElements)
	case 10:
		return dequantizeQ2KNative(fileData, offset, nElements)
	case 11:
		return dequantizeQ3KNative(fileData, offset, nElements)
	case 12:
		return dequantizeQ4KNative(fileData, offset, nElements)
	case 13:
		return dequantizeQ5KNative(fileData, offset, nElements)
	case 14:
		return dequantizeQ6KNative(fileData, offset, nElements)
	}
	return make([]float64, nElements)
}

// tensorTypeSupported reports whether a GGUF tensor data type can be decoded.
// Used to fail loudly on unknown quantizations instead of silently loading zeros.
func tensorTypeSupported(t uint32) bool {
	switch t {
	case 0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14:
		return true
	}
	return false
}

func dequantizeQ6KNative(data []byte, offset int, nElements int) []float64 {
	nBlocks := nElements / 256
	result := make([]float64, nElements)
	for i := 0; i < nBlocks; i++ {
		dequantizeQ6KBlock(data, offset+i*210, result, i*256)
	}
	return result
}

func dequantizeQ6KBlock(raw []byte, off int, out []float64, outOff int) {
	d := float16ToFloat64(binary.LittleEndian.Uint16(raw[off+208:]))
	qlOff := off
	qhOff := off + 128
	scalesOff := off + 192

	for n := 0; n < 256; n += 128 {
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := float64(int(int8((raw[qlOff+l]&0xF)|((raw[qhOff+l]>>0)&3)<<4)) - 32)
			q2 := float64(int(int8((raw[qlOff+l+32]&0xF)|((raw[qhOff+l]>>2)&3)<<4)) - 32)
			q3 := float64(int(int8((raw[qlOff+l]>>4)|((raw[qhOff+l]>>4)&3)<<4)) - 32)
			q4 := float64(int(int8((raw[qlOff+l+32]>>4)|((raw[qhOff+l]>>6)&3)<<4)) - 32)
			sc0 := float64(int8(raw[scalesOff+is+0]))
			sc2 := float64(int8(raw[scalesOff+is+2]))
			sc4 := float64(int8(raw[scalesOff+is+4]))
			sc6 := float64(int8(raw[scalesOff+is+6]))
			out[outOff+l+0] = d * sc0 * q1
			out[outOff+l+32] = d * sc2 * q2
			out[outOff+l+64] = d * sc4 * q3
			out[outOff+l+96] = d * sc6 * q4
		}
		outOff += 128
		qlOff += 64
		qhOff += 32
		scalesOff += 8
	}
}

func (g *GGUFModel) loadQuantized2D(fileData []byte, ti *tensorInfo, actualRows, actualCols, nElements int) interface{} {
	offset := int(ti.RawOffset)

	switch ti.Type {
	case 0, 1, 2, 3, 6, 8:
		groupSize := 32
		blockSize := 0
		qType := ""
		switch ti.Type {
		case 0, 1:
			flat := g.dequantize1D(fileData, ti, nElements)
			return reshape2D(flat, actualRows, actualCols)
		case 2:
			blockSize = 18
			qType = "q4"
		case 3:
			blockSize = 20
			qType = "q4_1"
		case 6:
			blockSize = 22
			qType = "q5"
		case 8:
			blockSize = 34
			qType = "q8"
		}

		if blockSize > 0 {
			nGroups := nElements / groupSize
			groupsPerRow := actualCols / groupSize
			totalBytes := nGroups * blockSize
			raw := make([]byte, totalBytes)
			copy(raw, fileData[offset:int(offset)+totalBytes])
			return &QuantWeight{
				QType:  qType,
				Raw:    raw,
				Groups: groupsPerRow,
				Rows:   actualRows,
				Cols:   actualCols,
			}
		}
	}

	flat := g.dequantize1D(fileData, ti, nElements)
	return reshape2D(flat, actualRows, actualCols)
}

func reshape2D(flat []float64, rows, cols int) [][]float64 {
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = flat[i*cols : (i+1)*cols]
	}
	return result
}

func splitTwo(s, sep string) []string {
	for i := 0; i < len(s)-len(sep)+1; i++ {
		if s[i:i+len(sep)] == sep {
			return []string{s[:i], s[i+len(sep):]}
		}
	}
	return []string{s}
}

func quantizeQ8Rows(matrix [][]float64) *QuantWeight {
	if len(matrix) == 0 {
		return nil
	}
	rows := len(matrix)
	cols := len(matrix[0])
	if cols%32 != 0 {
		return nil
	}
	groupsPerRow := cols / 32
	totalGroups := rows * groupsPerRow
	raw := make([]byte, 0, totalGroups*34)

	for r := 0; r < rows; r++ {
		row := matrix[r]
		for g := 0; g < groupsPerRow; g++ {
			base := g * 32
			var maxAbs float64
			for i := 0; i < 32; i++ {
				v := math.Abs(row[base+i])
				if v > maxAbs {
					maxAbs = v
				}
			}
			var scale float64
			if maxAbs > 0 {
				scale = maxAbs / 127.0
			}
			scaleBits := float64ToFloat16(scale)
			scaleBytes := make([]byte, 2)
			binary.LittleEndian.PutUint16(scaleBytes, scaleBits)
			raw = append(raw, scaleBytes...)
			invScale := 1.0 / scale
			for i := 0; i < 32; i++ {
				val := row[base+i] * invScale
				if val > 127 {
					val = 127
				} else if val < -128 {
					val = -128
				}
				// Round to nearest (matching ggml). Truncating toward zero would
				// systematically shrink every weight, biasing the logits.
				raw = append(raw, byte(int8(math.Round(val))))
			}
		}
	}

	return &QuantWeight{
		QType:  "q8",
		Raw:    raw,
		Groups: groupsPerRow,
		Rows:   rows,
		Cols:   cols,
	}
}

func quantizeQ8RowsF32(flat []float32, rows, cols int) *QuantWeight {
	if cols%32 != 0 || len(flat) == 0 {
		return nil
	}
	groupsPerRow := cols / 32
	totalGroups := rows * groupsPerRow
	raw := make([]byte, 0, totalGroups*34)

	for r := 0; r < rows; r++ {
		rowOff := r * cols
		for g := 0; g < groupsPerRow; g++ {
			base := rowOff + g*32
			var maxAbs float32
			for i := 0; i < 32; i++ {
				v := flat[base+i]
				if v < 0 {
					v = -v
				}
				if v > maxAbs {
					maxAbs = v
				}
			}
			var scale float32
			if maxAbs > 0 {
				scale = maxAbs / 127.0
			}
			scaleBits := float64ToFloat16(float64(scale))
			var scaleBytes [2]byte
			binary.LittleEndian.PutUint16(scaleBytes[:], scaleBits)
			raw = append(raw, scaleBytes[:]...)
			invScale := float32(1.0) / scale
			for i := 0; i < 32; i++ {
				val := float64(flat[base+i] * invScale)
				if val > 127 {
					val = 127
				} else if val < -128 {
					val = -128
				}
				raw = append(raw, byte(int8(math.Round(val))))
			}
		}
	}

	return &QuantWeight{
		QType:  "q8",
		Raw:    raw,
		Groups: groupsPerRow,
		Rows:   rows,
		Cols:   cols,
	}
}
