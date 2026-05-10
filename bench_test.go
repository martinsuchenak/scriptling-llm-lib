package scriptlingllmlib

import (
	"fmt"
	"os"
	"testing"
)

func loadBenchModelF32(b *testing.B) *InferenceModelF32 {
	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		b.Skip("model file not found")
	}
	gguf, err := LoadGGUF(path)
	if err != nil {
		b.Fatalf("LoadGGUF: %v", err)
	}
	gguf.Metadata["_path"] = path
	model, err := buildInferenceModelF32(gguf, path)
	if err != nil {
		b.Fatalf("buildInferenceModelF32: %v", err)
	}
	return model
}

func loadBenchModelF64(b *testing.B) *InferenceModel {
	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		b.Skip("model file not found")
	}
	gguf, err := LoadGGUF(path)
	if err != nil {
		b.Fatalf("LoadGGUF: %v", err)
	}
	gguf.Metadata["_path"] = path
	model, err := buildInferenceModel(gguf, path)
	if err != nil {
		b.Fatalf("buildInferenceModel: %v", err)
	}
	return model
}

func BenchmarkGenerateF32(b *testing.B) {
	model := loadBenchModelF32(b)
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)
	model.initKVCaches()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.initKVCaches()
		_, _, _, _ = model.Generate("Hello", 10, "greedy", 0, 0, 0, 0, 0, "", "", 0)
	}
}

func BenchmarkGenerateF64(b *testing.B) {
	model := loadBenchModelF64(b)
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)
	model.initKVCaches()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.initKVCaches()
		_, _, _, _ = model.Generate("Hello", 10, "greedy", 0, 0, 0, 0, 0, "", "", 0)
	}
}

func BenchmarkDecodeF32(b *testing.B) {
	model := loadBenchModelF32(b)
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)

	nextID := 0
	for i, v := range model.Forward(ids, 0) {
		if v > model.Forward(ids, 0)[i] || i == 0 {
			nextID = i
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Forward([]int{nextID}, len(ids)+i)
	}
}

func BenchmarkDecodeF64(b *testing.B) {
	model := loadBenchModelF64(b)
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)

	nextID := 0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits := model.Forward([]int{nextID}, len(ids)+i)
		maxIdx := 0
		for j, v := range logits {
			if v > logits[maxIdx] {
				maxIdx = j
			}
		}
		nextID = maxIdx
	}
}

func BenchmarkQ8DotGroupXF32(b *testing.B) {
	group := make([]byte, 34)
	for i := range group {
		group[i] = byte(i)
	}
	x := make([]float32, 32)
	for i := range x {
		x[i] = float32(i) * 0.1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q8DotGroupXF32(group, 0, x, 0)
	}
}

func BenchmarkQ8DotGroupX(b *testing.B) {
	group := make([]byte, 34)
	for i := range group {
		group[i] = byte(i)
	}
	x := make([]float64, 32)
	for i := range x {
		x[i] = float64(i) * 0.1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q8DotGroupX(group, 0, x, 0)
	}
}

func BenchmarkQ4DotGroupXF32(b *testing.B) {
	group := make([]byte, 18)
	for i := range group {
		group[i] = byte(i * 17)
	}
	x := make([]float32, 32)
	for i := range x {
		x[i] = float32(i) * 0.1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q4DotGroupXF32(group, 0, x, 0)
	}
}

func BenchmarkQ4DotGroupX(b *testing.B) {
	group := make([]byte, 18)
	for i := range group {
		group[i] = byte(i * 17)
	}
	x := make([]float64, 32)
	for i := range x {
		x[i] = float64(i) * 0.1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q4DotGroupX(group, 0, x, 0)
	}
}

func BenchmarkFloatMatmulF32(b *testing.B) {
	n := 576
	w := make([]float32, n*n)
	x := make([]float32, n)
	for i := range w {
		w[i] = float32(i) * 0.001
	}
	for i := range x {
		x[i] = float32(i) * 0.01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		modelMatmulFloatF32(w, x, 1, n)
	}
}

func loadBenchModelF32ByName(b *testing.B, name string) *InferenceModelF32 {
	if _, err := os.Stat(name); os.IsNotExist(err) {
		b.Skip("model file not found")
	}
	gguf, err := LoadGGUF(name)
	if err != nil {
		b.Fatalf("LoadGGUF: %v", err)
	}
	gguf.Metadata["_path"] = name
	model, err := buildInferenceModelF32(gguf, name)
	if err != nil {
		b.Fatalf("buildInferenceModelF32: %v", err)
	}
	return model
}

func loadBenchModelF64ByName(b *testing.B, name string) *InferenceModel {
	if _, err := os.Stat(name); os.IsNotExist(err) {
		b.Skip("model file not found")
	}
	gguf, err := LoadGGUF(name)
	if err != nil {
		b.Fatalf("LoadGGUF: %v", err)
	}
	gguf.Metadata["_path"] = name
	model, err := buildInferenceModel(gguf, name)
	if err != nil {
		b.Fatalf("buildInferenceModel: %v", err)
	}
	return model
}

func BenchmarkForwardSingleTokenF32_1_7B(b *testing.B) {
	model := loadBenchModelF32ByName(b, "models/SmolLM2-1.7B-Instruct-Q8_0.gguf")
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Forward([]int{0}, len(ids)+i)
	}
}

func BenchmarkForwardSingleTokenF64_1_7B(b *testing.B) {
	model := loadBenchModelF64ByName(b, "models/SmolLM2-1.7B-Instruct-Q8_0.gguf")
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Forward([]int{0}, len(ids)+i)
	}
}

func BenchmarkGenerateF32_1_7B(b *testing.B) {
	model := loadBenchModelF32ByName(b, "models/SmolLM2-1.7B-Instruct-Q8_0.gguf")
	model.initKVCaches()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := *model
		m.KVCaches = make([]KVCacheF32, len(model.KVCaches))
		for j := range m.KVCaches {
			m.KVCaches[j] = KVCacheF32{
				K: make([][]float32, model.nKVHeads),
				V: make([][]float32, model.nKVHeads),
			}
		}
		_, _, _, _ = m.Generate("Hello", 10, "greedy", 0, 0, 0, 0, 0, "", "", 0)
	}
}

func BenchmarkGenerateF64_1_7B(b *testing.B) {
	model := loadBenchModelF64ByName(b, "models/SmolLM2-1.7B-Instruct-Q8_0.gguf")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.initKVCaches()
		_, _, _, _ = model.Generate("Hello", 10, "greedy", 0, 0, 0, 0, 0, "", "", 0)
	}
}

func BenchmarkForwardSingleTokenF32(b *testing.B) {
	model := loadBenchModelF32(b)
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Forward([]int{0}, len(ids)+i)
	}
}

func BenchmarkForwardSingleTokenF64(b *testing.B) {
	model := loadBenchModelF64(b)
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Forward([]int{0}, len(ids)+i)
	}
}

func BenchmarkAllocsF32(b *testing.B) {
	model := loadBenchModelF32(b)
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Forward([]int{0}, len(ids)+i)
	}
}

func BenchmarkAllocsF64(b *testing.B) {
	model := loadBenchModelF64(b)
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Forward([]int{0}, len(ids)+i)
	}
}

func BenchmarkLoadF32_135M(b *testing.B) {
	path := "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		b.Skip("model file not found")
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gguf, err := LoadGGUF(path)
		if err != nil {
			b.Fatal(err)
		}
		_, err = buildInferenceModelF32(gguf, path)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func init() {
	_ = fmt.Sprintf("")
}
