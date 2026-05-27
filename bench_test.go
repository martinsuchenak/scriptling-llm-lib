package scriptlingllmlib

import (
	"os"
	"testing"
)

func loadBenchModel(b *testing.B, path string) *InferenceModelF32 {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		b.Skip("model file not found")
	}
	gguf, err := LoadGGUF(path)
	if err != nil {
		b.Fatalf("LoadGGUF: %v", err)
	}
	gguf.Metadata["_path"] = path
	model, err := buildInferenceModelF32(gguf, path)
	gguf.ReleaseFileData()
	if err != nil {
		b.Fatalf("buildInferenceModelF32: %v", err)
	}
	return model
}

func BenchmarkGenerate135M(b *testing.B) {
	model := loadBenchModel(b, "models/SmolLM2-135M-Instruct-Q8_0.gguf")
	model.initKVCaches()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.initKVCaches()
		_, _, _, _ = model.Generate("Hello", 10, "greedy", 0, 0, 0, 0, 0, "", "", 0)
	}
}

func BenchmarkGenerate1_7B(b *testing.B) {
	model := loadBenchModel(b, "models/SmolLM2-1.7B-Instruct-Q8_0.gguf")
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

func BenchmarkDecodeToken135M(b *testing.B) {
	model := loadBenchModel(b, "models/SmolLM2-135M-Instruct-Q8_0.gguf")
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Forward([]int{0}, len(ids)+i)
	}
}

func BenchmarkDecodeToken1_7B(b *testing.B) {
	model := loadBenchModel(b, "models/SmolLM2-1.7B-Instruct-Q8_0.gguf")
	ids := model.Tokenizer.Encode("Hello")
	model.initKVCaches()
	model.Forward(ids, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Forward([]int{0}, len(ids)+i)
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

func BenchmarkQ8DotRowsF32(b *testing.B) {
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
		q8DotRowsF32(group, 0, x, 0, 1)
	}
}

func BenchmarkQ8DotRowsF32_128groups(b *testing.B) {
	const groups = 128
	raw := make([]byte, groups*34)
	for i := range raw {
		raw[i] = byte(i)
	}
	x := make([]float32, groups*32)
	for i := range x {
		x[i] = float32(i) * 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q8DotRowsF32(raw, 0, x, 0, groups)
	}
}

func BenchmarkQ8DotGroupXF32_128groups(b *testing.B) {
	const groups = 128
	raw := make([]byte, groups*34)
	for i := range raw {
		raw[i] = byte(i)
	}
	x := make([]float32, groups*32)
	for i := range x {
		x[i] = float32(i) * 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sum float32
		for g := 0; g < groups; g++ {
			sum += q8DotGroupXF32(raw, g*34, x, g*32)
		}
		_ = sum
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

// BenchmarkParallelMatmulQ8 simulates a SmolLM2-135M FFN projection:
// 576 in_features → 1536 out_features (Q8_0, 18 groups/row).
// outFeatures=1536 >> parallelFor threshold=256, so this exercises the worker pool.
func BenchmarkParallelMatmulQ8(b *testing.B) {
	const inFeatures = 576
	const outFeatures = 1536
	flat := make([]float32, outFeatures*inFeatures)
	for i := range flat {
		flat[i] = float32(i%127) * 0.01
	}
	w := quantizeQ8RowsF32(flat, outFeatures, inFeatures)
	x := make([]float32, inFeatures)
	for i := range x {
		x[i] = float32(i) * 0.001
	}
	dst := make([]float32, outFeatures)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		modelMatmulRowQuantIntoF32(w, x, dst)
	}
}

func BenchmarkLoad135M(b *testing.B) {
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
