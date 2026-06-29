package scriptlingllmlib

import (
	"math"
	"os"
	"runtime"
	"testing"
)

// kquantKernelParity loads a real tensor of the given GGUF type, then checks that
// the packed dot kernel produces the same row dot products as the verified dense
// dequantizer (kquants.go) for several rows against a fixed activation vector.
func kquantKernelParity(t *testing.T, model string, gtype uint32,
	dot func(raw []byte, rOff int, x []float32, xOff, nSB int) float32, blockBytes int) {

	if _, err := os.Stat(model); err != nil {
		t.Skip("model not present")
	}
	g, err := LoadGGUF(model)
	if err != nil {
		t.Fatal(err)
	}
	defer g.ReleaseFileData()

	var name string
	var ti *tensorInfo
	for n, info := range g.Tensors {
		if info.Type == gtype && len(info.Dims) == 2 {
			name, ti = n, info
			break
		}
	}
	if ti == nil {
		t.Skipf("%s: no 2D tensor of type %d", model, gtype)
	}
	rows, cols := int(ti.Dims[1]), int(ti.Dims[0])
	nElem := rows * cols
	dense := g.dequantize1D(g.fileData, ti, nElem) // []float64, verified path
	raw := g.fileData[ti.RawOffset:]               // packed bytes
	nSB := cols / 256

	// Fixed pseudo-random activation.
	x := make([]float32, cols)
	seed := uint32(12345)
	for i := range x {
		seed = seed*1664525 + 1013904223
		x[i] = float32(int32(seed))/float32(1<<31) - 0.5
	}

	var worst float64
	for _, j := range []int{0, 1, rows / 2, rows - 1} {
		var want float64
		for i := 0; i < cols; i++ {
			want += dense[j*cols+i] * float64(x[i])
		}
		got := float64(dot(raw, j*nSB*blockBytes, x, 0, nSB))
		rel := math.Abs(got-want) / (math.Abs(want) + 1e-6)
		if rel > worst {
			worst = rel
		}
	}
	runtime.KeepAlive(g)
	t.Logf("%s type=%d: worst row-dot relErr vs dense decoder = %.2e", name, gtype, worst)
	if worst > 1e-4 {
		t.Errorf("type %d kernel diverges from dense decoder: relErr %.3e", gtype, worst)
	}
}

func TestQ4KKernelParity(t *testing.T) {
	kquantKernelParity(t, "models/SmolLM2-135M-Instruct-Q4_K_M.gguf", 12, q4kDotRowF32, q4kBlockBytes)
}
func TestQ5KKernelParity(t *testing.T) {
	kquantKernelParity(t, "models/SmolLM2-135M-Instruct-Q5_K_M.gguf", 13, q5kDotRowF32, q5kBlockBytes)
}
func TestQ6KKernelParity(t *testing.T) {
	kquantKernelParity(t, "models/SmolLM2-135M-Instruct-Q6_K.gguf", 14, q6kDotRowF32, q6kBlockBytes)
}

// TestQ4KToQ41Parity checks the Q4_K->Q4_1 conversion reproduces the Q4_K
// decoder's values (within f16 scale-rounding), so the fast q41q8 kernel sees
// the right weights.
func TestQ4KToQ41Parity(t *testing.T) {
	const model = "models/SmolLM2-135M-Instruct-Q4_K_M.gguf"
	if _, err := os.Stat(model); err != nil {
		t.Skip("model not present")
	}
	g, err := LoadGGUF(model)
	if err != nil {
		t.Fatal(err)
	}
	defer g.ReleaseFileData()
	var ti *tensorInfo
	for _, info := range g.Tensors {
		if info.Type == 12 && len(info.Dims) == 2 {
			ti = info
			break
		}
	}
	if ti == nil {
		t.Skip("no Q4_K tensor")
	}
	rows, cols := int(ti.Dims[1]), int(ti.Dims[0])
	n := rows * cols
	dense := g.dequantize1D(g.fileData, ti, n)
	fromQ41 := dequantizeQ4_1Native(convertQ4KToQ41(g.fileData, int(ti.RawOffset), rows, cols), 0, n)
	runtime.KeepAlive(g)
	var sad, smag float64
	for i := range dense {
		sad += math.Abs(dense[i] - fromQ41[i])
		smag += math.Abs(dense[i])
	}
	if rel := sad / smag; rel > 5e-3 {
		t.Errorf("Q4_K->Q4_1 conversion relErr %.3e too high", rel)
	}
}

// TestQ5KToTwoQ41Parity checks Q5_K == low Q4_1 + high Q4_1 (the two halves the
// q41q8 kernel sums), within f16 scale-rounding.
func TestQ5KToTwoQ41Parity(t *testing.T) {
	const model = "models/SmolLM2-135M-Instruct-Q5_K_M.gguf"
	if _, err := os.Stat(model); err != nil {
		t.Skip("model not present")
	}
	g, err := LoadGGUF(model)
	if err != nil {
		t.Fatal(err)
	}
	defer g.ReleaseFileData()
	var ti *tensorInfo
	for _, info := range g.Tensors {
		if info.Type == 13 && len(info.Dims) == 2 {
			ti = info
			break
		}
	}
	if ti == nil {
		t.Skip("no Q5_K tensor")
	}
	rows, cols := int(ti.Dims[1]), int(ti.Dims[0])
	n := rows * cols
	dense := g.dequantize1D(g.fileData, ti, n)
	conv := convertQ5KToTwoQ41(g.fileData, int(ti.RawOffset), rows, cols)
	low := dequantizeQ4_1Native(conv, 0, n)
	high := dequantizeQ4_1Native(conv, rows*(cols/32)*20, n)
	runtime.KeepAlive(g)
	var sad, smag float64
	for i := range dense {
		sad += math.Abs(dense[i] - (low[i] + high[i]))
		smag += math.Abs(dense[i])
	}
	if rel := sad / smag; rel > 5e-3 {
		t.Errorf("Q5_K->2xQ4_1 relErr %.3e too high", rel)
	}
}

// TestKQuantPackedEndToEnd exercises the full packed path (build + dispatch) and
// confirms perplexity matches the dense-float path within float-rounding noise.
func TestKQuantPackedEndToEnd(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping perplexity end-to-end under race detector")
	}
	const model = "models/SmolLM2-135M-Instruct-Q4_K_M.gguf"
	if _, err := os.Stat(model); err != nil {
		t.Skip("model not present")
	}
	mDense := loadModelForTest(t, model)
	ids := mDense.Tokenizer.Encode(pplText)
	dense := perplexity(mDense, ids)

	defer func() { kquantPacked = false }()
	kquantPacked = true
	mPacked := loadModelForTest(t, model)
	packed := perplexity(mPacked, ids)

	t.Logf("Q4_K_M perplexity: dense=%.4f packed=%.4f", dense, packed)
	if d := math.Abs(packed - dense); d/dense > 0.02 {
		t.Errorf("packed perplexity %.4f differs from dense %.4f by >2%%", packed, dense)
	}
}
