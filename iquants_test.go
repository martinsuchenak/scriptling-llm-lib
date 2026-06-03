package scriptlingllmlib

import (
	"encoding/binary"
	"math"
	"os"
	"runtime"
	"testing"
)

// TestIQ4NLDequant validates the IQ4_NL codebook and the Q4_0-style split nibble
// layout directly against a hand-built block.
func TestIQ4NLDequant(t *testing.T) {
	buf := make([]byte, 18)
	binary.LittleEndian.PutUint16(buf, float64ToFloat16(2.0)) // d = 2.0 (exact in f16)
	for j := 0; j < 16; j++ {
		buf[2+j] = byte(j) | (byte(15-j) << 4) // low nibble j, high nibble 15-j
	}
	out := dequantizeIQ4NLNative(buf, 0, 32)
	for j := 0; j < 16; j++ {
		wantLo := 2.0 * float64(iq4nlValues[j])
		wantHi := 2.0 * float64(iq4nlValues[15-j])
		if math.Abs(out[j]-wantLo) > 1e-3 {
			t.Errorf("elem %d = %v, want %v", j, out[j], wantLo)
		}
		if math.Abs(out[16+j]-wantHi) > 1e-3 {
			t.Errorf("elem %d = %v, want %v", 16+j, out[16+j], wantHi)
		}
	}
}

// TestIQ4NLModelsRun checks that models using IQ4_NL (the SmolLM2-135M Q2_K /
// Q3_K_L repacks are mostly IQ4_NL for their non-256-divisible rows) load and
// produce finite perplexity instead of failing with "unsupported type 20".
func TestIQ4NLModelsRun(t *testing.T) {
	for _, p := range []string{
		"models/SmolLM2-135M-Instruct-Q2_K.gguf",
		"models/SmolLM2-135M-Instruct-Q3_K_L.gguf",
	} {
		if _, err := os.Stat(p); err != nil {
			continue
		}
		m := loadModelForTest(t, p)
		ppl := perplexity(m, m.Tokenizer.Encode(pplText))
		runtime.KeepAlive(m)
		t.Logf("%s: ppl=%.4f", p, ppl)
		if math.IsNaN(ppl) || math.IsInf(ppl, 0) || ppl <= 1 || ppl > 100 {
			t.Errorf("%s: ppl=%v, want finite and sane (<100)", p, ppl)
		}
	}
}
