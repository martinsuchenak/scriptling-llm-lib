package scriptlingllmlib

import (
	"os"
	"strings"
	"testing"
)

// TestUnsupportedQuantErrors verifies that a model using a quantization the
// library does not implement (e.g. the IQ-family i-quants, GGUF type 20 used by
// some k-quant repacks for non-256-divisible rows) fails to build with a clear
// error rather than silently loading zeroed weights.
func TestUnsupportedQuantErrors(t *testing.T) {
	// These 135M repacks fall back to IQ4_NL (type 20) for the 576-wide rows.
	for _, path := range []string{
		"models/SmolLM2-135M-Instruct-Q3_K_L.gguf",
		"models/SmolLM2-135M-Instruct-Q2_K.gguf",
	} {
		if _, err := os.Stat(path); err != nil {
			continue
		}
		g, err := LoadGGUF(path)
		if err != nil {
			t.Fatalf("LoadGGUF(%s): %v", path, err)
		}
		g.Metadata["_path"] = path
		_, err = buildInferenceModelF32(g, path)
		g.ReleaseFileData()
		if err == nil {
			t.Errorf("%s: expected build to fail on unsupported quant type, got nil", path)
			continue
		}
		if !strings.Contains(err.Error(), "unsupported type") {
			t.Errorf("%s: error = %q, want it to mention 'unsupported type'", path, err)
		}
	}
}
