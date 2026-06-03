package scriptlingllmlib

import (
	"testing"
)

// TestUnsupportedQuantErrors verifies the support gate: types the library can
// decode are accepted, and others (the remaining IQ-family i-quants) are flagged
// so loadWeight* fails loudly instead of loading zeroed/garbage weights.
func TestUnsupportedQuantErrors(t *testing.T) {
	supported := []uint32{
		0, 1, // F32, F16
		2, 3, 6, 7, 8, // Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
		10, 11, 12, 13, 14, // Q2_K..Q6_K
		20, // IQ4_NL
	}
	for _, ty := range supported {
		if !tensorTypeSupported(ty) {
			t.Errorf("type %d should be supported", ty)
		}
	}
	// Still-unimplemented i-quants must be rejected.
	unsupported := []uint32{
		16, 17, 18, 19, // IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S
		21, 22, 23, // IQ3_S, IQ2_S, IQ4_XS
		99, // nonexistent
	}
	for _, ty := range unsupported {
		if tensorTypeSupported(ty) {
			t.Errorf("type %d should NOT be supported", ty)
		}
	}
}
