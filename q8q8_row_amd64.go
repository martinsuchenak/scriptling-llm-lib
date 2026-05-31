//go:build amd64

package scriptlingllmlib

//go:noescape
func q8q8RowDotAVX2(wPtr *byte, xqPtr *int8, scalePtr *float32, groups int) float32

func init() {
	// The fused row kernel needs AVX2 (same requirement as int8DotAVX2).
	if cpuHasAVX2() {
		q8q8RowFused = q8q8RowDotAVX2
		q8q8FusedAvail = true
	}
}
