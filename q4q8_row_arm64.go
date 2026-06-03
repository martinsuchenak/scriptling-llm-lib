//go:build arm64

package scriptlingllmlib

import "golang.org/x/sys/cpu"

//go:noescape
func q4q8RowDotAVX2(wPtr *byte, xqPtr *int8, xScalePtr *float32, corrPtr *int32, groups int) float32

func init() {
	// SDOT needs ARMv8.2 DotProd. Present on all Apple Silicon; gate for other
	// arm64 hosts so the kernel is never reached without it.
	if cpu.ARM64.HasASIMDDP {
		q4q8RowFused = q4q8RowDotAVX2
		q4q8FusedAvail = true
	}
}
