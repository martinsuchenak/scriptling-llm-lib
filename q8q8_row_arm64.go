//go:build arm64

package scriptlingllmlib

import "golang.org/x/sys/cpu"

//go:noescape
func q8q8RowDotAVX2(wPtr *byte, xqPtr *int8, scalePtr *float32, groups int) float32

func init() {
	if cpu.ARM64.HasASIMDDP {
		q8q8RowFused = q8q8RowDotAVX2
		q8q8FusedAvail = true
	}
}
