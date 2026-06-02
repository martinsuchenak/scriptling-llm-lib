//go:build arm64

package scriptlingllmlib

import "golang.org/x/sys/cpu"

//go:noescape
func q41q8RowDotAVX2(wPtr *byte, xqPtr *int8, xScalePtr *float32, sumXqPtr *int32, groups int) float32

func init() {
	if cpu.ARM64.HasASIMDDP {
		q41q8RowFused = q41q8RowDotAVX2
		q41q8FusedAvail = true
	}
}
