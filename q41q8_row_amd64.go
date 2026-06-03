//go:build amd64

package scriptlingllmlib

//go:noescape
func q41q8RowDotAVX2(wPtr *byte, xqPtr *int8, xScalePtr *float32, sumXqPtr *int32, groups int) float32

func init() {
	if cpuHasAVX2() {
		q41q8RowFused = q41q8RowDotAVX2
		q41q8FusedAvail = true
	}
}
