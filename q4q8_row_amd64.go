//go:build amd64

package scriptlingllmlib

//go:noescape
func q4q8RowDotAVX2(wPtr *byte, xqPtr *int8, scalePtr *float32, groups int) float32

func init() {
	if cpuHasAVX2() {
		q4q8RowFused = q4q8RowDotAVX2
		q4q8FusedAvail = true
	}
}
