//go:build amd64

package scriptlingllmlib

func init() {
	switch {
	case cpuHasAVX2():
		q8DotRowsImpl = q8DotRowsAsmAVX2
	case cpuHasF16C():
		q8DotRowsImpl = q8DotRowsAsmF16C
	default:
		q8DotRowsImpl = q8DotRowsAsmSSE
	}
}
