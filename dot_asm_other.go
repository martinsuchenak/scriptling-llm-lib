//go:build !arm64 && !amd64

package scriptlingllmlib

func q8DotRowsAsm(rawPtr *byte, xPtr *float32, groups int) float32 {
	return 0
}
