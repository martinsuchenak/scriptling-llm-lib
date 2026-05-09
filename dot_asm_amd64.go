//go:build amd64

package scriptlingllmlib

//go:noescape
func q8DotRowsAsm(rawPtr *byte, xPtr *float32, groups int) float32
