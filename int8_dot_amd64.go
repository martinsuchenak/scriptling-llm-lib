//go:build amd64

package scriptlingllmlib

//go:noescape
func int8DotAVX2(aPtr, bPtr *int8, n int) int32

func init() {
	// The AVX2 kernel needs AVX2; without it, fall back to the scalar int8 dot.
	// (Whether the int8 path is used at all is decided separately by the
	// kernel selector in q8q8.go, which benchmarks it against the float path.)
	if cpuHasAVX2() {
		int8Dot = int8DotAVX2
	}
}
