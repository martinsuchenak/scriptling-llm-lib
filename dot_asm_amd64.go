//go:build amd64

package scriptlingllmlib

// q8DotRowsImpl is set at init time (see cpu_amd64.go) to the best
// available implementation for the current CPU.
var q8DotRowsImpl func(rawPtr *byte, xPtr *float32, groups int) float32

//go:noescape
func q8DotRowsAsmSSE(rawPtr *byte, xPtr *float32, groups int) float32

//go:noescape
func q8DotRowsAsmF16C(rawPtr *byte, xPtr *float32, groups int) float32

//go:noescape
func q8DotRowsAsmAVX2(rawPtr *byte, xPtr *float32, groups int) float32

// cpuHasF16C returns true when the CPU reports OSXSAVE + F16C support.
func cpuHasF16C() bool

// cpuHasAVX2 returns true when the CPU reports OSXSAVE + AVX2 support.
// AVX2 implies F16C on all known hardware.
func cpuHasAVX2() bool

// q8DotRowsAsm is the entry point called by float32_ops.go.
// It dispatches to q8DotRowsImpl which is selected at init time.
func q8DotRowsAsm(rawPtr *byte, xPtr *float32, groups int) float32 {
	return q8DotRowsImpl(rawPtr, xPtr, groups)
}
