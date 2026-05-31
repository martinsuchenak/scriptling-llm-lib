//go:build amd64

package scriptlingllmlib

import "time"

func init() {
	var simd func(rawPtr *byte, xPtr *float32, groups int) float32
	switch {
	case cpuHasAVX2():
		simd = q8DotRowsAsmAVX2
	case cpuHasF16C():
		simd = q8DotRowsAsmF16C
	default:
		simd = q8DotRowsAsmSSE
	}

	// The SIMD kernels are normally faster than scalar, but on some virtualized
	// CPUs AVX2/F16C instructions are heavily penalized and the scalar path wins
	// by a wide margin. Pick whichever is actually faster on this machine.
	if pickFasterQ8Kernel(simd, q8DotRowsScalar) {
		q8DotRowsImpl = simd
	} else {
		q8DotRowsImpl = q8DotRowsScalar
	}
}

// pickFasterQ8Kernel micro-benchmarks the two kernels on a representative row
// and returns true when `simd` is at least as fast as `scalar`. Kept cheap
// (a few hundred microseconds) so it adds negligible startup cost.
func pickFasterQ8Kernel(simd, scalar func(rawPtr *byte, xPtr *float32, groups int) float32) bool {
	const groups = 64 // ~2KB row, representative of real weight rows
	raw := make([]byte, groups*34)
	for i := range raw {
		raw[i] = byte(i*31 + 7)
	}
	x := make([]float32, groups*32)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.013
	}

	bench := func(fn func(rawPtr *byte, xPtr *float32, groups int) float32) time.Duration {
		// Warm up to fault in pages and stabilize.
		for i := 0; i < 200; i++ {
			kernelBenchSink += fn(&raw[0], &x[0], groups)
		}
		best := time.Duration(1<<62 - 1)
		for trial := 0; trial < 5; trial++ {
			start := time.Now()
			for i := 0; i < 2000; i++ {
				kernelBenchSink += fn(&raw[0], &x[0], groups)
			}
			if d := time.Since(start); d < best {
				best = d
			}
		}
		return best
	}

	return bench(simd) <= bench(scalar)
}
