package scriptlingllmlib

import (
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"
)

var nWorkers int

// parThreshold is the item count below which parallelFor runs serially instead
// of splitting work across the pool.
//
// Fork/join is not free: every parallel round wakes worker goroutines and
// synchronizes on completion. On bare metal (incl. Apple silicon) that costs a
// few hundred nanoseconds and aggressive parallelism (low threshold) wins. But
// on some virtualized hosts goroutine wakeup is expensive enough that splitting
// even a sizable matmul is a net loss; there a high threshold (keeping the small
// per-token decode matmuls serial while still parallelizing the large
// prefill/output projections) is faster. resolveParThreshold measures which
// regime this host is in. Note this is independent of the kernel choice: a host
// can prefer the int8 kernel yet still have cheap fork/join (e.g. a bare-metal
// Ryzen), so the two must not be coupled.
var parThreshold = 256

func init() {
	nWorkers = runtime.NumCPU()
	if nWorkers > 8 {
		nWorkers = 8
	}
	resolveParThreshold()
}

// resolveParThreshold sets parThreshold. SLLM_PARALLEL_THRESHOLD always wins;
// otherwise it is calibrated by measuring fork/join on this host.
func resolveParThreshold() {
	if v := os.Getenv("SLLM_PARALLEL_THRESHOLD"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			parThreshold = n
			return
		}
	}
	parThreshold = calibrateParThreshold()
}

// calibrateParThreshold decides between aggressive parallelism (256) and a
// serial-favoring threshold (8192) by directly measuring whether splitting a
// decode-sized matmul across the pool actually beats running it serially on this
// host. On normal hardware fork/join is cheap and parallel wins big -> 256. On
// hosts where goroutine wakeup is very expensive (some VMs) parallel loses even
// for a sizable matmul -> 8192, so the small per-token matmuls stay serial while
// the large prefill/output projections (well above 8192) still parallelize.
func calibrateParThreshold() int {
	if nWorkers <= 1 {
		return 256
	}

	const groups = 18   // ~typical weight row
	const probeN = 1024 // decode-sized matmul (output rows)
	raw := make([]byte, groups*34)
	for i := range raw {
		raw[i] = byte(i*31 + 7)
	}
	x := make([]float32, groups*32)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.013
	}
	dst := make([]float32, probeN)
	rowFn := func(start, end int) {
		for j := start; j < end; j++ {
			dst[j] = q8DotRowsAsm(&raw[0], &x[0], groups)
		}
	}

	measure := func(parallel bool) time.Duration {
		run := func() {
			if parallel {
				parallelForChunked(probeN, rowFn)
			} else {
				rowFn(0, probeN)
			}
		}
		run() // warm
		best := time.Duration(1<<62 - 1)
		for t := 0; t < 7; t++ {
			start := time.Now()
			for r := 0; r < 16; r++ {
				run()
			}
			if d := time.Since(start); d < best {
				best = d
			}
		}
		return best
	}

	serial := measure(false)
	parallel := measure(true)

	// Only parallelize aggressively if splitting clearly helps (>10% faster).
	if parallel < serial*9/10 {
		return 256
	}
	return 8192
}

type workerPool struct {
	wg     sync.WaitGroup
	workCh chan func()
}

var wpool workerPool
var wpoolOnce sync.Once

func initWorkers() {
	if nWorkers <= 1 {
		return
	}
	wpool.workCh = make(chan func(), nWorkers*4)
	for i := 0; i < nWorkers; i++ {
		go func() {
			for f := range wpool.workCh {
				f()
				wpool.wg.Done()
			}
		}()
	}
}

func parallelFor(n int, fn func(start, end int)) {
	if n <= parThreshold || nWorkers <= 1 {
		fn(0, n)
		return
	}
	parallelForChunked(n, fn)
}

// parallelForChunked splits [0,n) into one chunk per worker and runs them on
// the pool, executing the final chunk inline on the calling goroutine.
func parallelForChunked(n int, fn func(start, end int)) {
	wpoolOnce.Do(initWorkers)
	chunk := (n + nWorkers - 1) / nWorkers
	if chunk < 1 {
		chunk = 1
	}
	for i := 0; i < n; i += chunk {
		end := i + chunk
		if end > n {
			end = n
		}
		start := i
		if i+chunk >= n {
			fn(start, end)
			break
		}
		wpool.wg.Add(1)
		wpool.workCh <- func() { fn(start, end) }
	}
	wpool.wg.Wait()
}
