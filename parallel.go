package scriptlingllmlib

import (
	"os"
	"runtime"
	"strconv"
	"sync"
)

var nWorkers int

// parThreshold is the item count below which parallelFor runs serially instead
// of splitting work across the pool.
//
// Fork/join is not free: every parallel round wakes worker goroutines and
// synchronizes on completion. On bare metal (incl. Apple silicon) that costs a
// few hundred nanoseconds and the default 256 is a good crossover. But on some
// virtualized hosts goroutine wakeup is expensive enough that splitting the
// small per-token decode matmuls is a net loss — there a much higher threshold
// (keeping those serial while still parallelizing the large prefill/output
// projections) is faster. See resolveParThreshold.
var parThreshold = 256

func init() {
	nWorkers = runtime.NumCPU()
	if nWorkers > 8 {
		nWorkers = 8
	}
}

// resolveParThreshold sets parThreshold. SLLM_PARALLEL_THRESHOLD always wins;
// otherwise, on hosts where SIMD/fork-join is penalized (penalizedHost, the
// same signal that selects the int8 kernel) it raises the threshold so small
// matmuls stay serial. Bare metal and Apple-silicon keep the default 256, so
// their behavior is unchanged. Called from init once the host has been probed.
func resolveParThreshold(penalizedHost bool) {
	if v := os.Getenv("SLLM_PARALLEL_THRESHOLD"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			parThreshold = n
			return
		}
	}
	if penalizedHost {
		parThreshold = 8192
	}
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
