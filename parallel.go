package scriptlingllmlib

import (
	"runtime"
	"sync"
)

var nWorkers int

func init() {
	nWorkers = runtime.NumCPU()
	if nWorkers > 4 {
		nWorkers = 4
	}
}

func parallelFor(n int, fn func(start, end int)) {
	if n <= 256 || nWorkers <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + nWorkers - 1) / nWorkers
	if chunk < 1 {
		chunk = 1
	}
	var wg sync.WaitGroup
	for i := 0; i < n; i += chunk {
		end := i + chunk
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(start, end int) {
			fn(start, end)
			wg.Done()
		}(i, end)
	}
	wg.Wait()
}
