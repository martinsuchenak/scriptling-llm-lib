package scriptlingllmlib

import (
	"runtime"
	"sync"
)

var nWorkers = runtime.NumCPU()

func parallelFor(n int, fn func(start, end int)) {
	if n <= 64 || nWorkers <= 1 {
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
