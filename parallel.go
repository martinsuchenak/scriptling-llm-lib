package scriptlingllmlib

import (
	"runtime"
	"sync"
)

var nWorkers int

func init() {
	nWorkers = runtime.NumCPU()
	if nWorkers > 8 {
		nWorkers = 8
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
	wpoolOnce.Do(initWorkers)

	if n <= 256 || nWorkers <= 1 {
		fn(0, n)
		return
	}

	nTasks := nWorkers
	chunk := (n + nTasks - 1) / nTasks

	if chunk < 1 {
		chunk = 1
	}

	actualTasks := 0
	for i := 0; i < n; i += chunk {
		actualTasks++
	}

	if actualTasks == 1 {
		fn(0, n)
		return
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
		wpool.workCh <- func() {
			fn(start, end)
		}
	}
	wpool.wg.Wait()
}
