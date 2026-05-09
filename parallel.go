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

type workerTask struct {
	fn    func(start, end int)
	start int
	end   int
}

type workerPool struct {
	wg     sync.WaitGroup
	workCh chan workerTask
}

var wpool workerPool
var wpoolOnce sync.Once

func initWorkers() {
	if nWorkers <= 1 {
		return
	}
	wpool.workCh = make(chan workerTask, nWorkers*2)
	for i := 0; i < nWorkers; i++ {
		go func() {
			for task := range wpool.workCh {
				task.fn(task.start, task.end)
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

	chunk := (n + nWorkers - 1) / nWorkers
	if chunk < 1 {
		chunk = 1
	}

	nTasks := 0
	for i := 0; i < n; i += chunk {
		end := i + chunk
		if end > n {
			end = n
		}
		nTasks++
	}

	if nTasks == 1 {
		fn(0, n)
		return
	}

	wpool.wg.Add(nTasks)
	for i := 0; i < n; i += chunk {
		end := i + chunk
		if end > n {
			end = n
		}
		wpool.workCh <- workerTask{fn: fn, start: i, end: end}
	}
	wpool.wg.Wait()
}
