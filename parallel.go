package scriptlingllmlib

import (
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
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
	// Use all cores up to 8; on bigger machines use NumCPU-2, leaving a couple
	// of cores spare so idle workers can spin without contending with the main
	// goroutine (the spare-core fast path below). Capped so the fork/join cost
	// stays bounded. 8-core and smaller hosts (incl. the constrained VM) are
	// unchanged.
	n := runtime.NumCPU()
	switch {
	case n <= 8:
		nWorkers = n
	case n <= 18:
		nWorkers = n - 2
	default:
		nWorkers = 16
	}
	// Idle workers spin before parking. Decode fires many parallel rounds per
	// token separated by serial gaps (rms-norm, attention); spinning across
	// those gaps avoids the thread park/wake cost, which is very expensive on
	// macOS. Spin long ONLY when there are spare cores (NumCPU > workers, e.g.
	// the 14-core Max chips) so the idle spin never contends with the main
	// goroutine; otherwise park quickly. This is set before calibration so the
	// threshold measurement sees the real park/spin behavior.
	spareCores = runtime.NumCPU() > nWorkers
	if spareCores {
		parkAfter = 50 * time.Millisecond
		// The pool uses at most nWorkers+1 goroutines, but Go runs GOMAXPROCS
		// (=NumCPU) Ps. The extra idle Ps' threads churn in the work-stealing
		// scheduler (findRunnable/usleep/lock2) — cheap on Linux, brutal on
		// macOS, where it dominated 1.7B decode. Cap GOMAXPROCS to what the pool
		// actually uses so those idle Ps go away. Compute is unaffected (workers
		// are already capped at nWorkers). Respect an explicit user setting and
		// allow opt-out; never raise GOMAXPROCS.
		if os.Getenv("GOMAXPROCS") == "" && os.Getenv("SLLM_NO_GOMAXPROCS") == "" {
			if want := nWorkers + 1; runtime.GOMAXPROCS(0) > want {
				runtime.GOMAXPROCS(want)
			}
		}
	}
	resolveParThreshold()
}

// spareCores is true when the host has more CPUs than pool workers, so idle
// spinning never contends with the main goroutine — letting both the worker
// idle-wait and the completion-wait busy-spin instead of touching the (very
// expensive on macOS) scheduler.
var spareCores bool

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

	// Each round is a serial "gap" (rowFn(0,probeN), ~a real inter-matmul gap of
	// a few hundred µs) followed by the matmul under test. The gap makes idle
	// workers park-or-keep-spinning exactly as they do between a token's matmuls,
	// so the parallel timing includes the host's real wake cost (cheap on Linux
	// futex / a hot spin pool, expensive on a constrained VM that parks). Tight
	// warm loops hid this and mis-picked the threshold.
	measure := func(parallel bool) time.Duration {
		run := func() {
			rowFn(0, probeN) // serial gap
			if parallel {
				parallelForChunked(probeN, rowFn)
			} else {
				rowFn(0, probeN)
			}
		}
		run()
		best := time.Duration(1<<62 - 1)
		for t := 0; t < 5; t++ {
			start := time.Now()
			for r := 0; r < 20; r++ {
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

	// Parallelize aggressively only if splitting clearly helps under the realistic
	// gap (>10% faster); otherwise keep small matmuls serial.
	if parallel < serial*9/10 {
		return 256
	}
	return 8192
}

// workerPool is a spin-then-park fork/join pool. Workers grab chunks of the
// current batch via an atomic cursor; between batches they spin briefly before
// parking on a condition variable. Decode fires ~hundreds of tiny parallel
// rounds per token, so the spin keeps back-to-back rounds from paying the
// thread park/wake cost — which is cheap on Linux (futex) but very expensive on
// macOS (pthread_cond), where the old channel pool spent ~75% of its time.
type workerPool struct {
	fn        func(start, end int) // current batch body
	n         int                  // current batch size
	chunk     int                  // rows per chunk
	numChunks int64                // total chunks this batch
	nextChunk int64                // atomic cursor into chunks
	remaining int64                // atomic chunks not yet completed
	gen       uint64               // atomic batch generation
	mu        sync.Mutex
	cond      *sync.Cond
}

var wpool workerPool
var wpoolOnce sync.Once

// parkAfter is how long an idle worker keeps spinning (checking the generation
// counter) before parking on the cond. Set in init: long on spare-core hosts so
// workers stay hot across a token's serial gaps (attention/rms-norm) and never
// pay the macOS thread park/wake cost during a generation; short elsewhere.
var parkAfter = 50 * time.Microsecond

func initWorkers() {
	if nWorkers <= 1 {
		return
	}
	wpool.cond = sync.NewCond(&wpool.mu)
	for i := 1; i < nWorkers; i++ { // main participates as the nWorkers-th
		go workerLoop()
	}
}

func workerLoop() {
	lastGen := atomic.LoadUint64(&wpool.gen)
	for {
		if atomic.LoadUint64(&wpool.gen) == lastGen {
			// Spin for the next batch, parking only after parkAfter of idleness.
			// The inner loop checks the generation cheaply; the time check (and
			// thus park decision) happens only every ~few microseconds.
			idleStart := time.Now()
			parked := false
		spin:
			for atomic.LoadUint64(&wpool.gen) == lastGen {
				for i := 0; i < 2048; i++ {
					if atomic.LoadUint64(&wpool.gen) != lastGen {
						break spin
					}
				}
				if time.Since(idleStart) >= parkAfter {
					parked = true
					break
				}
			}
			if parked {
				wpool.mu.Lock()
				for atomic.LoadUint64(&wpool.gen) == lastGen {
					wpool.cond.Wait()
				}
				wpool.mu.Unlock()
			}
		}
		lastGen = atomic.LoadUint64(&wpool.gen)
		runChunks()
	}
}

// runChunks claims and runs chunks of the current batch until none remain.
func runChunks() {
	nc := atomic.LoadInt64(&wpool.numChunks)
	for {
		c := atomic.AddInt64(&wpool.nextChunk, 1) - 1
		if c >= nc {
			return
		}
		start := int(c) * wpool.chunk
		end := start + wpool.chunk
		if end > wpool.n {
			end = wpool.n
		}
		wpool.fn(start, end)
		atomic.AddInt64(&wpool.remaining, -1)
	}
}

func parallelFor(n int, fn func(start, end int)) {
	if n <= parThreshold || nWorkers <= 1 {
		fn(0, n)
		return
	}
	parallelForChunked(n, fn)
}

// parallelForChunked splits [0,n) into one chunk per worker and runs the batch
// on the pool, with the calling goroutine participating, blocking until done.
func parallelForChunked(n int, fn func(start, end int)) {
	wpoolOnce.Do(initWorkers)
	if nWorkers <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + nWorkers - 1) / nWorkers
	if chunk < 1 {
		chunk = 1
	}
	numChunks := int64((n + chunk - 1) / chunk)

	wpool.fn = fn
	wpool.n = n
	wpool.chunk = chunk
	atomic.StoreInt64(&wpool.numChunks, numChunks)
	atomic.StoreInt64(&wpool.nextChunk, 0)
	atomic.StoreInt64(&wpool.remaining, numChunks)

	// Publish the batch (release) and wake any parked workers. Spinning workers
	// observe the generation bump without taking the lock.
	wpool.mu.Lock()
	atomic.AddUint64(&wpool.gen, 1)
	wpool.cond.Broadcast()
	wpool.mu.Unlock()

	runChunks() // the caller is a worker too

	// Wait for stragglers. With spare cores, pure busy-spin — touching the
	// scheduler (Gosched -> usleep) here is itself a big cost on macOS. Without
	// spare cores, yield periodically so a descheduled worker can still run.
	if spareCores {
		for atomic.LoadInt64(&wpool.remaining) > 0 {
		}
	} else {
		for spins := 0; atomic.LoadInt64(&wpool.remaining) > 0; spins++ {
			if spins&1023 == 1023 {
				runtime.Gosched()
			}
		}
	}
}
