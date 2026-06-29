package scriptlingllmlib

import (
	"os"
	"sync/atomic"
	"testing"
)

func TestMain(m *testing.M) {
	code := m.Run()
	shutdownWorkers()
	os.Exit(code)
}

func TestParallelFor(t *testing.T) {
	n := 100
	var count atomic.Int64
	parallelFor(n, func(start, end int) {
		for i := start; i < end; i++ {
			count.Add(1)
		}
	})
	if count.Load() != int64(n) {
		t.Errorf("parallelFor executed %d times, want %d", count.Load(), n)
	}
}

func TestParallelForResult(t *testing.T) {
	results := make([]int, 100)
	parallelFor(100, func(start, end int) {
		for i := start; i < end; i++ {
			results[i] = i * 2
		}
	})
	for i, v := range results {
		if v != i*2 {
			t.Errorf("results[%d] = %d, want %d", i, v, i*2)
			break
		}
	}
}

func TestParallelForZero(t *testing.T) {
	var iterations atomic.Int64
	parallelFor(0, func(start, end int) {
		for i := start; i < end; i++ {
			iterations.Add(1)
		}
	})
	if iterations.Load() != 0 {
		t.Errorf("should not iterate for n=0, got %d iterations", iterations.Load())
	}
}

func TestParallelForSmall(t *testing.T) {
	var sum atomic.Int64
	parallelFor(5, func(start, end int) {
		for i := start; i < end; i++ {
			sum.Add(int64(i))
		}
	})
	if sum.Load() != 10 {
		t.Errorf("parallelFor(5) sum = %d, want 10", sum.Load())
	}
}
