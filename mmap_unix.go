//go:build darwin || linux

package scriptlingllmlib

import (
	"fmt"
	"os"
	"syscall"
)

// mmapFile memory-maps path read-only and returns the mapped bytes plus a closer
// that unmaps them. Mapping the file (instead of os.ReadFile) avoids a full-size
// heap copy during model loading: the pages are clean and file-backed, so they
// don't count as dirty anonymous memory and can be reclaimed under pressure or
// shared across processes loading the same model. The fd may be closed once the
// mapping exists; the mapping stays valid until the closer runs.
func mmapFile(path string) ([]byte, func() error, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = f.Close() }()

	fi, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	size := fi.Size()
	if size <= 0 {
		return nil, nil, fmt.Errorf("mmap: empty file")
	}
	if int64(int(size)) != size {
		return nil, nil, fmt.Errorf("mmap: file too large to map (%d bytes)", size)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, fmt.Errorf("mmap: %w", err)
	}
	return data, func() error { return syscall.Munmap(data) }, nil
}
