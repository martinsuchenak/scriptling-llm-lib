//go:build !darwin && !linux

package scriptlingllmlib

import "fmt"

// mmapFile is unsupported on this platform; callers fall back to os.ReadFile.
func mmapFile(path string) ([]byte, func() error, error) {
	return nil, nil, fmt.Errorf("mmap: unsupported on this platform")
}
