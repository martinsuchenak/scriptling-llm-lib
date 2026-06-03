package scriptlingllmlib

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestMmapFileRoundTrip(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("mmap only implemented on linux/darwin")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "blob")
	want := []byte("hello mmap world\x00\x01\x02")
	if err := os.WriteFile(path, want, 0o644); err != nil {
		t.Fatal(err)
	}

	data, closer, err := mmapFile(path)
	if err != nil {
		t.Fatalf("mmapFile: %v", err)
	}
	if string(data) != string(want) {
		t.Errorf("mapped bytes = %q, want %q", data, want)
	}
	if err := closer(); err != nil {
		t.Errorf("unmap: %v", err)
	}

	// Empty files cannot be mapped; the caller must fall back to a read.
	empty := filepath.Join(dir, "empty")
	if err := os.WriteFile(empty, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := mmapFile(empty); err == nil {
		t.Error("expected error mapping empty file")
	}
}

// TestGGUFReleaseIdempotentAndLazyReload: ReleaseFileData (which unmaps when
// mapped) must be safe to call repeatedly, and tensors must still be loadable
// afterwards via the on-demand re-read path.
func TestGGUFReleaseIdempotentAndLazyReload(t *testing.T) {
	const path = "models/SmolLM2-135M-Instruct-Q8_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skip("model not present")
	}
	g, err := LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	g.Metadata["_path"] = path
	if len(g.Tensors) == 0 {
		t.Fatal("no tensors parsed")
	}

	g.ReleaseFileData()
	g.ReleaseFileData() // must not panic or double-unmap

	// fileData is gone; LoadTensor must transparently re-read the file.
	if _, err := g.LoadTensor("token_embedding.weight"); err != nil {
		t.Errorf("LoadTensor after release: %v", err)
	}
}
