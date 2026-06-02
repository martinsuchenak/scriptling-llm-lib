package scriptlingllmlib

import "testing"

// FuzzParseGGUF asserts the GGUF parser never panics on untrusted input: any byte
// string must yield either a parsed model or an error. The seed corpus runs as a
// normal unit test under `go test`; run `go test -fuzz=FuzzParseGGUF` to explore.
//
// Seeds are deliberately tiny structural inputs — seeding with a real multi-MB
// model would exceed the fuzzing engine's shared-memory transport once mutated,
// and small inputs reach the interesting header/length-handling paths anyway.
func FuzzParseGGUF(f *testing.F) {
	f.Add([]byte(nil))
	f.Add([]byte("GGUF"))
	f.Add([]byte("NOT_A_GGUF_FILE"))
	// Valid magic + version 3 + zero tensor/metadata counts.
	f.Add([]byte{0x47, 0x47, 0x55, 0x46, 3, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0})
	// Valid magic but absurd counts (must be rejected, not OOM/panic).
	f.Add([]byte{0x47, 0x47, 0x55, 0x46, 3, 0, 0, 0,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff})
	// Magic + version + one metadata entry whose string length is absurd.
	f.Add([]byte{0x47, 0x47, 0x55, 0x46, 3, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff})

	f.Fuzz(func(t *testing.T, data []byte) {
		// The only requirement is that this does not panic.
		_, _ = parseGGUF(data)
	})
}
