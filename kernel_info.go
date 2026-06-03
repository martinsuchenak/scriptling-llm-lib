package scriptlingllmlib

import (
	"fmt"
	"runtime"
)

// KernelInfo reports which compute kernels the library selected at init for this
// host (after the auto-calibration in q8q8.go / the *_amd64.go detectors). It is a
// diagnostic aid for performance work — e.g. confirming whether the int8 path or
// the AVX-VNNI kernel is actually engaged on a given machine.
func KernelInfo() string {
	return fmt.Sprintf(
		"kernels: q8.int8=%v q4.int8=%v q41.int8=%v | q8.vnni=%v q8.fusedAVX2=%v | workers=%d parThreshold=%d arch=%s",
		useInt8Q8, useInt8Q4, useInt8Q41,
		q8q8VNNIAvail, q8q8FusedAvail,
		nWorkers, parThreshold, runtime.GOARCH,
	)
}
