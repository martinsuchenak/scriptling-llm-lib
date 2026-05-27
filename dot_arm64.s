//go:build arm64

#include "textflag.h"

// func q8DotRowsAsm(rawPtr *byte, xPtr *float32, groups int) float32
//
// NEON ARM64 implementation. Each Q8_0 group: 2-byte f16 scale + 32 int8 = 34 bytes.
// x data: 32 float32 per group = 128 bytes.
//
// Per group:
//   1. Load f16 scale → scalar F8 (FCVTHS); F8 chosen to avoid V0 conflict
//   2. Preload all 32 x-values into V16..V23 (8 × 4S vectors via VLD1.P)
//   3. Load first 16 int8 → V0.B16; WORD-encoded SSHLL chain extends
//      int8→int16→int32; SCVTF converts int32→f32 (V3..V6); VFMLA
//      accumulates into V26.S4 against V16..V19
//   4. Same for last 16 int8 against V20..V23
//   5. WORD-encoded FADDP reduces V26.S4 → scalar S25 (=F25); F24 += F25 * F8
//
// WORD encodings used for instructions unsupported by Go's assembler:
//   SSHLL  Vd.8H, Vn.8B, #0  = 0x0F08A400 | (Vn<<5) | Vd  (bit10=1 required)
//   SSHLL2 Vd.8H, Vn.16B, #0 = 0x4F08A400 | (Vn<<5) | Vd
//   SSHLL  Vd.4S, Vn.4H, #0  = 0x0F10A400 | (Vn<<5) | Vd
//   SSHLL2 Vd.4S, Vn.8H, #0  = 0x4F10A400 | (Vn<<5) | Vd
//   SCVTF  Vd.4S, Vn.4S      = 0x4E21D800 | (Vn<<5) | Vd
//   FADDP  Vd.4S, Vn.4S, Vm.4S = 0x6E20D400 | (Vm<<16) | (Vn<<5) | Vd
//   FADDP  Sd, Vn.2S          = 0x7E30D800 | (Vn<<5) | Rd

TEXT ·q8DotRowsAsm(SB), NOSPLIT, $0-28
	MOVD rawPtr+0(FP), R0
	MOVD xPtr+8(FP), R1
	MOVD groups+16(FP), R2

	FMOVS $0, F24

loop:
	CBZ R2, done
	SUB $1, R2, R2

	// Prefetch the cache line starting ~2 groups ahead (64 bytes = 8×8, valid for PRFM imm).
	// Groups are 34 bytes each; +64 lands in the cache line [64,127] that covers groups i+2
	// and i+3, hiding L2/L3 latency behind the current group's NEON computation.
	PRFM 64(R0), $0

	// f16 scale → f32 in F8 (V0 is used for int8 data, must not overlap)
	MOVHU (R0), R3
	FMOVD R3, F8
	FCVTHS F8, F8

	// Preload all 32 x-values (128 bytes) into V16..V23
	MOVD   R1, R11
	VLD1.P 16(R11), [V16.S4]
	VLD1.P 16(R11), [V17.S4]
	VLD1.P 16(R11), [V18.S4]
	VLD1.P 16(R11), [V19.S4]
	VLD1.P 16(R11), [V20.S4]
	VLD1.P 16(R11), [V21.S4]
	VLD1.P 16(R11), [V22.S4]
	VLD1.P 16(R11), [V23.S4]

	// Clear per-group accumulator V26
	VEOR V26.B16, V26.B16, V26.B16

	// R4 = int8 data start (raw+2); VLD1.P will advance R4
	ADD $2, R0, R4

	// === First 16 int8 (elements 0..15) ===
	VLD1.P 16(R4), [V0.B16]         // load int8[0..15], R4 += 16

	// SSHLL  V1.8H, V0.8B, #0   — sign-extend lower 8 bytes to 8×int16
	WORD $0x0F08A401
	// SSHLL2 V2.8H, V0.16B, #0  — sign-extend upper 8 bytes to 8×int16
	WORD $0x4F08A402
	// SSHLL  V3.4S, V1.4H, #0   — sign-extend lower 4 int16 to 4×int32
	WORD $0x0F10A423
	// SSHLL2 V4.4S, V1.8H, #0   — sign-extend upper 4 int16 to 4×int32
	WORD $0x4F10A424
	// SSHLL  V5.4S, V2.4H, #0   — sign-extend lower 4 int16 to 4×int32
	WORD $0x0F10A445
	// SSHLL2 V6.4S, V2.8H, #0   — sign-extend upper 4 int16 to 4×int32
	WORD $0x4F10A446

	// SCVTF V3.4S, V3.4S  — int32 → float32
	WORD $0x4E21D863
	// SCVTF V4.4S, V4.4S
	WORD $0x4E21D884
	// SCVTF V5.4S, V5.4S
	WORD $0x4E21D8A5
	// SCVTF V6.4S, V6.4S
	WORD $0x4E21D8C6

	// V26 += V3..V6 * x[0..15]  (FMLA V26.4S, Vn.4S, Vm.4S)
	VFMLA V3.S4,  V16.S4, V26.S4
	VFMLA V4.S4,  V17.S4, V26.S4
	VFMLA V5.S4,  V18.S4, V26.S4
	VFMLA V6.S4,  V19.S4, V26.S4

	// === Second 16 int8 (elements 16..31) ===
	VLD1 (R4), [V0.B16]

	// SSHLL  V1.8H, V0.8B, #0
	WORD $0x0F08A401
	// SSHLL2 V2.8H, V0.16B, #0
	WORD $0x4F08A402
	// SSHLL  V3.4S, V1.4H, #0
	WORD $0x0F10A423
	// SSHLL2 V4.4S, V1.8H, #0
	WORD $0x4F10A424
	// SSHLL  V5.4S, V2.4H, #0
	WORD $0x0F10A445
	// SSHLL2 V6.4S, V2.8H, #0
	WORD $0x4F10A446

	// SCVTF V3.4S, V3.4S
	WORD $0x4E21D863
	// SCVTF V4.4S, V4.4S
	WORD $0x4E21D884
	// SCVTF V5.4S, V5.4S
	WORD $0x4E21D8A5
	// SCVTF V6.4S, V6.4S
	WORD $0x4E21D8C6

	// V26 += V3..V6 * x[16..31]
	VFMLA V3.S4,  V20.S4, V26.S4
	VFMLA V4.S4,  V21.S4, V26.S4
	VFMLA V5.S4,  V22.S4, V26.S4
	VFMLA V6.S4,  V23.S4, V26.S4

	// Horizontal float sum: V26.S4 → F25
	// FADDP V27.4S, V26.4S, V26.4S — pairwise: V27[0]=V26[0]+V26[1], V27[1]=V26[2]+V26[3]
	WORD $0x6E3AD75B
	// FADDP S25, V27.2S — scalar: F25 = V27[0]+V27[1]
	WORD $0x7E30DB79

	// Accumulate: F24 += F25 * scale (scale is in F8)
	FMULS F25, F8, F25
	FADDS F25, F24, F24

	// Advance pointers: raw += 34, x += 128
	ADD $34, R0, R0
	ADD $128, R1, R1

	JMP loop

done:
	FMOVS F24, ret+24(FP)
	RET
