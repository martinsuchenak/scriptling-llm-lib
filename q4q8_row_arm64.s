//go:build arm64

#include "textflag.h"

// func q4q8RowDotAVX2(wPtr *byte, xqPtr *int8, xScalePtr *float32, corrPtr *int32, groups int) float32
//
// NEON Q4_0 × int8 row dot (the arm64 sibling of the amd64 kernel; same exported
// name so the shared q4q8RowFused var can hold either). Uses the ARMv8.2 dot
// product (SDOT) on signed nibbles, so the Q4_0 zero point folds in (nibble-8)
// and corrPtr is ignored. Lane-accumulation: each group's int32×4 SDOT result is
// converted to float and FMLA'd — scaled by the group's f16 weight scale ×
// activation scale — into a 4-lane float accumulator, reduced once at the end.
//
// SDOT/SCVTF/FADDP are hand-encoded (Go's arm64 assembler doesn't accept them);
// the encodings are verified under qemu.
TEXT ·q4q8RowDotAVX2(SB), NOSPLIT, $0-44
	MOVD wPtr+0(FP), R0
	MOVD xqPtr+8(FP), R1
	MOVD xScalePtr+16(FP), R2
	MOVD groups+32(FP), R4

	VMOVI $15, V30.B16             // 0x0F nibble mask
	VMOVI $8, V31.B16              // Q4_0 zero point
	VEOR  V28.B16, V28.B16, V28.B16 // float accumulator (4 lanes)

loop:
	CBZ R4, done
	SUB $1, R4, R4

	// scale = f16(weight scale) × activation scale
	MOVHU  (R0), R5
	FMOVD  R5, F8
	FCVTHS F8, F8
	FMOVS  (R2), F9
	FMULS  F9, F8, F8

	// decode 16 nibble bytes -> 32 signed int8 (nibble-8)
	ADD   $2, R0, R6
	VLD1  (R6), [V0.B16]
	VAND  V30.B16, V0.B16, V1.B16 // low nibbles  -> weights 0..15
	VUSHR $4, V0.B16, V2.B16      // high nibbles -> weights 16..31
	VSUB  V31.B16, V1.B16, V1.B16 // - 8
	VSUB  V31.B16, V2.B16, V2.B16

	// load 32 int8 activations
	VLD1 (R1), [V3.B16]
	ADD  $16, R1, R7
	VLD1 (R7), [V4.B16]

	// int dot via SDOT, then -> float
	VEOR V5.B16, V5.B16, V5.B16
	WORD $0x4E839425 // SDOT V5.4S, V1.16B, V3.16B
	WORD $0x4E849445 // SDOT V5.4S, V2.16B, V4.16B
	WORD $0x4E21D8A6 // SCVTF V6.4S, V5.4S

	// accumulate: acc += partial × scale (broadcast)
	VDUP  V8.S[0], V7.S4
	VFMLA V6.S4, V7.S4, V28.S4

	ADD $18, R0, R0
	ADD $32, R1, R1
	ADD $4, R2, R2
	JMP loop

done:
	WORD $0x6E3CD79D // FADDP V29.4S, V28.4S, V28.4S
	WORD $0x7E30DBBE // FADDP S30, V29.2S
	FMOVS F30, ret+40(FP)
	RET
