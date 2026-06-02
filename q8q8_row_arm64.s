//go:build arm64

#include "textflag.h"

// func q8q8RowDotAVX2(wPtr *byte, xqPtr *int8, scalePtr *float32, groups int) float32
//
// NEON Q8_0 × int8 row dot (arm64 sibling of the amd64 fused kernel; same
// exported name for the shared q8q8RowFused var). wPtr: 34-byte groups (2-byte
// f16 scale + 32 int8 weights); xqPtr: groups*32 int8 activations; scalePtr:
// groups pre-combined float32 scales (weight scale × activation scale).
//
// The int8×int8 MACs use the ARMv8.2 dot product (SDOT); each group's int32×4
// result is converted to float and FMLA'd, scaled by the combined scalar, into a
// 4-lane float accumulator reduced once at the end. SDOT/SCVTF/FADDP are
// hand-encoded and verified under qemu.
TEXT ·q8q8RowDotAVX2(SB), NOSPLIT, $0-36
	MOVD wPtr+0(FP), R0
	MOVD xqPtr+8(FP), R1
	MOVD scalePtr+16(FP), R2
	MOVD groups+24(FP), R4

	VEOR V28.B16, V28.B16, V28.B16 // float accumulator

loop:
	CBZ R4, done
	SUB $1, R4, R4

	FMOVS (R2), F8 // combined scale

	ADD  $2, R0, R6
	VLD1 (R6), [V1.B16] // weights 0..15
	ADD  $16, R6, R7
	VLD1 (R7), [V2.B16] // weights 16..31
	VLD1 (R1), [V3.B16] // xq 0..15
	ADD  $16, R1, R8
	VLD1 (R8), [V4.B16] // xq 16..31

	VEOR V5.B16, V5.B16, V5.B16
	WORD $0x4E839425 // SDOT V5.4S, V1.16B, V3.16B
	WORD $0x4E849445 // SDOT V5.4S, V2.16B, V4.16B
	WORD $0x4E21D8A6 // SCVTF V6.4S, V5.4S

	VDUP  V8.S[0], V7.S4
	VFMLA V6.S4, V7.S4, V28.S4

	ADD $34, R0, R0
	ADD $32, R1, R1
	ADD $4, R2, R2
	JMP loop

done:
	WORD $0x6E3CD79D // FADDP V29.4S, V28.4S, V28.4S
	WORD $0x7E30DBBE // FADDP S30, V29.2S
	FMOVS F30, ret+32(FP)
	RET
