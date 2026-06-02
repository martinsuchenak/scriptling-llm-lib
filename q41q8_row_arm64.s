//go:build arm64

#include "textflag.h"

// func q41q8RowDotAVX2(wPtr *byte, xqPtr *int8, xScalePtr *float32, sumXqPtr *int32, groups int) float32
//
// NEON Q4_1 × int8 row dot (arm64 sibling of the amd64 kernel). Q4_1 group = 20
// bytes: f16 d + f16 m + 16 nibble bytes; weight = d*nibble + m. Per group it
// computes  xScale·(d·Σ(nibble·xq) + m·Σxq).  Nibbles stay unsigned (0..15, in
// range for signed SDOT); Σ(nibble·xq) and Σxq are both formed with SDOT (the
// latter against an all-ones vector). Lane-accumulation into a 4-lane float acc,
// reduced once at the end. sumXqPtr is ignored (Σxq is computed in-kernel).
TEXT ·q41q8RowDotAVX2(SB), NOSPLIT, $0-44
	MOVD wPtr+0(FP), R0
	MOVD xqPtr+8(FP), R1
	MOVD xScalePtr+16(FP), R2
	MOVD groups+32(FP), R4

	VMOVI $15, V30.B16             // 0x0F nibble mask
	VMOVI $1, V29.B16              // ones (for Σxq via SDOT)
	VEOR  V28.B16, V28.B16, V28.B16 // float accumulator

loop:
	CBZ R4, done
	SUB $1, R4, R4

	// d, m (f16) and activation scale -> d·xs, m·xs
	MOVHU  (R0), R5
	FMOVD  R5, F8
	FCVTHS F8, F8
	MOVHU  2(R0), R6
	FMOVD  R6, F9
	FCVTHS F9, F9
	FMOVS  (R2), F10
	FMULS  F10, F8, F8 // d·xs
	FMULS  F10, F9, F9 // m·xs

	// nibbles (unsigned 0..15), at offset 4
	ADD   $4, R0, R7
	VLD1  (R7), [V0.B16]
	VAND  V30.B16, V0.B16, V1.B16 // low  -> weights 0..15
	VUSHR $4, V0.B16, V2.B16      // high -> weights 16..31

	// activations
	VLD1 (R1), [V3.B16]
	ADD  $16, R1, R11
	VLD1 (R11), [V4.B16]

	// Σ(nibble·xq) in V5, Σxq in V6
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	WORD $0x4E839425 // SDOT V5.4S, V1.16B, V3.16B
	WORD $0x4E849445 // SDOT V5.4S, V2.16B, V4.16B
	WORD $0x4E8397A6 // SDOT V6.4S, V29.16B, V3.16B
	WORD $0x4E8497A6 // SDOT V6.4S, V29.16B, V4.16B
	WORD $0x4E21D8A7 // SCVTF V7.4S, V5.4S
	WORD $0x4E21D8D0 // SCVTF V16.4S, V6.4S

	// acc += Σ(nibble·xq)·(d·xs) + Σxq·(m·xs)
	VDUP  V8.S[0], V17.S4
	VFMLA V7.S4, V17.S4, V28.S4
	VDUP  V9.S[0], V18.S4
	VFMLA V16.S4, V18.S4, V28.S4

	ADD $20, R0, R0
	ADD $32, R1, R1
	ADD $4, R2, R2
	JMP loop

done:
	WORD $0x6E3CD79B // FADDP V27.4S, V28.4S, V28.4S
	WORD $0x7E30DB7A // FADDP S26, V27.2S
	FMOVS F26, ret+40(FP)
	RET
