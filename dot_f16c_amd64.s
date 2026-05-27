//go:build amd64

#include "textflag.h"

// func q8DotRowsAsmF16C(rawPtr *byte, xPtr *float32, groups int) float32
//
// F16C-accelerated variant. Identical to q8DotRowsAsmSSE except the
// f16 → f32 scale conversion uses VCVTPH2PS instead of integer arithmetic,
// reducing the per-group decode from ~11 GP instructions to 2.
//
// Requires: F16C + OSXSAVE (Intel Ivy Bridge 2012, AMD Bulldozer 2011).
// Selected at init time in cpu_amd64.go if CPUID confirms support.

TEXT ·q8DotRowsAsmF16C(SB), NOSPLIT, $0-28
	MOVQ rawPtr+0(FP), AX
	MOVQ xPtr+8(FP), BX
	MOVQ groups+16(FP), CX

	XORPS X7, X7

loop:
	TESTQ CX, CX
	JZ    done
	DECQ  CX

	PREFETCHT1 320(AX)       // ~9 groups ahead to cover ~200-cycle DRAM latency

	// --- f16 → f32 via F16C (2 instructions vs ~11) ---
	// Load the 2-byte f16 scale into the low 16 bits of X0, then
	// VCVTPH2PS converts 4 packed f16 → 4 packed f32. Only element 0 matters.
	MOVWQZX (AX), DX
	MOVL    DX, X0              // f16 bits into XMM0[15:0]
	// VCVTPH2PS XMM0, XMM0  (VEX3: c4 e2 79 13 c0)
	LONG $0x1379e2c4
	BYTE $0xc0

	// --- Sign-extend 32 int8 → int32, convert to float32 ---
	VPMOVSXBD  2(AX), X1
	VPMOVSXBD  6(AX), X2
	VPMOVSXBD 10(AX), X3
	VPMOVSXBD 14(AX), X4
	VPMOVSXBD 18(AX), X8
	VPMOVSXBD 22(AX), X9
	VPMOVSXBD 26(AX), X10
	VPMOVSXBD 30(AX), X11

	VCVTDQ2PS X1, X1
	VCVTDQ2PS X2, X2
	VCVTDQ2PS X3, X3
	VCVTDQ2PS X4, X4
	VCVTDQ2PS X8, X8
	VCVTDQ2PS X9, X9
	VCVTDQ2PS X10, X10
	VCVTDQ2PS X11, X11

	MOVUPS   (BX), X5
	MOVUPS 16(BX), X6
	MULPS X5, X1
	MULPS X6, X2
	MOVUPS 32(BX), X5
	MOVUPS 48(BX), X6
	MULPS X5, X3
	MULPS X6, X4

	ADDPS X2, X1
	ADDPS X4, X3
	ADDPS X3, X1

	MOVUPS  64(BX), X5
	MOVUPS  80(BX), X6
	MULPS X5, X8
	MULPS X6, X9
	MOVUPS  96(BX), X5
	MOVUPS 112(BX), X6
	MULPS X5, X10
	MULPS X6, X11

	ADDPS X9, X8
	ADDPS X11, X10
	ADDPS X10, X8
	ADDPS X8, X1

	HADDPS X1, X1
	HADDPS X1, X1

	MULSS X0, X1
	ADDSS X1, X7

	ADDQ $34, AX
	ADDQ $128, BX
	JMP loop

done:
	VZEROUPPER
	MOVSS X7, ret+24(FP)
	RET
