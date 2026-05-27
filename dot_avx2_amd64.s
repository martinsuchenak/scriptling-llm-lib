//go:build amd64

#include "textflag.h"

// func q8DotRowsAsmAVX2(rawPtr *byte, xPtr *float32, groups int) float32
//
// AVX2 + F16C variant.
//
// Key design choices:
//
// 1. 256-bit YMM registers: each VPMOVSXBD processes 8 int8 values instead
//    of 4, halving the instruction count vs the SSE/F16C variants.
//
// 2. Deferred horizontal reduction: all scaled partial sums are accumulated
//    into the 256-bit Y15 register. The expensive port-5 pair (VEXTRACTF128
//    + 2×VHADDPS) runs exactly once after the loop instead of per group.
//    This removes 2×VHADDPS from the per-group port-5 budget.
//
// 3. VPBROADCASTD broadcasts the f32 scale to all 8 YMM lanes so the scaled
//    partial sums can be accumulated in 256-bit before the final reduction.
//
// Per-group port-5 instructions: 4×VPMOVSXBD + 1×VPBROADCASTD = 5 cycles.
//
// Requires: AVX2 + OSXSAVE (Intel Haswell 2013+, AMD Ryzen Zen+).

TEXT ·q8DotRowsAsmAVX2(SB), NOSPLIT, $0-28
	MOVQ rawPtr+0(FP), AX
	MOVQ xPtr+8(FP), BX
	MOVQ groups+16(FP), CX

	VXORPS Y15, Y15, Y15    // 256-bit accumulator = 0.0

loop:
	TESTQ CX, CX
	JZ    done
	DECQ  CX

	PREFETCHT1 64(AX)

	// --- f16 → f32 via F16C (implied by AVX2) ---
	MOVWQZX (AX), DX
	MOVL    DX, X0
	// VCVTPH2PS XMM0, XMM0  (VEX3: c4 e2 79 13 c0)
	LONG $0x1379e2c4
	BYTE $0xc0

	// Broadcast scalar scale to all 8 YMM lanes.
	VPBROADCASTD X0, Y0

	// --- Sign-extend 32 int8 → 32 int32 in 4 YMM registers ---
	VPMOVSXBD  2(AX), Y1
	VPMOVSXBD 10(AX), Y2
	VPMOVSXBD 18(AX), Y3
	VPMOVSXBD 26(AX), Y4

	// --- int32 → float32 ---
	VCVTDQ2PS Y1, Y1
	VCVTDQ2PS Y2, Y2
	VCVTDQ2PS Y3, Y3
	VCVTDQ2PS Y4, Y4

	// --- Load x: 32 × float32 = 128 bytes = 4 × 256-bit ---
	VMOVUPS   (BX), Y5
	VMOVUPS 32(BX), Y6
	VMOVUPS 64(BX), Y7
	VMOVUPS 96(BX), Y8

	// --- Multiply: 4 independent accumulators (max ILP) ---
	VMULPS Y5, Y1, Y1
	VMULPS Y6, Y2, Y2
	VMULPS Y7, Y3, Y3
	VMULPS Y8, Y4, Y4

	// --- Tree-reduce to 8 floats in Y1 ---
	VADDPS Y2, Y1, Y1
	VADDPS Y4, Y3, Y3
	VADDPS Y3, Y1, Y1

	// --- Scale 8 partial sums, then accumulate into Y15 ---
	VMULPS Y0, Y1, Y1
	VADDPS Y1, Y15, Y15

	ADDQ $34, AX
	ADDQ $128, BX
	JMP loop

done:
	// One-time horizontal reduction of Y15 (8 floats) → scalar.
	VEXTRACTF128 $1, Y15, X1    // X1 = upper 4 floats; X15 aliases lower 4
	VADDPS X1, X15, X15          // X15 = lower + upper (lane-wise)
	VHADDPS X15, X15, X15
	VHADDPS X15, X15, X15        // X15[0] = total

	VZEROUPPER
	MOVSS X15, ret+24(FP)
	RET
