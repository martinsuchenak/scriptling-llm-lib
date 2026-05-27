//go:build amd64

#include "textflag.h"

// func q8DotRowsAsmSSE(rawPtr *byte, xPtr *float32, groups int) float32
//
// Baseline SSE4/AVX implementation. Used when F16C is not available.
// f16 → f32 conversion is done with integer arithmetic.
//
// ABI0 stack layout ($0-28, no locals):
//   rawPtr  +0(FP)  8 bytes
//   xPtr    +8(FP)  8 bytes
//   groups  +16(FP) 8 bytes
//   ret     +24(FP) 4 bytes

TEXT ·q8DotRowsAsmSSE(SB), NOSPLIT, $0-28
	MOVQ rawPtr+0(FP), AX
	MOVQ xPtr+8(FP), BX
	MOVQ groups+16(FP), CX

	XORPS X7, X7                  // accumulator = 0.0

loop:
	TESTQ CX, CX
	JZ    done
	DECQ  CX

	// Prefetch ~9 groups ahead to cover DRAM latency (~200 cycles / ~18 cycles-per-group).
	PREFETCHT1 320(AX)

	// --- f16 → f32 (inline integer decode, no F16C required) ---
	MOVWQZX (AX), DX
	MOVL    DX, SI
	SHRL    $10, SI
	ANDL    $0x1f, SI             // SI = biased exponent (5 bits)
	MOVL    DX, DI
	ANDL    $0x3ff, DI            // DI = mantissa (10 bits)
	TESTL   SI, SI
	JZ      f16_zero_or_denorm
	CMPL    SI, $31
	JE      f16_inf_or_nan

	SUBL  $15, SI
	ADDL  $127, SI
	SHLL  $23, SI
	MOVL  DX, R8
	ANDL  $0x8000, R8
	SHLL  $16, R8
	ORL   R8, SI
	SHLL  $13, DI
	ORL   DI, SI
	JMP   f16_done

f16_zero_or_denorm:
	TESTL DI, DI
	JZ    f16_zero
	MOVL  $0, SI                  // denormals are vanishingly rare in LLM weights
	JMP   f16_done
f16_zero:
	MOVL  $0, SI
	JMP   f16_done

f16_inf_or_nan:
	MOVL DX, R8
	ANDL $0x8000, R8
	SHLL $16, R8
	ORL  R8, DI
	ORL  $0x7f800000, DI
	MOVL DI, SI

f16_done:
	MOVL SI, X0                   // f32 bits → XMM0 (SSE2 MOVD, no stack touch)

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

	// q[i] * x[i] for all 32 pairs
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
	ADDPS X8, X1               // X1 = [lane0, lane1, lane2, lane3]

	HADDPS X1, X1              // X1 = [l0+l1, l2+l3, ...]
	HADDPS X1, X1              // X1[0] = total dot product

	MULSS X0, X1
	ADDSS X1, X7

	ADDQ $34, AX
	ADDQ $128, BX
	JMP loop

done:
	// Zero upper YMM state before returning to prevent AVX→SSE transition
	// penalties in any caller that uses legacy SSE after this function.
	VZEROUPPER
	MOVSS X7, ret+24(FP)
	RET


// func cpuHasF16C() bool
//
// Returns true when CPUID leaf 1 reports both OSXSAVE (ECX bit 27) and
// F16C (ECX bit 29). OSXSAVE being set on a 64-bit OS implies the kernel
// saves/restores AVX state, making VCVTPH2PS safe to execute.
// CPUID clobbers AX, BX, CX, DX — all caller-saved in ABI0.
TEXT ·cpuHasF16C(SB), NOSPLIT, $0-1
	MOVL $1, AX
	CPUID
	// bit 27 = OSXSAVE (1<<27 = 0x0800_0000)
	// bit 29 = F16C    (1<<29 = 0x2000_0000)
	// mask                    = 0x2800_0000
	ANDL  $0x28000000, CX
	MOVL  $0, AX
	CMPL  CX, $0x28000000
	JNE   no_f16c
	MOVL  $1, AX
no_f16c:
	MOVB  AX, ret+0(FP)
	RET


// func cpuHasAVX2() bool
//
// Returns true when CPUID confirms OSXSAVE (leaf 1 ECX bit 27) and
// AVX2 (leaf 7 sub-leaf 0 EBX bit 5). OSXSAVE being set on a 64-bit OS
// implies the kernel saves/restores the full YMM state, making 256-bit VEX
// instructions safe to execute. Any CPU with AVX2 also ships with F16C,
// so this path subsumes the F16C check.
// CPUID clobbers AX, BX, CX, DX; R9 is used to save leaf-1 ECX across the
// second CPUID call and is caller-saved in ABI0.
TEXT ·cpuHasAVX2(SB), NOSPLIT, $0-1
	MOVL $1, AX
	CPUID
	MOVL CX, R9                // save leaf-1 ECX

	MOVL $7, AX
	XORL CX, CX                // sub-leaf 0
	CPUID
	// BX = leaf-7 EBX; R9 = leaf-1 ECX
	// bit 27 of R9 = OSXSAVE (0x08000000)
	// bit  5 of BX = AVX2    (0x00000020)
	ANDL $0x08000000, R9
	ANDL $0x00000020, BX
	MOVL $0, AX
	CMPL R9, $0x08000000
	JNE  no_avx2
	CMPL BX, $0x00000020
	JNE  no_avx2
	MOVL $1, AX
no_avx2:
	MOVB AX, ret+0(FP)
	RET
