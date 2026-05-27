//go:build amd64

#include "textflag.h"

// func q8DotRowsAsm(rawPtr *byte, xPtr *float32, groups int) float32
//
// SSE/AVX implementation for x86-64. Each Q8_0 group: 2-byte f16 scale + 32 int8 = 34 bytes.
// x data: 32 float32 per group = 128 bytes.
//
// ABI0 stack layout ($0-28, no locals):
//   rawPtr  +0(FP)  8 bytes
//   xPtr    +8(FP)  8 bytes
//   groups  +16(FP) 8 bytes
//   ret     +24(FP) 4 bytes
//
// Register map:
//   AX  = rawPtr (advances +34 per group)
//   BX  = xPtr   (advances +128 per group)
//   CX  = groups counter
//   DX  = raw f16 bits
//   SI  = f32 bits of scale (assembled in GP, then moved to X0 via MOVL)
//   DI  = f16 mantissa scratch
//   R8  = sign-bit scratch
//   X0  = scale (f32 scalar)
//   X1..X4, X8..X11 = q*x products (first/second halves)
//   X5, X6 = x-vector loads (reused)
//   X7  = running accumulator

TEXT ·q8DotRowsAsm(SB), NOSPLIT, $0-28
	MOVQ rawPtr+0(FP), AX
	MOVQ xPtr+8(FP), BX
	MOVQ groups+16(FP), CX

	XORPS X7, X7                  // accumulator = 0

loop:
	TESTQ CX, CX
	JZ    done
	DECQ  CX

	// --- f16 → f32 (inline, no F16C required) ---
	MOVWQZX (AX), DX              // DX = raw uint16 f16 bits
	MOVL    DX, SI
	SHRL    $10, SI
	ANDL    $0x1f, SI             // SI = biased exponent (5 bits)
	MOVL    DX, DI
	ANDL    $0x3ff, DI            // DI = mantissa (10 bits)
	TESTL   SI, SI
	JZ      f16_zero_or_denorm
	CMPL    SI, $31
	JE      f16_inf_or_nan

	// Normal f16: f32 = sign | ((exp-15+127)<<23) | (frac<<13)
	SUBL  $15, SI
	ADDL  $127, SI
	SHLL  $23, SI                 // SI = exponent in f32 position
	MOVL  DX, R8
	ANDL  $0x8000, R8
	SHLL  $16, R8                 // R8 = sign bit in f32 position
	ORL   R8, SI
	SHLL  $13, DI
	ORL   DI, SI                  // SI = complete f32 bit pattern
	JMP   f16_done

f16_zero_or_denorm:
	TESTL DI, DI
	JZ    f16_zero
	MOVL  $0, SI                  // denormals are vanishingly rare in LLM weights → 0
	JMP   f16_done
f16_zero:
	MOVL  $0, SI
	JMP   f16_done

f16_inf_or_nan:
	MOVL DX, R8
	ANDL $0x8000, R8
	SHLL $16, R8
	ORL  R8, DI
	ORL  $0x7f800000, DI          // set all exponent bits → ±inf / NaN
	MOVL DI, SI

f16_done:
	// Move f32 bit pattern from GP register directly into XMM (SSE2 MOVD).
	// MOVL GP→XMM never touches the stack, so the return address at 0(SP) is safe.
	MOVL SI, X0

	// --- Sign-extend 32 int8 → int32, then convert to float32 ---
	// VPMOVSXBD: 4 bytes → 4 int32 (requires AVX / SSE4.1)
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

	// Multiply q[i] * x[i] for all 32 pairs
	MOVUPS   (BX), X5
	MOVUPS 16(BX), X6
	MULPS X5, X1                  // q[0..3]  * x[0..3]
	MULPS X6, X2                  // q[4..7]  * x[4..7]
	MOVUPS 32(BX), X5
	MOVUPS 48(BX), X6
	MULPS X5, X3                  // q[8..11] * x[8..11]
	MULPS X6, X4                  // q[12..15]* x[12..15]

	ADDPS X2, X1
	ADDPS X4, X3
	ADDPS X3, X1                  // X1 = partial sums for elements 0..15

	MOVUPS  64(BX), X5
	MOVUPS  80(BX), X6
	MULPS X5, X8                  // q[16..19]* x[16..19]
	MULPS X6, X9                  // q[20..23]* x[20..23]
	MOVUPS  96(BX), X5
	MOVUPS 112(BX), X6
	MULPS X5, X10                 // q[24..27]* x[24..27]
	MULPS X6, X11                 // q[28..31]* x[28..31]

	ADDPS X9, X8
	ADDPS X11, X10
	ADDPS X10, X8                 // X8 = partial sums for elements 16..31

	ADDPS X8, X1                  // X1 = [lane0, lane1, lane2, lane3] partial sums

	// Horizontal sum of X1's 4 lanes → scalar in X1[0]
	// HADDPS X1, X1: X1 = [lane0+lane1, lane2+lane3, lane0+lane1, lane2+lane3]
	HADDPS X1, X1
	// HADDPS X1, X1: X1[0] = (lane0+lane1) + (lane2+lane3) = total dot product
	HADDPS X1, X1

	// Scale and accumulate
	MULSS X0, X1                  // X1[0] *= scale
	ADDSS X1, X7                  // accumulator += this group's result

	// Advance pointers: raw += 34, x += 128
	ADDQ $34, AX
	ADDQ $128, BX

	JMP loop

done:
	MOVSS X7, ret+24(FP)
	RET
