//go:build amd64

#include "textflag.h"

DATA q41vlomask<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA q41vlomask<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL q41vlomask<>(SB), RODATA|NOPTR, $16

// func q41q8RowDotVNNI(wPtr *byte, xqPtr *int8, xScalePtr *float32, sumXqPtr *int32, groups int) float32
//
// AVX-VNNI variant of q41q8RowDotAVX2: one Q4_1 weight row dotted with a
// pre-quantized int8 activation vector. Per group:
//     result = xScale * ( d*Σ(nibble·xq) + m*Σxq )
// The Q4 nibbles are already unsigned (0..15), so VPDPBUSD (unsigned×signed,
// 32 int8 MACs/instruction, no int16 widening) computes Σ(nibble·xq) directly —
// no +128 bias or correction term, unlike the Q8 kernel. Σxq is passed in.
TEXT ·q41q8RowDotVNNI(SB), NOSPLIT, $0-44
	MOVQ wPtr+0(FP), SI
	MOVQ xqPtr+8(FP), DI
	MOVQ xScalePtr+16(FP), DX
	MOVQ sumXqPtr+24(FP), R8
	MOVQ groups+32(FP), CX

	VMOVDQU q41vlomask<>(SB), X13
	VXORPS  X15, X15, X15 // float result accumulator

loop:
	TESTQ CX, CX
	JZ    done

	VMOVDQU     4(SI), X0 // 16 nibble bytes (after d,m)
	VPAND       X13, X0, X1
	VPSRLW      $4, X0, X2
	VPAND       X13, X2, X2
	VINSERTI128 $1, X2, Y1, Y1 // Y1 = 32 unsigned nibbles (low half | high half)
	VMOVDQU     (DI), Y3       // Y3 = 32 int8 activations

	VPXOR    Y2, Y2, Y2 // zero the int32 accumulator
	VPDPBUSD Y3, Y1, Y2 // Y2 += unsigned(Y1)·signed(Y3), 8 int32 lanes

	VEXTRACTI128 $1, Y2, X5
	VPADDD       X5, X2, X2
	VPSHUFD      $0x4E, X2, X5
	VPADDD       X5, X2, X2
	VPSHUFD      $0xB1, X2, X5
	VPADDD       X5, X2, X2
	VMOVD        X2, AX // Σ nibble·xq

	MOVWLZX   (SI), BX  // d (f16)
	VMOVD     BX, X6
	VCVTPH2PS X6, X6
	MOVWLZX   2(SI), BX // m (f16)
	VMOVD     BX, X7
	VCVTPH2PS X7, X7

	VCVTSI2SSL AX, X0, X0   // Σ(nibble·xq) -> float
	VMULSS     X6, X0, X0   // d·Σ(nibble·xq)
	VCVTSI2SSL (R8), X8, X8 // Σxq -> float
	VMULSS     X7, X8, X8   // m·Σxq
	VADDSS     X8, X0, X0
	VMULSS     (DX), X0, X0 // × activation scale
	VADDSS     X0, X15, X15

	ADDQ $20, SI
	ADDQ $32, DI
	ADDQ $4, DX
	ADDQ $4, R8
	DECQ CX
	JMP  loop

done:
	VMOVSS X15, ret+40(FP)
	VZEROUPPER
	RET
