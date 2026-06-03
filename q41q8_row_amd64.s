//go:build amd64

#include "textflag.h"

// 0x0F per byte and 16×int16(1), shared layout with the Q4_0 kernel constants
// but defined separately to keep this file self-contained.
DATA q41lomask<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA q41lomask<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL q41lomask<>(SB), RODATA|NOPTR, $16
DATA q41ones<>+0(SB)/8, $0x0001000100010001
DATA q41ones<>+8(SB)/8, $0x0001000100010001
DATA q41ones<>+16(SB)/8, $0x0001000100010001
DATA q41ones<>+24(SB)/8, $0x0001000100010001
GLOBL q41ones<>(SB), RODATA|NOPTR, $32

// func q41q8RowDotAVX2(wPtr *byte, xqPtr *int8, xScalePtr *float32, sumXqPtr *int32, groups int) float32
//
// One Q4_1 weight row dotted with a pre-quantized int8 activation vector.
// Q4_1 group = 20 bytes: f16 d (scale) + f16 m (min) + 16 nibble bytes; the
// dequantized weight is d*nibble + m. So per group:
//     result = xScale * ( d*Σ(nibble·xq) + m*Σxq )
// Σ(nibble·xq) uses VPMADDUBSW on the unsigned nibbles (0..15); Σxq is passed in
// via sumXqPtr (per group, activation-only). d and m are decoded in-kernel with
// VCVTPH2PS. All ops VEX-encoded.
TEXT ·q41q8RowDotAVX2(SB), NOSPLIT, $0-44
	MOVQ wPtr+0(FP), SI
	MOVQ xqPtr+8(FP), DI
	MOVQ xScalePtr+16(FP), DX
	MOVQ sumXqPtr+24(FP), R8
	MOVQ groups+32(FP), CX

	VMOVDQU q41lomask<>(SB), X13
	VMOVDQU q41ones<>(SB), Y12
	VXORPS  X15, X15, X15

loop:
	TESTQ CX, CX
	JZ    done

	VMOVDQU     4(SI), X0   // 16 nibble bytes (after d,m)
	VPAND       X13, X0, X1
	VPSRLW      $4, X0, X2
	VPAND       X13, X2, X2
	VINSERTI128 $1, X2, Y1, Y1
	VMOVDQU     (DI), Y3

	VPMADDUBSW Y3, Y1, Y4
	VPMADDWD   Y12, Y4, Y4

	VEXTRACTI128 $1, Y4, X5
	VPADDD       X5, X4, X4
	VPSHUFD      $0x4E, X4, X5
	VPADDD       X5, X4, X4
	VPSHUFD      $0xB1, X4, X5
	VPADDD       X5, X4, X4
	VMOVD        X4, AX // Σ nibble·xq

	MOVWLZX   (SI), BX     // d (f16)
	VMOVD     BX, X6
	VCVTPH2PS X6, X6
	MOVWLZX   2(SI), BX    // m (f16)
	VMOVD     BX, X7
	VCVTPH2PS X7, X7

	VCVTSI2SSL AX, X0, X0    // Σ(nibble·xq) -> float
	VMULSS     X6, X0, X0    // d·Σ(nibble·xq)
	VCVTSI2SSL (R8), X8, X8  // Σxq -> float
	VMULSS     X7, X8, X8    // m·Σxq
	VADDSS     X8, X0, X0    // d·Σ(nibble·xq) + m·Σxq
	VMULSS     (DX), X0, X0  // × activation scale
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
