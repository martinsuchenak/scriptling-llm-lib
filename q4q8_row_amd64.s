//go:build amd64

#include "textflag.h"

// 0x0F per byte (nibble mask) and 16×int16(1) for the VPMADDWD pair-sum.
DATA q4lomask<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA q4lomask<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL q4lomask<>(SB), RODATA|NOPTR, $16
DATA q4ones<>+0(SB)/8, $0x0001000100010001
DATA q4ones<>+8(SB)/8, $0x0001000100010001
DATA q4ones<>+16(SB)/8, $0x0001000100010001
DATA q4ones<>+24(SB)/8, $0x0001000100010001
GLOBL q4ones<>(SB), RODATA|NOPTR, $32

// func q4q8RowDotAVX2(wPtr *byte, xqPtr *int8, xScalePtr *float32, corrPtr *int32, groups int) float32
//
// One Q4_0 weight row dotted with a pre-quantized int8 activation vector, in a
// single call. wPtr points at the row's first 18-byte group (2-byte f16 scale +
// 16 nibble bytes); byte j holds weight j (low nibble) and weight j+16 (high
// nibble). xqPtr holds groups*32 int8 activations, xScalePtr the per-group
// activation scales, corrPtr the per-group correction 8·Σxq.
//
// Nibbles are kept UNSIGNED (0..15) so the int8 MACs use VPMADDUBSW directly —
// no int16 widening, no per-element work. The Q4_0 zero point is applied via
// the correction term: Σ(nibble-8)·xq = Σnibble·xq − 8·Σxq. The f16 weight
// scale is decoded in-kernel (VCVTPH2PS). All ops VEX-encoded.
TEXT ·q4q8RowDotAVX2(SB), NOSPLIT, $0-48
	MOVQ wPtr+0(FP), SI
	MOVQ xqPtr+8(FP), DI
	MOVQ xScalePtr+16(FP), DX
	MOVQ corrPtr+24(FP), R8
	MOVQ groups+32(FP), CX

	VMOVDQU q4lomask<>(SB), X13 // 0x0F per byte
	VMOVDQU q4ones<>(SB), Y12   // 16×int16(1)
	VXORPS  X15, X15, X15       // float accumulator (lane 0)

loop:
	TESTQ CX, CX
	JZ    done

	VMOVDQU     2(SI), X0   // 16 nibble bytes
	VPAND       X13, X0, X1 // weights 0..15  (unsigned 0..15) -> low lane of Y1
	VPSRLW      $4, X0, X2
	VPAND       X13, X2, X2 // weights 16..31 (unsigned 0..15)
	VINSERTI128 $1, X2, Y1, Y1 // Y1 = [w0..15 | w16..31]
	VMOVDQU     (DI), Y3    // xq 0..31 (signed)

	VPMADDUBSW Y3, Y1, Y4   // unsigned(Y1)·signed(Y3) -> 16 int16
	VPMADDWD   Y12, Y4, Y4  // pair-sum -> 8 int32

	VEXTRACTI128 $1, Y4, X5
	VPADDD       X5, X4, X4
	VPSHUFD      $0x4E, X4, X5
	VPADDD       X5, X4, X4
	VPSHUFD      $0xB1, X4, X5
	VPADDD       X5, X4, X4
	VMOVD        X4, AX // Σ nibble·xq

	SUBL (R8), AX // − 8·Σxq  => Σ(nibble-8)·xq

	MOVWLZX   (SI), BX
	VMOVD     BX, X6
	VCVTPH2PS X6, X6        // f16 weight scale -> f32
	VMULSS    (DX), X6, X6  // × activation scale
	VCVTSI2SSL AX, X0, X0   // int dot -> float32
	VMULSS     X6, X0, X0   // × combined scale
	VADDSS     X0, X15, X15

	ADDQ $18, SI
	ADDQ $32, DI
	ADDQ $4, DX
	ADDQ $4, R8
	DECQ CX
	JMP  loop

done:
	VMOVSS X15, ret+40(FP)
	VZEROUPPER
	RET
