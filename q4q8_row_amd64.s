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
// One Q4_0 weight row dotted with a pre-quantized int8 activation vector.
// wPtr: 18-byte groups (2-byte f16 scale + 16 nibble bytes); xqPtr: groups*32
// int8 activations; xScalePtr: per-group activation scales; corrPtr: per-group
// 8·Σxq (the Q4_0 zero-point correction, activation-only).
//
// Lane-accumulation (à la llama.cpp): the per-group int32×8 partial is NOT
// horizontally reduced. It is converted to float and FMA'd — scaled by the
// group's scalar (f16 weight scale × activation scale, broadcast) — into an
// 8-lane float accumulator, which is reduced ONCE at the end. This removes the
// per-group horizontal-reduce shuffle chain (the kernel's bottleneck once the
// MACs were cheap). The zero-point term Σ scale·(8·Σxq) is accumulated
// separately and subtracted at the end. All ops VEX/FMA-encoded.
TEXT ·q4q8RowDotAVX2(SB), NOSPLIT, $0-48
	MOVQ wPtr+0(FP), SI
	MOVQ xqPtr+8(FP), DI
	MOVQ xScalePtr+16(FP), DX
	MOVQ corrPtr+24(FP), R8
	MOVQ groups+32(FP), CX

	VMOVDQU q4lomask<>(SB), X13 // 0x0F per byte
	VMOVDQU q4ones<>(SB), Y12   // 16×int16(1)
	VXORPS  Y15, Y15, Y15       // 8-lane float accumulator
	VXORPS  X14, X14, X14       // scalar zero-point accumulator (lane 0)

loop:
	TESTQ CX, CX
	JZ    done

	VMOVDQU     2(SI), X0   // 16 nibble bytes
	VPAND       X13, X0, X1 // weights 0..15  (unsigned 0..15)
	VPSRLW      $4, X0, X2
	VPAND       X13, X2, X2 // weights 16..31 (unsigned 0..15)
	VINSERTI128 $1, X2, Y1, Y1
	VMOVDQU     (DI), Y3    // xq (signed)

	VPMADDUBSW Y3, Y1, Y4   // unsigned·signed -> 16 int16
	VPMADDWD   Y12, Y4, Y4  // pair-sum -> int32×8 (group partial, unreduced)
	VCVTDQ2PS  Y4, Y5       // -> float32×8

	// scale_g = f16 weight scale (group bytes 0..1) × activation scale
	MOVWLZX   (SI), BX
	VMOVD     BX, X6
	VCVTPH2PS X6, X6
	VMULSS    (DX), X6, X6
	VBROADCASTSS X6, Y7

	VFMADD231PS Y5, Y7, Y15 // acc += partial × scale_g (per lane)

	// zero point: corrAcc += scale_g × (8·Σxq)
	VCVTSI2SSL (R8), X8, X8
	VMULSS     X6, X8, X8
	VADDSS     X8, X14, X14

	ADDQ $18, SI
	ADDQ $32, DI
	ADDQ $4, DX
	ADDQ $4, R8
	DECQ CX
	JMP  loop

done:
	VEXTRACTF128 $1, Y15, X0 // fold 8 lanes -> 1
	VADDPS       X0, X15, X15
	VHADDPS      X15, X15, X15
	VHADDPS      X15, X15, X15
	VSUBSS       X14, X15, X15 // − Σ scale·(8·Σxq)
	VMOVSS       X15, ret+40(FP)
	VZEROUPPER
	RET
