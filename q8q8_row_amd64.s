//go:build amd64

#include "textflag.h"

// func q8q8RowDotAVX2(wPtr *byte, xqPtr *int8, scalePtr *float32, groups int) float32
//
// Computes one Q8_0 weight row dotted with a pre-quantized int8 activation
// vector, in a single call. wPtr points at the row's first 34-byte group
// (2-byte f16 scale + 32 int8 quants). xqPtr holds groups*32 int8 activations.
// scalePtr holds groups float32 *combined* scales (decoded weight scale ×
// activation scale), so no f16 decode happens in the hot loop.
//
// Per group the int8×int8 MACs use the integer pipeline (VPMADDWD), then a
// single scalar float multiply applies the group scale — keeping all of the
// penalized float-SIMD machinery (F16C, 256-bit float converts) out of the loop.
TEXT ·q8q8RowDotAVX2(SB), NOSPLIT, $0-36
	MOVQ wPtr+0(FP), SI
	MOVQ xqPtr+8(FP), DI
	MOVQ scalePtr+16(FP), DX
	MOVQ groups+24(FP), CX
	VXORPS X8, X8, X8 // float accumulator (lane 0)

loop:
	TESTQ CX, CX
	JZ    done

	VPMOVSXBW 2(SI), Y1   // weight int8[0:16]  -> int16  (skip 2-byte scale)
	VPMOVSXBW 18(SI), Y2  // weight int8[16:32] -> int16
	VPMOVSXBW (DI), Y3    // activation int8[0:16]
	VPMOVSXBW 16(DI), Y4  // activation int8[16:32]
	VPMADDWD  Y3, Y1, Y1  // int16*int16 -> int32 pair sums
	VPMADDWD  Y4, Y2, Y2
	VPADDD    Y2, Y1, Y1  // 8 int32 partials

	VEXTRACTI128 $1, Y1, X5
	VPADDD       X5, X1, X1
	VPSHUFD      $0x4E, X1, X5
	VPADDD       X5, X1, X1
	VPSHUFD      $0xB1, X1, X5
	VPADDD       X5, X1, X1
	VMOVD        X1, AX     // int32 dot for this group

	// VEX-encoded scalar float ops — mixing legacy SSE with the VEX vector ops
	// above would trigger AVX↔SSE transition penalties every group.
	VCVTSI2SSL AX, X0, X0    // int32 -> float32
	VMULSS     (DX), X0, X0  // × combined scale
	VADDSS     X0, X8, X8    // accumulate

	ADDQ $34, SI
	ADDQ $32, DI
	ADDQ $4, DX
	DECQ CX
	JMP  loop

done:
	VMOVSS X8, ret+32(FP)
	VZEROUPPER
	RET
