//go:build amd64

#include "textflag.h"

// 16-byte constants: 0x0F per byte (nibble mask) and 0x08 per byte (the Q4_0
// zero point). RODATA, no pointers.
DATA q4lomask<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA q4lomask<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL q4lomask<>(SB), RODATA|NOPTR, $16
DATA q4eight<>+0(SB)/8, $0x0808080808080808
DATA q4eight<>+8(SB)/8, $0x0808080808080808
GLOBL q4eight<>(SB), RODATA|NOPTR, $16

// func q4q8RowDotAVX2(wPtr *byte, xqPtr *int8, scalePtr *float32, groups int) float32
//
// One Q4_0 weight row dotted with a pre-quantized int8 activation vector, in a
// single call. wPtr points at the row's first 18-byte group (2-byte f16 scale +
// 16 nibble bytes). Byte j holds weight j in its low nibble and weight j+16 in
// its high nibble, each a signed value nibble-8. xqPtr holds groups*32 int8
// activations; scalePtr holds groups float32 *combined* scales (weight × act).
//
// The nibble decode happens entirely in SIMD (VPAND/VPSRLW/VPSUBB), so unlike
// the scalar path there is no per-element Go work. All ops are VEX-encoded to
// avoid AVX↔SSE transition penalties with the scalar-float scaling.
TEXT ·q4q8RowDotAVX2(SB), NOSPLIT, $0-36
	MOVQ wPtr+0(FP), SI
	MOVQ xqPtr+8(FP), DI
	MOVQ scalePtr+16(FP), DX
	MOVQ groups+24(FP), CX

	VMOVDQU q4lomask<>(SB), X13 // 0x0F per byte
	VMOVDQU q4eight<>(SB), X14  // 0x08 per byte
	VXORPS  X15, X15, X15       // float accumulator (lane 0)

loop:
	TESTQ CX, CX
	JZ    done

	// --- decode 16 nibble bytes -> 32 int8 weights ---
	VMOVDQU 2(SI), X0    // 16 packed nibble bytes (skip 2-byte scale)
	VPAND   X13, X0, X1  // low nibbles  -> weights 0..15 (0..15)
	VPSRLW  $4, X0, X2
	VPAND   X13, X2, X2  // high nibbles -> weights 16..31 (0..15)
	VPSUBB  X14, X1, X1  // weights 0..15  - 8  (signed -8..7)
	VPSUBB  X14, X2, X2  // weights 16..31 - 8

	VMOVDQU (DI), X3   // xq 0..15
	VMOVDQU 16(DI), X4 // xq 16..31

	// --- dot weights[0..15] · xq[0..15] ---
	VPMOVSXBW X1, X5
	VPMOVSXBW X3, X6
	VPMADDWD  X6, X5, X5
	VPSRLDQ   $8, X1, X7
	VPMOVSXBW X7, X7
	VPSRLDQ   $8, X3, X6
	VPMOVSXBW X6, X6
	VPMADDWD  X6, X7, X7
	VPADDD    X7, X5, X5

	// --- dot weights[16..31] · xq[16..31] ---
	VPMOVSXBW X2, X6
	VPMOVSXBW X4, X7
	VPMADDWD  X7, X6, X6
	VPSRLDQ   $8, X2, X8
	VPMOVSXBW X8, X8
	VPSRLDQ   $8, X4, X9
	VPMOVSXBW X9, X9
	VPMADDWD  X9, X8, X8
	VPADDD    X8, X6, X6

	VPADDD X6, X5, X5 // 4 int32 lanes

	VPSHUFD $0x4E, X5, X6
	VPADDD  X6, X5, X5
	VPSHUFD $0xB1, X5, X6
	VPADDD  X6, X5, X5
	VMOVD   X5, AX // int32 dot for this group

	VCVTSI2SSL AX, X0, X0   // -> float32
	VMULSS     (DX), X0, X0 // × combined scale
	VADDSS     X0, X15, X15 // accumulate

	ADDQ $18, SI
	ADDQ $32, DI
	ADDQ $4, DX
	DECQ CX
	JMP  loop

done:
	VMOVSS X15, ret+32(FP)
	VZEROUPPER
	RET
