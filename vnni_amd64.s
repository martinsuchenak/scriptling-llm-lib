//go:build amd64

#include "textflag.h"

// 32-byte constant: 0x80 per byte, to flip signed weight bytes to the unsigned
// (+128) representation VPDPBUSD needs.
DATA vnnimask<>+0(SB)/8, $0x8080808080808080
DATA vnnimask<>+8(SB)/8, $0x8080808080808080
DATA vnnimask<>+16(SB)/8, $0x8080808080808080
DATA vnnimask<>+24(SB)/8, $0x8080808080808080
GLOBL vnnimask<>(SB), RODATA|NOPTR, $32

// func cpuHasAVXVNNI() bool
// CPUID leaf 7 sub-leaf 1, EAX bit 4 (AVX-VNNI), plus OSXSAVE (leaf 1 ECX 27).
TEXT ·cpuHasAVXVNNI(SB), NOSPLIT, $0-1
	MOVL $1, AX
	CPUID
	MOVL CX, R9          // leaf-1 ECX
	MOVL $7, AX
	MOVL $1, CX          // sub-leaf 1
	CPUID
	ANDL $0x08000000, R9 // OSXSAVE
	ANDL $0x00000010, AX // AVX-VNNI (EAX bit 4)
	MOVL $0, DX
	CMPL R9, $0x08000000
	JNE  no_vnni
	CMPL AX, $0x00000010
	JNE  no_vnni
	MOVL $1, DX
no_vnni:
	MOVB DX, ret+0(FP)
	RET

// func q8q8RowDotVNNI(wPtr *byte, xqPtr *int8, scalePtr *float32, corrPtr *int32, groups int) float32
//
// Like q8q8RowDotAVX2 but uses AVX-VNNI's VPDPBUSD for the int8 MACs: 32 MACs
// per instruction with no int16 widening. VPDPBUSD is unsigned×signed, so the
// weight bytes are flipped to unsigned (XOR 0x80 = +128) and a per-group
// correction of 128·Σxq (passed in via corrPtr) is subtracted before scaling.
// All VEX-encoded to avoid AVX↔SSE transitions.
TEXT ·q8q8RowDotVNNI(SB), NOSPLIT, $0-44
	MOVQ wPtr+0(FP), SI
	MOVQ xqPtr+8(FP), DI
	MOVQ scalePtr+16(FP), DX
	MOVQ corrPtr+24(FP), R8
	MOVQ groups+32(FP), CX

	VMOVDQU vnnimask<>(SB), Y14 // 0x80 per byte
	VXORPS  X11, X11, X11       // float accumulator (lane 0)

loop:
	TESTQ CX, CX
	JZ    done

	VPXOR    Y2, Y2, Y2   // zero int32 accumulator for this group
	VMOVDQU  2(SI), Y0    // 32 weight int8 (skip 2-byte scale)
	VPXOR    Y14, Y0, Y0  // -> unsigned (weight + 128)
	VMOVDQU  (DI), Y1     // 32 activation int8
	VPDPBUSD Y1, Y0, Y2   // Y2 += unsigned(Y0) · signed(Y1)

	VEXTRACTI128 $1, Y2, X3
	VPADDD       X3, X2, X2
	VPSHUFD      $0x4E, X2, X3
	VPADDD       X3, X2, X2
	VPSHUFD      $0xB1, X2, X3
	VPADDD       X3, X2, X2
	VMOVD        X2, AX // raw VPDPBUSD sum for this group

	SUBL       (R8), AX     // -= 128·Σxq (offset correction)
	VCVTSI2SSL AX, X4, X4    // -> float32
	VMULSS     (DX), X4, X4  // × combined scale
	VADDSS     X4, X11, X11  // accumulate

	ADDQ $34, SI
	ADDQ $32, DI
	ADDQ $4, DX
	ADDQ $4, R8
	DECQ CX
	JMP  loop

done:
	VMOVSS X11, ret+40(FP)
	VZEROUPPER
	RET
