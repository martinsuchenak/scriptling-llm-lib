//go:build amd64

#include "textflag.h"

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
// Lane-accumulation Q4_0×int8 row dot (see git history for the technique). This
// version unrolls 4 groups into 4 INDEPENDENT float accumulators to hide the
// VFMADD latency chain (a single accumulator is loop-carried at ~4 cyc/group and
// starves wide cores). The Q4_0 zero point is folded into lane 0 of each group's
// int32 partial (VPSUBD by corr) so there is no separate scalar correction
// chain. Reduced once at the end.
//
// Registers: Y0=ones, X1=0x0F mask, Y2..Y5 = 4 accumulators, the rest scratch.
#define Q4BODY(ACC) \
	VMOVDQU 2(SI), X6        \
	VPAND X1, X6, X7         \
	VPSRLW $4, X6, X6        \
	VPAND X1, X6, X6         \
	VINSERTI128 $1, X6, Y7, Y7 \
	VMOVDQU (DI), Y8         \
	VPMADDUBSW Y8, Y7, Y7    \
	VPMADDWD Y0, Y7, Y7      \
	VMOVD (R8), X9           \
	VPSUBD Y9, Y7, Y7        \
	VCVTDQ2PS Y7, Y7         \
	MOVWLZX (SI), BX         \
	VMOVD BX, X10            \
	VCVTPH2PS X10, X10       \
	VMULSS (DX), X10, X10    \
	VBROADCASTSS X10, Y10    \
	VFMADD231PS Y7, Y10, ACC \
	ADDQ $18, SI             \
	ADDQ $32, DI             \
	ADDQ $4, DX              \
	ADDQ $4, R8

TEXT ·q4q8RowDotAVX2(SB), NOSPLIT, $0-48
	MOVQ wPtr+0(FP), SI
	MOVQ xqPtr+8(FP), DI
	MOVQ xScalePtr+16(FP), DX
	MOVQ corrPtr+24(FP), R8
	MOVQ groups+32(FP), CX

	VMOVDQU q4ones<>(SB), Y0
	VMOVDQU q4lomask<>(SB), X1
	VXORPS  Y2, Y2, Y2 // 4 independent accumulators
	VXORPS  Y3, Y3, Y3
	VXORPS  Y4, Y4, Y4
	VXORPS  Y5, Y5, Y5

unroll4:
	CMPQ CX, $4
	JL   tail
	Q4BODY(Y2)
	Q4BODY(Y3)
	Q4BODY(Y4)
	Q4BODY(Y5)
	SUBQ $4, CX
	JMP  unroll4

tail:
	TESTQ CX, CX
	JZ    done
	Q4BODY(Y2)
	DECQ CX
	JMP  tail

done:
	VADDPS Y3, Y2, Y2
	VADDPS Y5, Y4, Y4
	VADDPS Y4, Y2, Y2
	VEXTRACTF128 $1, Y2, X3
	VADDPS  X3, X2, X2
	VHADDPS X2, X2, X2
	VHADDPS X2, X2, X2
	VMOVSS  X2, ret+40(FP)
	VZEROUPPER
	RET
