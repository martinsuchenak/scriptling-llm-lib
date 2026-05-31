//go:build amd64

#include "textflag.h"

// func int8DotAVX2(aPtr, bPtr *int8, n int) int32
//
// Returns sum(a[i]*b[i]) over n int8 values (n must be a multiple of 32) as an
// int32, using AVX2 integer instructions only — no F16C and no int→float
// conversion. This matters because the float SIMD path (VCVTPH2PS / 256-bit
// float converts) is heavily penalized on some virtualized hosts, whereas the
// integer pipeline runs at full speed; keeping the hot loop purely integer is
// what makes Q8×Q8 worthwhile there.
TEXT ·int8DotAVX2(SB), NOSPLIT, $0-28
	MOVQ aPtr+0(FP), AX
	MOVQ bPtr+8(FP), BX
	MOVQ n+16(FP), CX
	VPXOR Y0, Y0, Y0 // int32 accumulator (8 lanes)

loop:
	TESTQ CX, CX
	JZ    done

	VPMOVSXBW (AX), Y1   // a[0:16]  int8 -> int16
	VPMOVSXBW 16(AX), Y2 // a[16:32] int8 -> int16
	VPMOVSXBW (BX), Y3   // b[0:16]
	VPMOVSXBW 16(BX), Y4 // b[16:32]
	VPMADDWD  Y3, Y1, Y1 // int16*int16 -> int32 pair sums
	VPMADDWD  Y4, Y2, Y2
	VPADDD    Y1, Y0, Y0
	VPADDD    Y2, Y0, Y0

	ADDQ $32, AX
	ADDQ $32, BX
	SUBQ $32, CX
	JMP  loop

done:
	VEXTRACTI128 $1, Y0, X1   // fold high 128 into low
	VPADDD       X1, X0, X0
	VPSHUFD      $0x4E, X0, X1
	VPADDD       X1, X0, X0
	VPSHUFD      $0xB1, X0, X1
	VPADDD       X1, X0, X0
	VMOVD        X0, AX
	VZEROUPPER
	MOVL         AX, ret+24(FP)
	RET
