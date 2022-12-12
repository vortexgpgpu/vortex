#include "riscv_vector.h"
#define _MM_ALIGN64 __attribute__((aligned (64)))

#define MUSTINLINE __attribute__((always_inline))

//---------------------------------------------------------------------------
// DATA TYPES

#define _MMR_f64        	vfloat64m1_t //<vscale x 1 x double>  //__epi_1xf64
#define _MMR_f32        	vfloat32m1_t //<vscale x 2 x float>  //__epi_2xf32

#define _MMR_2xf64			  vfloat64m2_t //<vscale x 2 x double> //__epi_2xf64
#define _MMR_4xf64			  vfloat32m2_t //<vscale x 4 x float> //__epi_4xf32

#define _MMR_i64          vint64m1_t //<vscale x 1 x i64> //__epi_1xi64
#define _MMR_u64          vuint64m1_t
#define _MMR_i32        	vint32m1_t //<vscale x 2 x i32> //__epi_2xi32

//---------------------------------------------------------------------------
// TYPE CAST

#define _MM_CAST_u64_i64(op1)     vreinterpret_v_u64m1_i64m1(op1)

#define _MM_CAST_i64_u64(op1)     vreinterpret_v_i64m1_u64m1(op1)

#define _MM_CAST_f64_i64(op1)     vreinterpret_v_f64m1_i64m1(op1)

#define _MM_CAST_f64_u64(op1)     vreinterpret_v_f64m1_u64m1(op1)

#define _MM_CAST_i64_f64(op1)     vreinterpret_v_i64m1_f64m1(op1)

#define _MM_CAST_u32_i32(op1)     vreinterpret_v_u32m1_i32m1(op1)

#define _MM_CAST_f32_i32(op1)     vreinterpret_v_f32m1_i32m1(op1)

#define _MM_CAST_f32_u32(op1)     vreinterpret_v_f32m1_u32m1(op1)

#define _MM_CAST_i32_f32(op1)     vreinterpret_v_i32m1_f32m1(op1)


#define _MM_CAST_i64_i32(op1)     vreinterpret_v_i64m1_i32m1(op1)


//---------------------------------------------------------------------------
// INTEGER INTRINSICS

//#define _MM_LOAD_i64        __builtin_epi_vload_1xi64
#define _MM_LOAD_i64(op1, op2)     vle64_v_i64m1(op1, op2)

#define _MM_LOAD_u64(op1, op2)     vle64_v_u64m1(op1, op2)

//#define _MM_LOAD_i32    	__builtin_epi_vload_2xi32
#define _MM_LOAD_i32(op1, op2)     vle32_v_i32m1(op1, op2)

//#define _MM_LOAD_INDEX_i64 __builtin_epi_vload_indexed_1xi64
#define _MM_LOAD_INDEX_i64(op1, op2, op3)  vloxei64_v_i64m1(op1, op2, op3)

//#define _MM_LOAD_INDEX_i32 __builtin_epi_vload_indexed_2xi32
#define _MM_LOAD_INDEX_i32(op1, op2, op3) vloxei32_v_i32m1(op1, op2, op3)

//#define _MM_STORE_i64   	__builtin_epi_vstore_1xi64
#define _MM_STORE_i64(op1, op2, op3)    vse64_v_i64m1(op1, op2, op3)

//#define _MM_STORE_i32   	__builtin_epi_vstore_2xi32
#define _MM_STORE_i32(op1, op2, op3)    vse32_v_i32m1(op1, op2, op3)

//#define _MM_ADD_i64     	__builtin_epi_vadd_1xi64
#define _MM_ADD_i64(op1, op2, op3) vadd_vv_i64m1(op1, op2, op3)

//#define _MM_ADD_i32     	__builtin_epi_vadd_2xi32
#define _MM_ADD_i32(op1, op2, op3) vadd_vv_i32m1(op1, op2, op3)

//#define _MM_SUB_i64			  __builtin_epi_vsub_1xi64
#define _MM_SUB_i64(op1, op2, op3) vsub_vv_i64m1(op1, op2, op3)

//#define _MM_SUB_i32			  __builtin_epi_vsub_2xi32
#define _MM_SUB_i32(op1, op2, op3)  vsub_vv_i32m1(op1, op2, op3)

//#define _MM_ADD_i64_MASK  __builtin_epi_vadd_1xi64_mask
/*
0.7.1
__epi_1xi64 test_vadd_1xi64_mask(__epi_1xi64 arg_0, __epi_1xi64 arg_1, __epi_1xi64 arg_2, __epi_1xi1 arg_3, unsigned long int arg_4)
{
    return __builtin_epi_vadd_1xi64_mask(arg_0, arg_1, arg_2, arg_3, arg_4);
}
0.9/1.0:
vint64m1_t vadd_vv_i64m1_m (vbool64_t mask, vint64m1_t maskedoff, vint64m1_t op1, vint64m1_t op2);
*/
#define _MM_ADD_i64_MASK(op1, op2, op3, op4, op5) vadd_vv_i64m1_m(op4, op1, op2, op3, op5)

// #define _MM_ADD_i32_MASK  __builtin_epi_vadd_2xi32_mask 
#define _MM_ADD_i32_MASK(op1, op2, op3, op4, op5) vadd_vv_i32m1_m(op4, op1, op2, op3, op5)

//#define _MM_MUL_i64       __builtin_epi_vmul_1xi64
#define _MM_MUL_i64(op1, op2, op3) vmul_vv_i64m1(op1, op2, op3)

//#define _MM_MUL_i32       __builtin_epi_vmul_2xi32
#define _MM_MUL_i32(op1, op2, op3) vmul_vv_i32m1(op1, op2, op3)

//#define _MM_DIV_i64       __builtin_epi_vdiv_1xi64
#define _MM_DIV_i64(op1, op2, op3) vdiv_vv_i64m1(op1, op2, op3)

//#define _MM_DIV_i32       __builtin_epi_vdiv_2xi32
#define _MM_DIV_i32(op1, op2, op3) vdiv_vv_i32m1(op1, op2, op3)

//#define _MM_REM_i64       __builtin_epi_vrem_1xi64
#define _MM_REM_i64(op1, op2, op3) vrem_vv_i64m1(op1, op2, op3)

//#define _MM_REM_i32       __builtin_epi_vrem_2xi32
#define _MM_REM_i32(op1, op2, op3) vrem_vv_i32m1(op1, op2, op3)

/*
log:
 Remove 'vbroadcast' builtin & intrinsic in favour of 'vmv.v.x' and 'vfmv.v.f'
    - Also replace RISCVISD::VBROADCAST node with RISCVISD::VMV_V_X and
    RISCVISD::VFMV_V_F, so that they can be more precisely constrained.
    - Update EPIFoldBroadcast phase accordingly.

0.7.1:
__epi_1xi64 test_vbroadcast_1xi64(signed long int arg_0, unsigned long int arg_1)
{
    return __builtin_epi_vbroadcast_1xi64(arg_0, arg_1);
}

0.9/1.0:
vint64m1_t vmv_v_x_i64m1 (int64_t src);
*/
//#define _MM_SET_i64     	__builtin_epi_vbroadcast_1xi64
#define _MM_SET_i64(op1, op2) vmv_v_x_i64m1(op1, op2)

#define _MM_SET_u64(op1, op2) vmv_v_x_u64m1(op1, op2)

//#define _MM_SET_i32     	__builtin_epi_vbroadcast_2xi32 
#define _MM_SET_i32(op1, op2) vmv_v_x_i32m1(op1, op2)

#define _MM_SET_u32(op1, op2) vmv_v_x_u32m1(op1, op2)

//#define _MM_MIN_i64         __builtin_epi_vmin_1xi64
#define _MM_MIN_i64(op1, op2, op3) vmin_vv_i64m1(op1, op2, op3)

//#define _MM_MIN_i32         __builtin_epi_vmin_2xi32
#define _MM_MIN_i32(op1, op2, op3) vmin_vv_i32m1(op1, op2, op3)

//#define _MM_MAX_i64         __builtin_epi_vmax_1xi64
#define _MM_MAX_i64(op1, op2, op3) vmax_vv_i64m1(op1, op2, op3)

//#define _MM_MAX_i32         __builtin_epi_vmax_2xi32
#define _MM_MAX_i32(op1, op2, op3) vmax_vv_i32m1(op1, op2, op3)

//#define _MM_SLL_i64     	__builtin_epi_vsll_1xi64
#define _MM_SLL_i64(op1, op2, op3) vsll_vv_i64m1(op1, op2, op3)

//#define _MM_SLL_i32     	__builtin_epi_vsll_2xi32
#define _MM_SLL_i32(op1, op2, op3) vsll_vv_i32m1(op1, op2, op3)

//#define _MM_SRL_i64     	__builtin_epi_vsrl_1xi64
#define _MM_SRL_u64(op1, op2, op3) vsrl_vv_u64m1(op1, op2, op3)

//#define _MM_SRL_i32     	__builtin_epi_vsrl_2xi32
#define _MM_SRL_u32(op1, op2, op3) vsrl_vv_u32m1(op1, op2, op3)

//#define _MM_AND_i64     	__builtin_epi_vand_1xi64
#define _MM_AND_i64(op1, op2, op3) vand_vv_i64m1(op1, op2, op3)

//#define _MM_AND_i32     	__builtin_epi_vand_2xi32
#define _MM_AND_i32(op1, op2, op3) vand_vv_i32m1(op1, op2, op3)

//#define _MM_OR_i64     		__builtin_epi_vor_1xi64
#define _MM_OR_i64(op1, op2, op3) vor_vv_i64m1(op1, op2, op3)

//#define _MM_OR_i32     		__builtin_epi_vor_2xi32
#define _MM_OR_i32(op1, op2, op3) vor_vv_i32m1(op1, op2, op3)

//#define _MM_XOR_i64     	__builtin_epi_vxor_1xi64
#define _MM_XOR_i64(op1, op2, op3) vxor_vv_i64m1(op1, op2, op3)

//#define _MM_XOR_i32     	__builtin_epi_vxor_2xi32
#define _MM_XOR_i32(op1, op2, op3) vxor_vv_i32m1(op1, op2, op3)

//#define _MM_NOT_i64(x)     	_MM_XOR_i64((x),(x), gvl)
#define _MM_NOT_i64(x)  vnot_v_i64m1(x)

//#define _MM_NOT_i32(x)     	_MM_XOR_i32((x),(x), gvl) 
#define _MM_NOT_i32(x) vnot_v_i32m1(x)

/*
0.7.1
__epi_1xi64 test_vredsum_1xi64(__epi_1xi64 arg_0, __epi_1xi64 arg_1, unsigned long int arg_2)
{
    return __builtin_epi_vredsum_1xi64(arg_0, arg_1, arg_2);
}

# *** IR Dump After Finalize ISel and expand pseudo-instructions ***:
# Machine code for function test_vredsum_1xi64: IsSSA, TracksLiveness
Function Live Ins: $v16 in %0, $v17 in %1, $x10 in %2

bb.0.entry:
  liveins: $v16, $v17, $x10
  %2:gpr = COPY $x10
  %1:vr = COPY $v17
  %0:vr = COPY $v16
  %4:vr = IMPLICIT_DEF
  dead %5:gpr = PseudoVSETVLI %2:gpr, 12, implicit-def $vl, implicit-def $vtype
  %3:vr = PseudoVREDSUM_VS_M1 %4:vr(tied-def 0), %0:vr, %1:vr, $noreg, $noreg, -1, implicit $vl, implicit $vtype
  $v16 = COPY %3:vr
  PseudoRET implicit $v16
0.9/1.0:
vint64m1_t vredsum_vs_i64m1_i64m1 (vint64m1_t dst, vint64m1_t vector, vint64m1_t scalar);
*/
//#define _MM_REDSUM_i64   	__builtin_epi_vredsum_1xi64 
#define _MM_REDSUM_i64(op1, op2, op3) vredsum_vs_i64m1_i64m1(op2, op1, op2, op3)

//#define _MM_REDSUM_i32   	__builtin_epi_vredsum_2xi32 
#define _MM_REDSUM_i32(op1, op2, op3) vredsum_vs_i32m1_i32m1(op2, op1, op2, op3)

/*
0.7.1
__epi_1xi64 test_vmerge_1xi64(__epi_1xi64 arg_0, __epi_1xi64 arg_1, __epi_1xi1 arg_2, unsigned long int arg_3)
{
    return __builtin_epi_vmerge_1xi64(arg_0, arg_1, arg_2, arg_3);
}
0.9/1.0:
vint64m1_t vmerge_vvm_i64m1 (vbool64_t mask, vint64m1_t op1, vint64m1_t op2);
*/
//#define _MM_MERGE_i64         __builtin_epi_vmerge_1xi64
#define _MM_MERGE_i64(op1, op2, op3, op4)  vmerge_vvm_i64m1(op3, op1, op2, op4)

//#define _MM_MERGE_i32  		__builtin_epi_vmerge_2xi32
#define _MM_MERGE_i32(op1, op2, op3, op4)  vmerge_vvm_i32m1(op3, op1, op2, op4)

//#define _MM_MADD_i64  		__builtin_epi_vmadd_1xi64
//#define _MM_MADD_i32  		__builtin_epi_vmadd_2xi32
//---------------------------------------------------------------------------
// FLOATING POINT INTRINSICS

//#define _MM_LOAD_f64    	__builtin_epi_vload_1xf64
#define _MM_LOAD_f64(op1, op2) vle64_v_f64m1(op1, op2)

//#define _MM_LOAD_f32    	__builtin_epi_vload_2xf32
#define _MM_LOAD_f32(op1, op2) vle32_v_f32m1(op1, op2)

#define _MM_LOAD_INDEX_f64 __builtin_epi_vload_indexed_1xf64 //TODO, not being used
#define _MM_LOAD_INDEX_f32 __builtin_epi_vload_indexed_2xf32 //TODO, not being used

//#define _MM_STORE_f64   	__builtin_epi_vstore_1xf64
#define _MM_STORE_f64(op1, op2, op3) vse64_v_f64m1(op1, op2, op3)

//#define _MM_STORE_f32   	__builtin_epi_vstore_2xf32
#define _MM_STORE_f32(op1, op2, op3) vse32_v_f32m1(op1, op2, op3)

//#define _MM_MUL_f64     	__builtin_epi_vfmul_1xf64
#define _MM_MUL_f64(op1, op2, op3) vfmul_vv_f64m1(op1, op2, op3)

//#define _MM_MUL_f32     	__builtin_epi_vfmul_2xf32
#define _MM_MUL_f32(op1, op2, op3) vfmul_vv_f32m1(op1, op2, op3)

//#define _MM_ADD_f64     	__builtin_epi_vfadd_1xf64
#define _MM_ADD_f64(op1, op2, op3) vfadd_vv_f64m1(op1, op2, op3)

//#define _MM_ADD_f32     	__builtin_epi_vfadd_2xf32
#define _MM_ADD_f32(op1, op2, op3) vfadd_vv_f32m1(op1, op2, op3)

//#define _MM_SUB_f64     	__builtin_epi_vfsub_1xf64
#define _MM_SUB_f64(op1, op2, op3) vfsub_vv_f64m1(op1, op2, op3)

//#define _MM_SUB_f32     	__builtin_epi_vfsub_2xf32
#define _MM_SUB_f32(op1, op2, op3) vfsub_vv_f32m1(op1, op2, op3)

//#define _MM_SUB_f64_MASK	__builtin_epi_vfsub_1xf64_mask
#define _MM_SUB_f64_MASK(op1, op2, op3, op4, op5) vfsub_vv_f64m1_m(op4, op1, op2, op3, op5)

//#define _MM_SUB_f32_MASK	__builtin_epi_vfsub_2xf32_mask 
#define _MM_SUB_f32_MASK(op1, op2, op3, op4, op5) vfsub_vv_f32m1_m(op4, op1, op2, op3, op5)

//#define _MM_ADD_f64_MASK  __builtin_epi_vfadd_1xf64_mask 
#define _MM_ADD_f64_MASK(op1, op2, op3, op4, op5) vfadd_vv_f64m1_m(op4, op1, op2, op3, op5)

//#define _MM_ADD_f32_MASK  __builtin_epi_vfadd_2xf32_mask
#define _MM_ADD_f32_MASK(op1, op2, op3, op4, op5) vfadd_vv_f32m1_m(op4, op1, op2, op3, op5)

//#define _MM_DIV_f64     	__builtin_epi_vfdiv_1xf64
#define _MM_DIV_f64(op1, op2, op3) vfdiv_vv_f64m1(op1, op2, op3)

//#define _MM_DIV_f32     	__builtin_epi_vfdiv_2xf32
#define _MM_DIV_f32(op1, op2, op3) vfdiv_vv_f32m1(op1, op2, op3)

//#define _MM_DIV_2xf64		__builtin_epi_vfdiv_2xf64
#define _MM_DIV_2xf64(op1, op2, op3)  vfdiv_vv_f64m2(op1, op2, op3)

//#define _MM_DIV_4xf32     	__builtin_epi_vfdiv_4xf32
#define _MM_DIV_4xf32(op1, op2, op3)  vfdiv_vv_f32m2(op1, op2, op3)

//#define _MM_SQRT_f64    	__builtin_epi_vfsqrt_1xf64 
#define _MM_SQRT_f64(op1, op2) vfsqrt_v_f64m1(op1, op2)

//#define _MM_SQRT_f32    	__builtin_epi_vfsqrt_2xf32
#define _MM_SQRT_f32(op1, op2) vfsqrt_v_f32m1(op1, op2) 

//#define _MM_SET_f64     	__builtin_epi_vbroadcast_1xf64 
#define _MM_SET_f64(op1, op2) vfmv_v_f_f64m1(op1, op2)

//#define _MM_SET_f32     	__builtin_epi_vbroadcast_2xf32 
#define _MM_SET_f32(op1, op2) vfmv_v_f_f32m1(op1, op2)

//#define _MM_MIN_f64         __builtin_epi_vfmin_1xf64
#define _MM_MIN_f64(op1, op2, op3) vfmin_vv_f64m1(op1, op2, op3) 

//#define _MM_MIN_f32         __builtin_epi_vfmin_2xf32
#define _MM_MIN_f32(op1, op2, op3) vfmin_vv_f32m1(op1, op2, op3)

//#define _MM_MAX_f64         __builtin_epi_vfmax_1xf64
#define _MM_MAX_f64(op1, op2, op3) vfmax_vv_f64m1(op1, op2, op3)

//#define _MM_MAX_f32         __builtin_epi_vfmax_2xf32
#define _MM_MAX_f32(op1, op2, op3) vfmax_vv_f32m1(op1, op2, op3)

//#define _MM_VFSGNJ_f64      __builtin_epi_vfsgnj_1xf64
#define _MM_VFSGNJ_f64(op1, op2, op3) vfsgnj_vv_f64m1(op1, op2, op3)

//#define _MM_VFSGNJ_f32      __builtin_epi_vfsgnj_2xf32
#define _MM_VFSGNJ_f32(op1, op2, op3) vfsgnj_vv_f32m1(op1, op2, op3)

//#define _MM_VFSGNJN_f64     __builtin_epi_vfsgnjn_1xf64
#define _MM_VFSGNJN_f64(op1, op2, op3) vfsgnjn_vv_f64m1(op1, op2, op3)

//#define _MM_VFSGNJN_f32     __builtin_epi_vfsgnjn_2xf32
#define _MM_VFSGNJN_f32(op1, op2, op3) vfsgnjn_vv_f32m1(op1, op2, op3)

//#define _MM_VFSGNJX_f64     __builtin_epi_vfsgnjx_1xf64
#define _MM_VFSGNJX_f64(op1, op2, op3) vfsgnjx_vv_f64m1(op1, op2, op3)

//#define _MM_VFSGNJX_f32 	__builtin_epi_vfsgnjx_2xf32
#define _MM_VFSGNJX_f32(op1, op2, op3) vfsgnjx_vv_f32m1(op1, op2, op3)

/*
0.7.1:
__epi_1xf64 test_vfmerge_1xf64(__epi_1xf64 arg_0, __epi_1xf64 arg_1, __epi_1xi1 arg_2, unsigned long int arg_3)
{
    return __builtin_epi_vfmerge_1xf64(arg_0, arg_1, arg_2, arg_3);
}

0.9/1.0:
vfloat64m1_t vmerge_vvm_f64m1 (vbool64_t mask, vfloat64m1_t op1, vfloat64m1_t op2);
*/
//#define _MM_MERGE_f64  		__builtin_epi_vfmerge_1xf64
#define _MM_MERGE_f64(op1, op2, op3, op4) vmerge_vvm_f64m1(op3, op1, op2, op4)

//#define _MM_MERGE_f32 		__builtin_epi_vfmerge_2xf32
#define _MM_MERGE_f32(op1, op2, op3, op4) vmerge_vvm_f32m1(op3, op1, op2, op4)

//#define _MM_REDSUM_f64  	__builtin_epi_vfredsum_1xf64 
#define _MM_REDSUM_f64(op1, op2, op3) vfredosum_vs_f64m1_f64m1(op2, op1, op2, op3)

//#define _MM_REDSUM_f32  	__builtin_epi_vfredsum_2xf32
#define _MM_REDSUM_f32(op1, op2, op3) vfredosum_vs_f32m1_f32m1(op2, op1, op2, op3)

#define _MM_REDSUM_f64_MASK __builtin_epi_vfredsum_1xf64_mask //TODO, not being used
#define _MM_REDSUM_f32_MASK __builtin_epi_vfredsum_2xf32_mask  //TODO, not being used


//#define _MM_MACC_f64  		__builtin_epi_vfmacc_1xf64
#define _MM_MACC_f64(op1, op2, op3, op4) vfmacc_vv_f64m1(op1, op2, op3, op4)

//#define _MM_MACC_f32  		__builtin_epi_vfmacc_2xf32
#define _MM_MACC_f32(op1, op2, op3, op4) vfmacc_vv_f32m1(op1, op2, op3, op4)

//#define _MM_NMACC_f64  		__builtin_epi_vfnmacc_1xf64 error
//#define _MM_NMACC_f32  		__builtin_epi_vfnmacc_2xf32 error

//#define _MM_NMSAC_f64  		__builtin_epi_vfnmsac_1xf64 error
//#define _MM_NMSAC_f32  		__builtin_epi_vfnmsac_2xf32 error

//#define _MM_NMSUB_f64  		__builtin_epi_vfnmsub_1xf64 error
//#define _MM_NMSUB_f32  		__builtin_epi_vfnmsub_2xf32 error

//#define _MM_MADD_f64  		__builtin_epi_vfmadd_1xf64
#define _MM_MADD_f64(op1, op2, op3, op4) vfmadd_vv_f64m1(op1, op2, op3, op4)

//#define _MM_MADD_f32  		__builtin_epi_vfmadd_2xf32
#define _MM_MADD_f32(op1, op2, op3, op4) vfmadd_vv_f32m1(op1, op2, op3, op4)

//#define _MM_MADD_f64_MASK  	__builtin_epi_vfmadd_1xf64_mask
//#define _MM_MADD_f32_MASK  	__builtin_epi_vfmadd_2xf32_mask

//---------------------------------------------------------------------------
// CONVERSION INTRINSICS

//#define _MM_VFCVT_F_X_f64   __builtin_epi_vfcvt_f_x_1xf64_1xi64
#define _MM_VFCVT_F_X_f64(op1, op2) vfcvt_f_x_v_f64m1(op1, op2)

//#define _MM_VFCVT_F_X_f32   __builtin_epi_vfcvt_f_x_2xf32_2xi32
#define _MM_VFCVT_F_X_f32(op1, op2) vfcvt_f_x_v_f32m1(op1, op2)

//#define _MM_VFCVT_X_F_i64   __builtin_epi_vfcvt_x_f_1xi64_1xf64
#define _MM_VFCVT_X_F_i64(op1, op2) vfcvt_x_f_v_i64m1(op1, op2)

//#define _MM_VFCVT_X_F_i32   __builtin_epi_vfcvt_x_f_2xi32_2xf32
#define _MM_VFCVT_X_F_i32(op1, op2) vfcvt_x_f_v_i32m1(op1, op2)

//#define _MM_VFWCVT_F_F_f64  __builtin_epi_vfwcvt_f_f_2xf64_2xf32
#define _MM_VFWCVT_F_F_f64(op1, op2) vfwcvt_f_f_v_f64m2(op1, op2)

//#define _MM_VFNCVT_F_F_f32  __builtin_epi_vfncvt_f_f_2xf32_2xf64
#define _MM_VFNCVT_F_F_f32(op1, op2) vfncvt_f_f_w_f32m1(op1, op2)

//#define _MM_VFWCVT_F_X_f64  __builtin_epi_vfwcvt_f_x_2xf64_2xi32
#define _MM_VFWCVT_F_X_f64(op1, op2) vfwcvt_f_x_v_f64m2(op1, op2)

//#define _MM_VFCVT_f32_i32   __builtin_epi_vfcvt_f_x_2xf32_2xi32
#define _MM_VFCVT_f32_i32(op1, op2) vfcvt_f_x_v_f32m1(op1, op2)

#define FENCE()   asm volatile( "fence" : : );

//---------------------------------------------------------------------------
// VECTOR ELEMENT MANIPULATION

// int
/*
0.7.1
__epi_2xi32 test_vslideup_2xi32(__epi_2xi32 arg_0, unsigned long int arg_1, unsigned long int arg_2)
{
    return __builtin_epi_vslideup_2xi32(arg_0, arg_1, arg_2);
}
0.9/1.0:
vint32m2_t vslideup_vx_i32m2 (vint32m2_t dst, vint32m2_t src, size_t offset)
*/
#define _MM_VSLIDEUP_i32    __builtin_epi_vslideup_2xi32 //TODO, not being used
#define _MM_VSLIDEUP_i64    __builtin_epi_vslideup_1xi64 //TODO, not being used

//#define _MM_VSLIDE1UP_i32    __builtin_epi_vslide1up_2xi32
#define _MM_VSLIDE1UP_i32(op1, op2, op3) vslide1up_vx_i32m1(op1, op2, op3)

//#define _MM_VSLIDE1UP_i64    __builtin_epi_vslide1up_1xi64
#define _MM_VSLIDE1UP_i64(op1, op2, op3) vslide1up_vx_i64m1(op1, op2, op3)

#define _MM_VSLIDEUP_i32_MASK    __builtin_epi_vslideup_2xi32_mask //TODO, not being used
#define _MM_VSLIDEUP_i64_MASK    __builtin_epi_vslideup_1xi64_mask //TODO, not being used

//#define _MM_VSLIDEDOWN_i32    __builtin_epi_vslidedown_2xi32 
#define _MM_VSLIDEDOWN_i32(op1, op2, op3) vslidedown_vx_i32m1(op1, op2, op3)

//#define _MM_VSLIDEDOWN_i64    __builtin_epi_vslidedown_1xi64
#define _MM_VSLIDEDOWN_i64(op1, op2, op3) vslidedown_vx_i64m1(op1, op2, op3)

//#define _MM_VSLIDE1DOWN_i32    __builtin_epi_vslide1down_2xi32
#define _MM_VSLIDE1DOWN_i32(op1, op2, op3) vslide1down_vx_i32m1(op1, op2, op3)

//#define _MM_VSLIDE1DOWN_i64    __builtin_epi_vslide1down_1xi64
#define _MM_VSLIDE1DOWN_i64(op1, op2, op3) vslide1down_vx_i64m1(op1, op2, op3)

#define _MM_VSLIDEDOWN_i32_MASK    __builtin_epi_vslidedown_2xi32_mask //TODO, not being used
#define _MM_VSLIDEDOWN_i64_MASK    __builtin_epi_vslidedown_1xi64_mask //TODO, not being used

// fp
// log : Rename 'vsetfirst' and 'vgetfirst' builtins to 'vmv.s.x'/'vfmv.s.f' and 'vmv.x.s'/'vfmv.f.s'
//0.9 float32_t vfmv_f_s_f32m1_f32 (vfloat32m1_t src); 
//#define _MM_VGETFIRST_f32   __builtin_epi_vgetfirst_2xf32 
#define _MM_VGETFIRST_f32(op1, op2) vfmv_f_s_f32m1_f32(op1)

//#define _MM_VGETFIRST_f64   __builtin_epi_vgetfirst_1xf64
#define _MM_VGETFIRST_f64(op1, op2) vfmv_f_s_f64m1_f64(op1)

#define _MM_VSLIDEUP_f32    __builtin_epi_vslideup_2xf32 //TODO, not being used
#define _MM_VSLIDEUP_f64    __builtin_epi_vslideup_1xf64 //TODO, not being used

//#define _MM_VSLIDE1UP_f32    __builtin_epi_vslide1up_2xf32
#define _MM_VSLIDE1UP_f32(op1, op2, op3) vfslide1up_vf_f32m1(op1, op2, op3)

//#define _MM_VSLIDE1UP_f64    __builtin_epi_vslide1up_1xf64
#define _MM_VSLIDE1UP_f64(op1, op2, op3) vfslide1up_vf_f64m1(op1, op2, op3)

#define _MM_VSLIDEUP_f32_MASK    __builtin_epi_vslideup_2xf32_mask //TODO, not being used
#define _MM_VSLIDEUP_f64_MASK    __builtin_epi_vslideup_1xf64_mask //TODO, not being used

//#define _MM_VSLIDEDOWN_f32    __builtin_epi_vslidedown_2xf32
#define _MM_VSLIDEDOWN_f32(op1, op2, op3) vfslidedown_vf_f32m1(op1, op2, op3)

//#define _MM_VSLIDEDOWN_f64    __builtin_epi_vslidedown_1xf64
#define _MM_VSLIDEDOWN_f64(op1, op2, op3) vfslidedown_vf_f64m1(op1, op2, op3)

//#define _MM_VSLIDE1DOWN_f32    __builtin_epi_vslide1down_2xf32
#define _MM_VSLIDE1DOWN_f32(op1, op2, op3) vfslide1down_vf_f32m1(op1, op2, op3)

//#define _MM_VSLIDE1DOWN_f64    __builtin_epi_vslide1down_1xf64
#define _MM_VSLIDE1DOWN_f64(op1, op2, op3) vfslide1down_vf_f64m1(op1, op2, op3)

#define _MM_VSLIDEDOWN_f32_MASK    __builtin_epi_vslidedown_2xf32_mask //TODO, not being used
#define _MM_VSLIDEDOWN_f64_MASK    __builtin_epi_vslidedown_1xf64_mask //TODO, not being used
//---------------------------------------------------------------------------
// MASK DEFINITIONS

#define _MMR_MASK_i64   	vbool64_t //__epi_1xi1
#define _MMR_MASK_i32   	vbool32_t //__epi_2xi1
/*
data type -> mask type
__epi_1xi1 test_cast_1xi1_1xi64(__epi_1xi64 arg_0)
{
    return __builtin_epi_cast_1xi1_1xi64(arg_0);
}
trunc <vscale x 1 x i64> [[ARG_0:%.*]] to <vscale x 1 x i1>

https://github.com/riscv/rvv-intrinsic-doc/issues/37

*/
//#define _MM_CAST_i1_i64  	__builtin_epi_cast_1xi1_1xi64 
#define _MM_CAST_i1_i64(op1, op2) vmseq_vx_i64m1_b64(op1, 1, op2)

//#define _MM_CAST_i1_i32  	__builtin_epi_cast_2xi1_2xi32
#define _MM_CAST_i1_i32(op1, op2) vmseq_vx_i32m1_b32(op1, 1, op2)

// mask type -> data type
//#define _MM_CAST_i64_i1  	__builtin_epi_cast_1xi64_1xi1 
#define _MM_CAST_i64_i1(op1, op2) vmerge_vxm_i64m1(op1, vundefined_i64m1(), 1, op2)

//#define _MM_CAST_i32_i1  	__builtin_epi_cast_2xi32_2xi1
#define _MM_CAST_i32_i1(op1, op2) vmerge_vxm_i32m1(op1, vundefined_i32m1(), 1, op2)

// OPERATIONS WITH MASKS
 
//#define _MM_VMFIRST_i64 	__builtin_epi_vmfirst_1xi1 
#define _MM_VMFIRST_i64(op1, op2) vfirst_m_b64(op1, op2) //This function is not found in epi's testcases

//#define _MM_VMFIRST_i32 	__builtin_epi_vmfirst_2xi1
#define _MM_VMFIRST_i32(op1, op2) vfirst_m_b32(op1, op2) //This function is not found in epi's testcases

#if __clang__
    #define _MM_VMPOPC_i64(op1, op2) vcpop_m_b64(op1, op2)
    #define _MM_VMPOPC_i32(op1, op2) vcpop_m_b32(op1, op2)
#else
    //#define _MM_VMPOPC_i64 		__builtin_epi_vmpopc_1xi1
    #define _MM_VMPOPC_i64(op1, op2) vpopc_m_b64(op1, op2) //This function is not found in epi's testcases
    //#define _MM_VMPOPC_i32 		__builtin_epi_vmpopc_2xi1 
    #define _MM_VMPOPC_i32(op1, op2) vpopc_m_b32(op1, op2) //This function is not found in epi's testcases
#endif

//#define _MM_VMAND_i64 		__builtin_epi_vmand_1xi1
#define _MM_VMAND_i64(op1, op2, op3) vmand_mm_b64(op1, op2, op3)

//#define _MM_VMAND_i32 		__builtin_epi_vmand_2xi1 
#define _MM_VMAND_i32(op1, op2, op3) vmand_mm_b32(op1, op2, op3)
 
//#define _MM_VMNOR_i64 		__builtin_epi_vmnor_1xi1 
#define _MM_VMNOR_i64(op1, op2, op3) vmnor_mm_b64(op1, op2, op3)

//#define _MM_VMNOR_i32 		__builtin_epi_vmnor_2xi1 
#define _MM_VMNOR_i32(op1, op2, op3) vmnor_mm_b32(op1, op2, op3)

//#define _MM_VMOR_i64 		__builtin_epi_vmor_1xi1
#define _MM_VMOR_i64(op1, op2, op3) vmor_mm_b64(op1, op2, op3)

//#define _MM_VMOR_i32 		__builtin_epi_vmor_2xi1
#define _MM_VMOR_i32(op1, op2, op3) vmor_mm_b32(op1, op2, op3)

//#define _MM_VMXOR_i64 		__builtin_epi_vmxor_1xi1 
#define _MM_VMXOR_i64(op1, op2, op3) vmxor_mm_b64(op1, op2, op3)

//#define _MM_VMXOR_i32 		__builtin_epi_vmxor_2xi1
#define _MM_VMXOR_i32(op1, op2, op3) vmxor_mm_b32(op1, op2, op3)

// OPERATIONS TO CREATE A MASK

// Int

//#define _MM_VMSLT_i64     __builtin_epi_vmslt_1xi64
#define _MM_VMSLT_i64(op1, op2, op3) vmslt_vv_i64m1_b64(op1, op2, op3)

//#define _MM_VMSLT_i32     __builtin_epi_vmslt_2xi32
#define _MM_VMSLT_i32(op1, op2, op3) vmslt_vv_i32m1_b32(op1, op2, op3)

//#define _MM_VMSEQ_i64		__builtin_epi_vmseq_1xi64							
#define _MM_VMSEQ_i64(op1, op2, op3) vmseq_vv_i64m1_b64(op1, op2, op3)

//#define _MM_VMSEQ_i32		__builtin_epi_vmseq_2xi32
#define _MM_VMSEQ_i32(op1, op2, op3) vmseq_vv_i32m1_b32(op1, op2, op3)

// Fp

//#define _MM_VFGT_f64        __builtin_epi_vmfgt_1xf64
#define _MM_VFGT_f64(op1, op2, op3) vmfgt_vv_f64m1_b64(op1, op2, op3)

//#define _MM_VFGT_f32        __builtin_epi_vmfgt_2xf32
#define _MM_VFGT_f32(op1, op2, op3) vmfgt_vv_f32m1_b32(op1, op2, op3)

//#define _MM_VFGE_f64        __builtin_epi_vmfge_1xf64
#define _MM_VFGE_f64(op1, op2, op3) vmfge_vv_f64m1_b64(op1, op2, op3)

//#define _MM_VFGE_f32        __builtin_epi_vmfge_2xf32
#define _MM_VFGE_f32(op1, op2, op3) vmfge_vv_f32m1_b32(op1, op2, op3)

//#define _MM_VFLT_f64        __builtin_epi_vmflt_1xf64
#define _MM_VFLT_f64(op1, op2, op3) vmflt_vv_f64m1_b64(op1, op2, op3)

//#define _MM_VFLT_f32        __builtin_epi_vmflt_2xf32
#define _MM_VFLT_f32(op1, op2, op3) vmflt_vv_f32m1_b32(op1, op2, op3)

//#define _MM_VFLE_f64        __builtin_epi_vmfle_1xf64
#define _MM_VFLE_f64(op1, op2, op3) vmfle_vv_f64m1_b64(op1, op2, op3)

//#define _MM_VFLE_f32        __builtin_epi_vmfle_2xf32
#define _MM_VFLE_f32(op1, op2, op3) vmfle_vv_f32m1_b32(op1, op2, op3)
//---------------------------------------------------------------------------
// ADVANCE RISC-V MATH LIBRARY

#ifndef _MM_LOG
#define _MM_LOG
#include "__log.h"
#define _MM_LOG_f64 		__log_1xf64
#define _MM_LOG_f32 		__log_2xf32
#endif

#ifndef _MM_EXP
#define _MM_EXP
#include "__exp.h"
#define _MM_EXP_f64 		__exp_1xf64
#define _MM_EXP_f32 		__exp_2xf32
#endif

#ifndef _MM_COS
#define _MM_COS
#include "__cos.h"
#define _MM_COS_f64 		__cos_1xf64
#define _MM_COS_f32 		__cos_1xf32
#endif

//---------------------------------------------------------------------------

