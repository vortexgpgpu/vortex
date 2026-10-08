#ifndef _SGEMM_TCU_WG_SP_DXA_COMMON_H_
#define _SGEMM_TCU_WG_SP_DXA_COMMON_H_

#include <stdint.h>

#ifndef WGMMA_NRC
  #define WGMMA_NRC 8
#endif

// Stage count for the DEEP_BUF N-stage software-pipelined DXA kernel
// (kernel.cpp). Unused unless DEEP_BUF is defined. Matches
// sgemm_tcu_wg_dxa_deepbuf/common.h's default.
#ifndef PIPE_N
  #define PIPE_N 4
#endif

// K-tiles per DEEP_BUF pipeline stage (kernel.cpp: KB). 1 = one K-tile per
// stage (original behavior).
#ifndef KB
  #define KB 1
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr;    // compressed A in global memory (M x K/2 elements)
  uint64_t B_addr;    // dense B in global memory (K x N elements)
  uint64_t C_addr;    // output C in global memory (M x N elements)
  uint64_t meta_sp_addr; // smem-format sparse metadata per tile
} kernel_arg_t;

#endif
