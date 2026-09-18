#ifndef _SGEMM_TCU_WG_DXA_COMMON_H_
#define _SGEMM_TCU_WG_DXA_COMMON_H_

#include <stdint.h>

#ifndef WGMMA_NRC
  #define WGMMA_NRC 8
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// Software-pipeline depth for DEEP_BUF (N-stage generalization of
// DOUBLE_BUF). Shared between kernel.cpp (loop structure) and main.cpp
// (SMEM sizing) so they can't drift out of sync.
#ifndef PIPE_N
#define PIPE_N 4
#endif

typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
