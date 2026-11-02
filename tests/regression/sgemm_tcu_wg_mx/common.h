#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef WGMMA_NRC
#define WGMMA_NRC 8
#endif
#ifndef ITYPE
#define ITYPE mxfp8
#endif
#ifndef OTYPE
#define OTYPE fp32
#endif

// LDST_MATRIX=1 moves each tensor fragment with one vx_ldm/vx_stm where its
// shape has one; 0 keeps one flw/fsw per register.
#ifndef LDST_MATRIX
#define LDST_MATRIX 0
#endif

// K tiles per iteration of the K loop
#ifndef KUNROLL
#define KUNROLL 1
#endif

typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr, B_addr, C_addr;
  uint64_t MX_A_addr, MX_B_addr;
#ifdef TCU_MX_TLS
  uint64_t A_tensor_scale_addr, B_tensor_scale_addr;
#endif
} kernel_arg_t;

#endif
