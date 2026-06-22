#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef VX_CFG_NUM_THREADS
#define VX_CFG_NUM_THREADS 4
#endif

#ifndef ITYPE
#define ITYPE mxfp8
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t MX_A_addr;
  uint64_t MX_B_addr;
#ifdef VX_CFG_TCU_MX_TLS
  uint64_t A_tensor_scale_addr;
  uint64_t B_tensor_scale_addr;
#endif
} kernel_arg_t;

#endif
