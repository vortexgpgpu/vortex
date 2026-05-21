#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <VX_config.h>

#ifndef VX_CFG_NUM_THREADS
#define VX_CFG_NUM_THREADS 4
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef struct {
  uint32_t M, N, K;
  uint32_t cta_M;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
