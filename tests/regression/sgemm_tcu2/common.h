#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

#ifndef TCU_WGMMA_ENABLE
#define TCU_WGMMA_ENABLE
#endif

typedef struct {
  uint32_t M, N, K;
  uint32_t mode;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t A_desc;
  uint64_t B_desc;
} kernel_arg_t;

#endif
