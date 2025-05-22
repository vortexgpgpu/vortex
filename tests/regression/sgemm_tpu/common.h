#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <hfloats.h>

#ifndef I_TYPE
#define I_TYPE float
#endif

#ifndef O_TYPE
#define O_TYPE float
#endif

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t tileM, tileN, tileK;
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
