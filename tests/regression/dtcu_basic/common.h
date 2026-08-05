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

#include <dtcu_cfg.h>   // DTCU_ENGINE_{SOCKET,CLUSTER}

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t M, N, K;
  // DTCU_ENGINE_SOCKET or DTCU_ENGINE_CLUSTER. Selects WHICH start instruction the
  // kernel issues. The descriptor itself is identical for both, which is the point of
  // putting the choice in the opcode rather than in a descriptor field.
  uint32_t engine;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t D_addr;
  uint64_t desc_addr;
} kernel_arg_t;

#endif
