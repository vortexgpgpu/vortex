#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

#ifndef CTA_SIZE
#define CTA_SIZE NUM_THREADS
#endif

typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr;    // compressed A in global memory (M × K/2 elements)
  uint64_t B_addr;    // dense B in global memory (K × N elements)
  uint64_t C_addr;    // output C in global memory (M × N elements)
  uint64_t meta_sp_addr; // smem-format sparse metadata per tile
} kernel_arg_t;

#endif
