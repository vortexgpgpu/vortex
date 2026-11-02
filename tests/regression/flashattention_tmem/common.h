#ifndef _FLASHATTENTION_TMEM_COMMON_H_
#define _FLASHATTENTION_TMEM_COMMON_H_

#include <stdint.h>

#ifndef WGMMA_NRC
  #define WGMMA_NRC 32
#endif

#ifndef ITYPE
  #define ITYPE fp16
#endif

#ifndef OTYPE
  #define OTYPE fp32
#endif

// One head over NB KV blocks of L keys: Q is [M][D], K_T is [NB*L][D], V_T is [N][NB*L].
typedef struct {
  uint32_t L, N, D, NB;
  float scale;
  uint32_t pad;
  uint64_t O_addr;
} kernel_arg_t;

#endif
