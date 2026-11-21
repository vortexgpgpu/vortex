#ifndef _FLASHATTENTION4_TMEM_COMMON_H_
#define _FLASHATTENTION4_TMEM_COMMON_H_

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

// Shared-memory pipeline stages.
#ifndef PIPE_N
  #define PIPE_N 2
#endif

// K-tiles per pipeline stage, one tc.mma each.
#ifndef KB
  #define KB 4
#endif

// Control warps per CTA: a producer warp stages the operands and an MMA warp
// issues tc.mma and tc.commit.
#define CTRL_WARPS 2

// One head over NB KV blocks of L keys: Q is [M][D], K_T is [NB*L][D], V_T is [N][NB*L].
// Shared memory starts with the CTA's TMEM column, padded to keep the stages aligned.
#define TMEM_SLOT_BYTES 64

typedef struct {
  uint32_t L, N, D, NB;
  float scale;
  uint32_t pad;
  uint64_t O_addr;
} kernel_arg_t;

#endif
