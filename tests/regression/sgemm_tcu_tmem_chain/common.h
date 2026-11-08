#ifndef _SGEMM_TCU_TMEM_CHAIN_COMMON_H_
#define _SGEMM_TCU_TMEM_CHAIN_COMMON_H_

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

// K-tiles per pipeline stage, one tc.mma each.
#ifndef KB
  #define KB 4
#endif

// S = Q . K^T and O = P . V^T, with P the fp16 conversion of S held in TMEM:
// Q is [M][K1], K_T is [L][K1], V_T is [N][L] and L is S's N and P's K.
// Shared memory starts with the CTA's TMEM column, padded to keep the stages aligned.
#define TMEM_SLOT_BYTES 64

typedef struct {
  uint32_t M, L, N, K1;
  uint64_t Q_addr;
  uint64_t KT_addr;
  uint64_t VT_addr;
  uint64_t S_addr;
  uint64_t O_addr;
} kernel_arg_t;

#endif
