#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// LDST_WIDE=1 reads and writes each thread's consecutive words with the wide
// loads and stores (vx_flw_d/q, vx_fsw_d/q); 0 keeps scalar accesses.
#ifndef LDST_WIDE
#define LDST_WIDE 0
#endif

// Per-thread output tile
#ifndef THREAD_SIZE_Y
#define THREAD_SIZE_Y 4
#endif
#ifndef THREAD_SIZE_X
#define THREAD_SIZE_X 4
#endif

// CTA block dimensions (in threads).
#ifndef BLOCK_DIM_X
#define BLOCK_DIM_X 4
#endif
#ifndef BLOCK_DIM_Y
#define BLOCK_DIM_Y 4
#endif

// Shared-memory tile depth along K.
#ifndef BLOCK_SIZE_K
#define BLOCK_SIZE_K 4
#endif

typedef struct {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  float    alpha;
  float    beta;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
