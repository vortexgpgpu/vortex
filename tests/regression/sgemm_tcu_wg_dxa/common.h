#ifndef _SGEMM_TCU_WG_DXA_COMMON_H_
#define _SGEMM_TCU_WG_DXA_COMMON_H_

#include <stdint.h>

#ifndef WGMMA_NRC
  #define WGMMA_NRC 8
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// Bytes of local memory left unused below the tile buffers, so the tiles and
// their shared-memory descriptors can sit above a given LMEM offset.
#ifndef LMEM_PAD
#define LMEM_PAD 0
#endif

// K tiles per iteration of the K loop
#ifndef KUNROLL
#define KUNROLL 1
#endif

typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
