#ifndef _SGEMM_TCU_WG_DXA_MW_COMMON_H_
#define _SGEMM_TCU_WG_DXA_MW_COMMON_H_

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

// Intra-core multicast variant of sgemm_tcu_wg_dxa. `mc_group_size` CTAs
// co-resident on one core share the same B tile via DXA multicast.
typedef struct {
  uint32_t M, N, K;
  uint32_t mc_group_size;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
