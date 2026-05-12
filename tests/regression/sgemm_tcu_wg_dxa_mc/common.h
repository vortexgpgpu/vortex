#ifndef _SGEMM_TCU_WG_DXA_MC_COMMON_H_
#define _SGEMM_TCU_WG_DXA_MC_COMMON_H_

#include <stdint.h>
#include <VX_config.h>

#ifndef WGMMA_NRC
  #define WGMMA_NRC 8
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// Inter-core multicast variant. `mc_group_size` (= NUM_CORES) CTAs on distinct
// cores share the same B tile via global-barrier-routed multicast.
typedef struct {
  uint32_t M, N, K;
  uint32_t mc_group_size;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
