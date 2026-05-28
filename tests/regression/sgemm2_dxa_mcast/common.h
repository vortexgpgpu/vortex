#ifndef _SGEMM2_DXA_MW_COMMON_H_
#define _SGEMM2_DXA_MW_COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

// Intra-core multicast (mw = "multicast warp-group", multiple CTAs on one core).
// `mc_group_size` CTAs co-resident on one core share the same B column block.
// Each CTA computes one row of the output; A is per-CTA, B is multicast.
typedef struct {
  uint32_t size;
  uint32_t tile_size;     // columns per CTA's output (B column block width)
  uint32_t chunk_k;       // K dimension chunk (full K for single-buffer mode)
  uint32_t mc_group_size; // CTAs sharing one B tile (= co-resident CTAs/core)
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
