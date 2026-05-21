#ifndef _SGEMM2_DXA_MC_COMMON_H_
#define _SGEMM2_DXA_MC_COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

// Inter-core multicast (mc = "multicast cluster", one CTA per core).
// `mc_group_size` CTAs on distinct cores share the same B column block.
// Selected by the global-barrier scope of the multicast handle.
typedef struct {
  uint32_t size;
  uint32_t tile_size;     // tile_size × tile_size output per CTA
  uint32_t chunk_k;       // K-chunk (full K for single-buffer mode)
  uint32_t mc_group_size; // = VX_CFG_NUM_CORES participating in multicast
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
