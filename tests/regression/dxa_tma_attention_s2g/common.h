#ifndef DXA_TMA_ATTENTION_S2G_COMMON_H
#define DXA_TMA_ATTENTION_S2G_COMMON_H

#include <stdint.h>

#ifndef WGMMA_NRC
#define WGMMA_NRC 16
#endif

#ifndef ATT_K_TILES
#define ATT_K_TILES 2
#endif

#ifndef ATT_ITERS
#define ATT_ITERS 4
#endif

#ifndef ATT_STAGE_GLOBAL_STORE
#define ATT_STAGE_GLOBAL_STORE 0
#endif

#ifndef ATT_COMPUTE_WARPS
#define ATT_COMPUTE_WARPS 2
#endif

#ifndef ATT_OUTPUT_STAGES
#define ATT_OUTPUT_STAGES 2
#endif

// One CTA models a warp-specialized attention epilogue:
//   warps 0-1: QK score/WGMMA and epilogue
//   warp 2:    TMA loader
//   warp 3:    S2G issuer
//   warps 4-7: reserved CTA roles (kept resident to exercise the real warp map)
enum : uint32_t {
  ATT_WARPS = ATT_COMPUTE_WARPS + 4,
  ATT_LOAD_WARP = ATT_COMPUTE_WARPS,
  ATT_STORE_WARP = ATT_COMPUTE_WARPS + 1,
  ATT_OUT_STAGES = ATT_OUTPUT_STAGES,
};

typedef struct {
  uint32_t mode; // 0 = TMA+S2G, 1 = TMA+global store, 2 = LSU load+global store
  uint64_t q_addr;
  uint64_t k_addr;
  uint64_t v_addr;
  uint64_t out_addr;
  uint64_t stats_addr;
} kernel_arg_t;

#endif
