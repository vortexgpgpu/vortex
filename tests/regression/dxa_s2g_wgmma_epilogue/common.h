#ifndef DXA_S2G_WGMMA_EPILOGUE_COMMON_H
#define DXA_S2G_WGMMA_EPILOGUE_COMMON_H

#include <stdint.h>

#ifndef WGMMA_NRC
#define WGMMA_NRC 16
#endif
#if WGMMA_NRC != 8 && WGMMA_NRC != 16
#error "This RS epilogue test supports WGMMA_NRC=8 or 16"
#endif
#ifndef S2G_TE_ITERS
#define S2G_TE_ITERS 8
#endif
#ifndef S2G_TE_K_TILES
#define S2G_TE_K_TILES 8
#endif
#ifndef S2G_TE_STATS_X
#define S2G_TE_STATS_X 32
#endif
#ifndef S2G_TE_STATS_Y
#define S2G_TE_STATS_Y 16
#endif
#ifndef S2G_TE_STATS_Z
#define S2G_TE_STATS_Z 2
#endif
#ifndef S2G_TE_STATS_STAGES
#define S2G_TE_STATS_STAGES 3
#endif
#if S2G_TE_STATS_STAGES < 3
#error "The pipelined statistics stream requires at least three SMEM stages"
#endif

enum : uint32_t {
  S2G_TE_WARPS = 8,
  S2G_TE_COMPUTE_WARPS = 2,
};

typedef struct {
  uint32_t mode;
  uint64_t a_addr;
  uint64_t b_addr;
} kernel_arg_t;

#endif
