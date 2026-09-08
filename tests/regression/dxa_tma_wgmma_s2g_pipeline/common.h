#pragma once
#include <stdint.h>

#ifndef WGMMA_NRC
#define WGMMA_NRC 16
#endif
#ifndef TMA_PIPE_K_TILES
#define TMA_PIPE_K_TILES 2
#endif
#ifndef TMA_PIPE_ITERS
#define TMA_PIPE_ITERS 8
#endif
#ifndef TMA_PIPE_IN_STAGES
#define TMA_PIPE_IN_STAGES 3
#endif
#ifndef TMA_PIPE_OUT_STAGES
#define TMA_PIPE_OUT_STAGES 3
#endif
#ifndef TMA_PIPE_STATE_STAGES
#define TMA_PIPE_STATE_STAGES 3
#endif
#ifndef TMA_PIPE_META_WORDS
#define TMA_PIPE_META_WORDS 8
#endif

static_assert(TMA_PIPE_IN_STAGES >= 2, "input pipeline needs at least two stages");
static_assert(TMA_PIPE_OUT_STAGES >= 2 && TMA_PIPE_OUT_STAGES <= 8, "output stage count must fit the group ring");
static_assert(TMA_PIPE_STATE_STAGES >= 2 && TMA_PIPE_STATE_STAGES <= 8, "state stage count must fit the group ring");

enum : uint32_t {
  TMA_PIPE_WARPS = 8, TMA_PIPE_COMPUTE_WARPS = 2, TMA_PIPE_LOAD_WARP = 2,
  TMA_PIPE_STORE_WARP = 3, TMA_PIPE_STATE_WARP = 4,
  TMA_PIPE_CANARY_WORDS = 16, TMA_PIPE_CANARY = 0xbaadf00d,
  TMA_PIPE_TMA_S2G = 0, TMA_PIPE_TMA_GLOBAL = 1, TMA_PIPE_LSU_GLOBAL = 2, TMA_PIPE_SERIAL_S2G = 3,
};

struct kernel_arg_t {
  uint32_t mode;
  uint64_t a_addr, b_addr, out_addr, bias_addr, stats_addr, state_addr, meta_addr, canary_addr;
};
