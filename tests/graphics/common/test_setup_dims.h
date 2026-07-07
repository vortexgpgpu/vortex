#pragma once

// Test-oracle render-target dimensions + derived bin grid for the gfx setup/
// binning tests. These are NOT part of the graphics ABI — real draws carry
// width/height in pipe_arg_t and size the bin grid at runtime. The shared ABI
// (setup_vertex_t, pipe_arg_t, PIPE_STAGE_*, PIPE_PRIM_*, SETUP_*) comes from
// <gfx_frontend_abi.h>. (Was sw/gfx/setup_types.h + sw/gfx/pipe_abi.h, which
// mixed these test fixtures into the device stack.)

#include <gfx_frontend_abi.h>

#define SETUP_W        512
#define SETUP_H        512
#define SETUP_BIN_LOG  7      // 128px coarse bin (Binning() tileLogSize)

// Coarse bin granularity (= the device front end's VX_CFG_RASTER_BIN_LOG_SIZE);
// guarded so it coexists with the device header that also defines it.
#ifndef VX_CFG_RASTER_BIN_LOG_SIZE
#define VX_CFG_RASTER_BIN_LOG_SIZE 7
#endif
#ifndef PIPE_BIN_LOG
#define PIPE_BIN_LOG   VX_CFG_RASTER_BIN_LOG_SIZE
#endif

// Compile-time test bin grid derived from the oracle dimensions.
#define PIPE_BIN_COLS  ((SETUP_W + (1 << PIPE_BIN_LOG) - 1) >> PIPE_BIN_LOG)
#define PIPE_BIN_ROWS  ((SETUP_H + (1 << PIPE_BIN_LOG) - 1) >> PIPE_BIN_LOG)
#define PIPE_NUM_BINS  (PIPE_BIN_COLS * PIPE_BIN_ROWS)
