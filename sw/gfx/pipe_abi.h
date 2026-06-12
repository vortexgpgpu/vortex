#ifndef _PIPE_ABI_H_
#define _PIPE_ABI_H_

// gfx_v2 fused setup -> binning front end — test-side bin-grid macros.
//
// The shared host/device ABI (pipe_arg_t, PIPE_STAGE_*, PIPE_PRIM_*,
// PIPE_BIN_LOG) now lives in <gfx_frontend_abi.h> (sw/common). This header
// keeps only the compile-time bin-grid macros derived from the test-oracle
// dimensions (SETUP_W/H), which are NOT part of the ABI — real draws size the
// grid at runtime from pipe_arg_t.width/height.

#include <stdint.h>
#include <gfx_frontend_abi.h>   // pipe_arg_t, PIPE_STAGE_*, PIPE_PRIM_*
#include "setup_types.h"        // SETUP_W / SETUP_H (test oracle dimensions)

// gfx_v2 §6.3: bins are coarse 128 px regions (VX_CFG_RASTER_BIN_LOGSIZE); the
// RASTER front end descends bin -> block -> quad. PIPE_BIN_LOG is log2 of the
// bin edge in pixels. Coarser than the RASTER 32 px tile: far fewer coverage
// entries, identical pixel coverage.
#ifndef VX_CFG_RASTER_BIN_LOGSIZE
#define VX_CFG_RASTER_BIN_LOGSIZE 7
#endif
#define PIPE_BIN_LOG   VX_CFG_RASTER_BIN_LOGSIZE

#define PIPE_BIN_COLS  ((SETUP_W + (1 << PIPE_BIN_LOG) - 1) >> PIPE_BIN_LOG)
#define PIPE_BIN_ROWS  ((SETUP_H + (1 << PIPE_BIN_LOG) - 1) >> PIPE_BIN_LOG)
#define PIPE_NUM_BINS  (PIPE_BIN_COLS * PIPE_BIN_ROWS)

#endif
