#ifndef _PIPE_ABI_H_
#define _PIPE_ABI_H_

// gfx_v2 fused setup -> binning front end — shared host/device ABI.
//
// The pipeline runs nine CP-sequenced launches across two kernel entries
// (setup_k stages 0-2, binning_k stages 3-8; see pipe_frontend.h) to produce
// RASTER's gfx-v1 buffers — a dense rast_prim_t primbuf and a dense
// rast_tile_header_t tilebuf over a tile grid sized to the render target, so
// RASTER's tile count is host-known (no num_tiles readback). Reused by the
// FF-feeding tests (gfx_pipeline_raster, gfx_pipeline_om, ...).

#include <stdint.h>
#include "setup_types.h"   // setup_vertex_t, setup_bbox_t, clip_tri_t, SETUP_*

#ifndef VX_CFG_RASTER_TILE_LOGSIZE
#define VX_CFG_RASTER_TILE_LOGSIZE 5
#endif

// Bins == RASTER tiles, so the binning granularity matches the unit consuming it.
#define PIPE_BIN_LOG   VX_CFG_RASTER_TILE_LOGSIZE
#define PIPE_BIN_COLS  ((SETUP_W + (1 << PIPE_BIN_LOG) - 1) >> PIPE_BIN_LOG)
#define PIPE_BIN_ROWS  ((SETUP_H + (1 << PIPE_BIN_LOG) - 1) >> PIPE_BIN_LOG)
#define PIPE_NUM_BINS  (PIPE_BIN_COLS * PIPE_BIN_ROWS)
#define PIPE_PRIM_BITS 20
#define PIPE_PRIM_MASK ((1u << PIPE_PRIM_BITS) - 1)

#define PIPE_STAGE_SETUP    0
#define PIPE_STAGE_SCAN     1
#define PIPE_STAGE_EMIT     2
#define PIPE_STAGE_BCOUNT   3
#define PIPE_STAGE_BSCAN    4
#define PIPE_STAGE_BEMIT    5
#define PIPE_STAGE_BHIST    6
#define PIPE_STAGE_BBASE    7
#define PIPE_STAGE_BSCATTER 8

typedef struct {
  uint32_t num_tris;
  uint32_t stage;
  uint32_t width, height;
  uint32_t bin_stripe;    // bins per CTA (contiguous) for HIST/SCATTER
  uint32_t bin_cols;      // tiles across the render target (ceil(width / tilesize))
  uint32_t num_bins;      // bin_cols * bin_rows — the dense tile grid count
  uint32_t _pad;
  uint64_t verts_addr;     // setup_vertex_t[3*num_tris]      (in)
  uint64_t slot_prim_addr; // rast_prim_t[num_tris*MAX_SUB]   (scratch)
  uint64_t slot_bbox_addr; // setup_bbox_t[num_tris*MAX_SUB]  (scratch)
  uint64_t keep_addr;      // uint32[num_tris]                (scratch)
  uint64_t offset_addr;    // uint32[num_tris + 1]            (scratch)
  uint64_t tsum_addr;      // uint32[T]                       (scratch)
  uint64_t prim_addr;      // rast_prim_t[num_tris*MAX_SUB]   (out: dense primbuf, pinned)
  uint64_t bbox_addr;      // setup_bbox_t[num_tris*MAX_SUB]  (dense: setup out / binning in)
  uint64_t bcount_addr;    // uint32[P]                       (scratch)
  uint64_t boffset_addr;   // uint32[P + 1]                   (scratch)
  uint64_t keys_addr;      // uint32[keys]                    (scratch)
  uint64_t btsum_addr;     // uint32[T]                       (scratch)
  uint64_t thist_addr;     // uint32[T * num_bins]            (scratch)
  uint64_t bincount_addr;  // uint32[num_bins]                (scratch)
  uint64_t binbase_addr;   // uint32[num_bins]                (scratch)
  uint64_t tilebuf_addr;   // rast_tile_header_t[num_bins] then uint32 pids[keys]  (out, pinned)
  uint64_t meta_addr;      // uint32[3] = { P, keys, nb }     (out)
} pipe_arg_t;

#endif
