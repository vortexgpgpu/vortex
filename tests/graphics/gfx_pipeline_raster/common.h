#ifndef _COMMON_H_
#define _COMMON_H_

// gfx_v2 on-device front end -> RASTER fixed-function unit, end to end.
//
// The fused setup+binning pipeline (gfx_pipeline_kernel) runs on the SIMT cores
// to produce RASTER's exact gfx-v1 buffers (dense rast_prim_t primbuf + the
// rast_tile_header_t tilebuf) into pinned memory; those buffers are then bound
// to the RASTER unit via its DCRs and a trivial fragment kernel writes the
// covered pixels. The rendered image is checked against the gfx-v1 reference —
// proving the device front end drives the FF unit to a pixel-correct result,
// with no host Binning() in the loop.
//
// Binning granularity must equal the RASTER tile size, so the bins ARE RASTER's
// tiles (PIPE_BIN_LOG = VX_CFG_RASTER_TILE_LOGSIZE).

#include <stdint.h>
#include <setup_types.h>   // setup_vertex_t, setup_bbox_t, clip_tri_t, SETUP_*

#ifndef VX_CFG_RASTER_TILE_LOGSIZE
#define VX_CFG_RASTER_TILE_LOGSIZE 5
#endif

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

// Front-end pipeline args (see gfx_pipeline_kernel). Output prim/tilebuf live in
// pinned memory so the RASTER unit's AXI master can read them.
typedef struct {
  uint32_t num_tris;
  uint32_t stage;
  uint32_t width, height;
  uint32_t bin_stripe;
  uint32_t bin_cols;      // tiles across the render target (ceil(width / tilesize))
  uint32_t num_bins;      // bin_cols * bin_rows — the dense tile grid count
  uint32_t _pad;
  uint64_t verts_addr;     // setup_vertex_t[3*num_tris]      (in)
  uint64_t slot_prim_addr; // rast_prim_t[num_tris*MAX_SUB]   (scratch)
  uint64_t slot_bbox_addr; // setup_bbox_t[num_tris*MAX_SUB]  (scratch)
  uint64_t keep_addr;      // uint32[num_tris]                (scratch)
  uint64_t offset_addr;    // uint32[num_tris + 1]            (scratch)
  uint64_t tsum_addr;      // uint32[T]                       (scratch)
  uint64_t prim_addr;      // rast_prim_t[num_tris*MAX_SUB]   (out: primbuf, pinned)
  uint64_t bbox_addr;      // setup_bbox_t[num_tris*MAX_SUB]  (dense: setup out / binning in)
  uint64_t bcount_addr;    // uint32[P]                       (scratch)
  uint64_t boffset_addr;   // uint32[P + 1]                   (scratch)
  uint64_t keys_addr;      // uint32[keys]                    (scratch)
  uint64_t btsum_addr;     // uint32[T]                       (scratch)
  uint64_t thist_addr;     // uint32[T * NUM_BINS]            (scratch)
  uint64_t bincount_addr;  // uint32[NUM_BINS]                (scratch)
  uint64_t binbase_addr;   // uint32[NUM_BINS]                (scratch)
  uint64_t tilebuf_addr;   // rast_tile_header_t[nb] + uint32 pids[keys]  (out: tilebuf, pinned)
  uint64_t meta_addr;      // uint32[3] = { P, keys, nbins }  (out)
} pipe_arg_t;

// Fragment kernel args (writes covered pixels; same shape as gfx_raster).
typedef struct {
  uint32_t dst_width;
  uint32_t dst_height;
  uint64_t cbuf_addr;
  uint8_t  cbuf_stride;
  uint32_t cbuf_pitch;
  uint64_t prim_addr;
} frag_arg_t;

#endif
