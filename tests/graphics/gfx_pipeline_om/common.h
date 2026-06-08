#ifndef _COMMON_H_
#define _COMMON_H_

// gfx_v2 device front end -> RASTER -> fragment interpolation -> OM, end to end.
//
// The fused setup+binning pipeline produces RASTER's tilebuf + primbuf (pinned,
// dense tile grid so the tile count is host-known); RASTER + the interpolate
// kernel + the OM fixed-function unit then turn that into shaded pixels with no
// host Binning() in the loop. The rendered colour image is checked against the
// gfx-v1 reference (gfx_draw3d), extending the device-side FF chain to OM.

#include <stdint.h>
#include <setup_types.h>

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

// Front-end pipeline args (dense tile grid sized to the render target).
typedef struct {
  uint32_t num_tris;
  uint32_t stage;
  uint32_t width, height;
  uint32_t bin_stripe;
  uint32_t bin_cols;
  uint32_t num_bins;
  uint32_t _pad;
  uint64_t verts_addr;
  uint64_t slot_prim_addr;
  uint64_t slot_bbox_addr;
  uint64_t keep_addr;
  uint64_t offset_addr;
  uint64_t tsum_addr;
  uint64_t prim_addr;      // dense primbuf (pinned: RASTER + the fragment kernel read it)
  uint64_t bbox_addr;
  uint64_t bcount_addr;
  uint64_t boffset_addr;
  uint64_t keys_addr;
  uint64_t btsum_addr;
  uint64_t thist_addr;
  uint64_t bincount_addr;
  uint64_t binbase_addr;
  uint64_t tilebuf_addr;   // dense tilebuf (pinned)
  uint64_t meta_addr;
} pipe_arg_t;

// Fragment kernel args (gfx_draw3d): interpolate from primbuf, write via vx_om.
typedef struct {
  uint64_t prim_addr;
  uint32_t depth_enabled;
  uint32_t color_enabled;
  uint32_t tex_enabled;
  uint32_t tex_modulate;
} frag_arg_t;

#endif
