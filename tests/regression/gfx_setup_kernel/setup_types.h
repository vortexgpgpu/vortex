#ifndef _SETUP_TYPES_H_
#define _SETUP_TYPES_H_

// gfx_v2 setup front end — shared data types (no kernel-ABI, no stdlib).
// Split out of common.h so setup_math.h is self-contained and reusable by
// other tests (e.g. the fused setup->binning pipeline) without dragging in a
// test-specific kernel_arg_t.

#include <stdint.h>

#define SETUP_W        512
#define SETUP_H        512
#define SETUP_NEAR     0.0f
#define SETUP_FAR      1.0f
#define SETUP_BIN_LOG  7      // 128px coarse bin (Binning() tileLogSize; bbox-independent)

// VS-output vertex in clip space. Byte-identical to graphics::vertex_t
// (sw/runtime/include/graphics.h) so the host can feed Binning() directly.
typedef struct {
  float pos[4];        // clip-space x, y, z, w
  float color[4];      // r, g, b, a in [0, 1]
  float texcoord[2];   // u, v
} setup_vertex_t;

// Per-prim screen bbox (pixels, clamped to the render target) — the bridge
// record on-device binning consumes (binsort_prim_t shape).
typedef struct {
  uint32_t bbL, bbR, bbT, bbB;
} setup_bbox_t;

// Near-plane clip (z_clip + w_clip >= 0) yields at most 2 subtriangles per
// input triangle (2-inside case -> quad -> fan of 2). One clipped (sub)triangle
// as 3 post-clip vertices.
#define SETUP_MAX_SUB 2
typedef struct {
  setup_vertex_t v[3];
} clip_tri_t;

#endif
