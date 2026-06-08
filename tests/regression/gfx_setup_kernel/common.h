#ifndef _COMMON_H_
#define _COMMON_H_

// gfx_v2 on-device triangle setup — shared host/device ABI.
//
// Tests the setup front end (charter §6.1 / gfx_v2_vertex_setup_pipeline.md
// stages A–D, no clipping yet): VS output (resident vertex_t) -> per-triangle
// edge equations + attribute deltas + screen bbox, emitted as a dense
// rast_prim_t[] + bbox[] — exactly binning stage 1's input. The device output
// is checked bit-for-bit against the host Binning() oracle.
//
// Three CP-sequenced launches (the launch-drain is the device barrier):
//   SETUP  multi-CTA, 1 thread/tri: full setup -> per-tri slot prim+bbox+keep
//   SCAN   single-CTA: prefix-sum keep[] -> offset[], total kept P
//   EMIT   multi-CTA, 1 thread/tri: compact kept slots -> dense prim[]+bbox[]
//
// Baseline = non-indexed triangle list (assembly is i={3t,3t+1,3t+2}); no
// near-plane clip (all inputs generated in front of the near plane so Binning()
// stays a valid oracle — clipping is the next increment).

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

#define SETUP_STAGE_SETUP 0   // multi-CTA: per-tri full setup -> slot+keep
#define SETUP_STAGE_SCAN  1   // single-CTA: prefix-sum keep -> offset, P
#define SETUP_STAGE_EMIT  2   // multi-CTA: compact kept slots -> dense out

typedef struct {
  uint32_t num_prims;     // input triangle count
  uint32_t stage;
  uint32_t width, height;
  uint64_t verts_addr;      // setup_vertex_t[3*num_prims]  (in: triangle list)
  uint64_t slot_prim_addr;  // rast_prim_t[num_prims]       (scratch)
  uint64_t slot_bbox_addr;  // setup_bbox_t[num_prims]      (scratch)
  uint64_t keep_addr;       // uint32[num_prims]            (scratch: 0/1)
  uint64_t offset_addr;     // uint32[num_prims + 1]        (scratch)
  uint64_t tsum_addr;       // uint32[T]                    (scratch: block scan)
  uint64_t prim_addr;       // rast_prim_t[num_prims]       (out: dense)
  uint64_t bbox_addr;       // setup_bbox_t[num_prims]      (out: dense)
  uint64_t meta_addr;       // uint32[1] = { P }            (out)
} kernel_arg_t;

#endif
