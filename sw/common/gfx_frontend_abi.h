// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// gfx_v2 on-device setup -> binning front end — canonical host/device ABI.
//
// The front end runs as nine CP-sequenced launches across two kernel entries
// (setup_k stages 0-2, binning_k stages 3-8) over device-resident memory, to
// produce RASTER's gfx-v1 buffers: a dense rast_prim_t primbuf and a dense
// rast_tile_header_t tilebuf over a tile grid sized to the render target (so
// RASTER's tile count is host-known — no num_tiles readback). This header is
// the shared contract between the host (which builds pipe_arg_t and the CP
// command sequence) and the device kernels (which read it); it lives in
// sw/common/ alongside vx_gfx_abi.h so the runtime, simx, the kernels, and
// downstream drivers (vortexpipe) reference one source of truth.
//
// Render-target dimensions are runtime values carried in pipe_arg_t; the
// host-oracle dimension defaults (SETUP_W/H) and the compile-time bin-grid
// macros are NOT part of this ABI and stay test-side.

#ifndef _GFX_FRONTEND_ABI_H_
#define _GFX_FRONTEND_ABI_H_

#include <stdint.h>

// Near-plane clip + triangle-setup depth range. The device clip stage uses
// these directly (see pipe_clip_and_setup), so they are part of the contract.
#ifndef SETUP_NEAR
#define SETUP_NEAR     0.0f
#endif
#ifndef SETUP_FAR
#define SETUP_FAR      1.0f
#endif

// Composite bin-sort key layout: high bits = bin index, low PIPE_PRIM_BITS =
// primitive id. Bounds the in-flight primitive count (after clip) at 2^20.
#define PIPE_PRIM_BITS 20
#define PIPE_PRIM_MASK ((1u << PIPE_PRIM_BITS) - 1)

// Face-cull mode (applied in triangle setup on the signed-area sign). NONE
// keeps both windings (two-sided — the gfx-v1 default); BACK culls the
// negative-area winding, FRONT the positive. Kept triangles still have their
// edge equations normalised so the RASTER interior test stays positive.
#define SETUP_CULL_NONE  0
#define SETUP_CULL_BACK  1
#define SETUP_CULL_FRONT 2

// The nine front-end stages, in CP-launch order. Stages 0-2 run on setup_k,
// stages 3-8 on binning_k (the split point is PIPE_STAGE_BCOUNT).
#define PIPE_STAGE_SETUP    0
#define PIPE_STAGE_SCAN     1
#define PIPE_STAGE_EMIT     2
#define PIPE_STAGE_BCOUNT   3
#define PIPE_STAGE_BSCAN    4
#define PIPE_STAGE_BEMIT    5
#define PIPE_STAGE_BHIST    6
#define PIPE_STAGE_BBASE    7
#define PIPE_STAGE_BSCATTER 8

// VS-output vertex in clip space. Byte-identical to graphics::vertex_t
// (sw/runtime/include/graphics.h) so the host can feed Binning() directly.
typedef struct {
  float pos[4];        // clip-space x, y, z, w
  float color[4];      // r, g, b, a in [0, 1]
  float texcoord[2];   // u, v
} setup_vertex_t;

// Per-prim screen bbox (pixels, clamped to the render target) — the bridge
// record on-device binning consumes.
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

// On-device vertex assembly: expand_k turns the resident VS-output records
// into the setup_vertex_t[] the front end consumes, so the VS output never
// round-trips to the host. Record slot 0 is the clip-space POS; slots 1.. are
// generic varyings (16 bytes each). The gfx-v1 attribute routing maps a
// 2-component varying to texcoord and a 3/4-component varying to colour (alpha
// defaults to 1) — the same mapping the FS translator uses. One thread/vertex.
#define EXPAND_MAX_VARYINGS 16   // >= VP_VS_MAX_VARYINGS (vortexpipe layout)
typedef struct {
  uint64_t vsrec_addr;    // VS output records[num_verts], vstride bytes each (in)
  uint64_t verts_addr;    // setup_vertex_t[num_verts]                       (out)
  uint32_t num_verts;
  uint32_t vstride;       // bytes per VS output record
  uint32_t num_varyings;  // generic varyings after POS
  uint32_t varying_comps[EXPAND_MAX_VARYINGS];  // component count per varying
} expand_arg_t;

// Per-launch front-end argument block. Addresses are device byte addresses;
// counts/config are runtime values the kernels read (so launch dims and pool
// addresses can stay static while the work scales with the in-memory counts).
typedef struct {
  uint32_t num_tris;
  uint32_t stage;
  uint32_t width, height;
  uint32_t bin_stripe;    // bins per CTA (contiguous) for HIST/SCATTER
  uint32_t bin_cols;      // tiles across the render target (ceil(width / tilesize))
  uint32_t num_bins;      // bin_cols * bin_rows — the dense tile grid count
  uint32_t cull_mode;     // SETUP_CULL_* (0 = none / two-sided)
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

#endif // _GFX_FRONTEND_ABI_H_
