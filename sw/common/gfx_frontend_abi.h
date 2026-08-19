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

#pragma once

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

// Viewport transform (screen = ndc*scale + bias), captured from the app's bound
// VkViewport and applied by triangle setup (ClipToScreen / ClipToHDC). The
// device otherwise hardwires a full-framebuffer y-DOWN viewport, which ignores
// a negative-height (y-flip) Vulkan viewport and so mirrors the screen-space
// triangle — flipping the signed-area sign the face cull reads. Carrying the
// app transform makes both the rendered orientation AND the cull decision agree
// with the app's framebuffer winding. A negative sy is the common Vulkan y-flip.
// minz/maxz are the app's depth range (VkViewport minDepth/maxDepth), and halfz
// selects the clip-space Z convention the vertices arrive in: 0 = OpenGL, where
// clip z spans [-w, w] and the near plane is z + w >= 0; 1 = Vulkan/D3D, where
// it spans [0, w] and the near plane is z >= 0. The two need different near-plane
// clips AND different screen mappings, so a stream fed under one convention and
// mapped under the other lands at the wrong absolute depth while keeping its
// ordering -- which is why only a test against a known depth clear catches it.
typedef struct {
  float sx, tx;   // screen_x = ndc_x*sx + tx
  float sy, ty;   // screen_y = ndc_y*sy + ty
  float minz, maxz;   // screen_z range (depth range)
  uint32_t halfz;     // 0 = GL [-w,w] clip z, 1 = Vulkan [0,w]
} setup_viewport_t;

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
// The six scalar `varying2` planes extend the colour+texcoord six with room for
// wider varyings (e.g. samplerCube textureGrad's coord + dPdx + dPdy = 9), mapped
// past [u,v,r,g,b,a] onto the setup w0..w5 planes; a draw carrying only a colour
// and a texcoord leaves them 0 (bit-identical to the six-plane front end).
typedef struct {
  float pos[4];        // clip-space x, y, z, w
  float color[4];      // r, g, b, a in [0, 1]
  float texcoord[2];   // u, v
  float varying2[6];   // extra varying scalars -> setup planes w0..w5
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
// generic varyings (16 bytes each). expand_k packs the varyings into the 6
// scalar interpolation planes [u,v,r,g,b,a] in declaration order — each varying
// claims the next nc lanes — and the FS translator reads them back the same way,
// so a draw may carry any mix of varyings without two colliding on one plane.
// One thread/vertex.
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
  // App viewport transform (screen = ndc*scale + bias). All four zero => unset:
  // the front end derives the default full-framebuffer y-down transform from
  // width/height (the gfx-v1 default, and what the standalone setup tests —
  // which zero-init this block — expect). A negative vp_sy carries the Vulkan
  // y-flip so cull + orientation match the app's framebuffer winding.
  float vp_sx, vp_tx;     // screen_x = ndc_x*vp_sx + vp_tx
  float vp_sy, vp_ty;     // screen_y = ndc_y*vp_sy + vp_ty
  float vp_minz, vp_maxz; // depth range; both zero => SETUP_NEAR..SETUP_FAR
  uint32_t vp_halfz;      // clip-z convention, see setup_viewport_t
  uint32_t vp_pad_;       // keep the uint64 block below 8-byte aligned
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
  // Flat varyings: the provoking vertex's varying words per emitted primitive,
  // GFX_FS_FLAT_WORDS of them, copied verbatim with no interpolation applied.
  // Zero when no bound shader declares a flat input, and the EMIT stage then
  // skips the copy entirely. See gfx_fs_desc_abi.h for why these travel beside
  // the interpolation planes rather than through them.
  uint64_t flat_addr;      // uint32[GFX_FS_FLAT_WORDS * P]   (out)
} pipe_arg_t;
