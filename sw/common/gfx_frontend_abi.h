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

// Face-cull mode (applied in triangle setup). NONE keeps both windings
// (two-sided — the gfx-v1 default); BACK culls the back-facing winding, FRONT the
// front-facing one, as SETUP_FRONT_* defines them. Kept triangles still have
// their edge equations normalised so the RASTER interior test stays positive.
#define SETUP_CULL_NONE  0
#define SETUP_CULL_BACK  1
#define SETUP_CULL_FRONT 2
#define SETUP_CULL_ALL   3   // drops every primitive (cull both faces, rasterizer discard)

// Which winding is front-facing, by the sign of the triangle's area in y-down
// screen space. Facing drives the cull, gl_FrontFacing and the two-sided stencil
// state alike. Vulkan's area carries a leading minus sign, so its counter-
// clockwise front face is SETUP_FRONT_NEGATIVE.
#define SETUP_FRONT_POSITIVE 0
#define SETUP_FRONT_NEGATIVE 1

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

// Depth bias (VkPipelineRasterizationStateCreateInfo::depthBias*), added to the
// depth plane of every filled triangle in window-z units:
//   o = constant * r + slope * max(|dz/dx|, |dz/dy|), then clamped by `clamp`
// (a positive clamp bounds o above, a negative one below, 0 does not clamp).
// For a UNORM depth the host folds r into `constant`; for a float depth
// (float_depth = 1) r is 2^(e - 23), e being the largest exponent of the
// primitive's three depths, so setup computes it.
typedef struct {
  uint32_t enable;
  uint32_t float_depth;
  float    constant, slope, clamp;
} setup_depth_bias_t;

// Without a near plane, setup clips at w >= SETUP_W_EPS instead.
#define SETUP_W_EPS 1e-6f

// A coverage-only primitive record (one fed from VS records) carries no varying
// planes, so the first unused one holds the depth bias its depth plane already
// includes (FloatA, window-z units): depth clipping tests the depth before bias.
#define SETUP_PRIM_DEPTH_BIAS(prim) ((prim).attribs.r.x)

#define SETUP_CULL_DIST_PACK(off, count) \
  (((uint32_t)(off) & 0xffffu) | (((uint32_t)(count) & 0xffu) << 16))
#define SETUP_CULL_DIST_OFF(w)   ((w) & 0xffffu)
#define SETUP_CULL_DIST_COUNT(w) (((w) >> 16) & 0xffu)

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

// Varying source of one emitted primitive. Varyings are not set up per plane:
// the fragment stage interpolates them straight out of the VS output records
// with perspective-correct barycentrics, beta_i = F_i / (F_0 + F_1 + F_2), where
// F_i = bary[i][0]*x + bary[i][1]*y + bary[i][2] at integer pixel (x,y) (the
// half-pixel centre offset is folded into bary[i][2]). The planes are the 2DH
// edge functions of the SOURCE triangle, so the barycentrics stay exact for
// every visible pixel of a near-clipped triangle's subtriangles, and clipping
// never touches a varying. vbase indexes the source triangle's first VS record;
// its three vertices are records vbase..vbase+2, the first being provoking.
typedef struct {
  float    bary[3][3];
  uint32_t vbase;
  uint32_t pad[3];
} rast_vary_t;

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
  uint32_t front_face;    // SETUP_FRONT_* (0 = positive-area winding is front)
  setup_depth_bias_t depth_bias;
  // 1: the near plane does not clip (depth clamp, or depth clip disabled);
  // setup still clips at w = SETUP_W_EPS, since nothing behind the eye can be
  // rasterized.
  uint32_t near_clip_off;
  // Cull distances in the VS records (vsrec_addr): SETUP_CULL_DIST_PACK(byte
  // offset, count). A primitive whose vertices all have one distance negative
  // is dropped. 0 = none.
  uint32_t cull_dist;
  uint32_t pad1_;
  uint64_t verts_addr;     // setup_vertex_t[3*num_tris]      (in, when vsrec_addr is 0)
  // VS output records (in): when set, triangle t's vertices are records 3t..3t+2,
  // vstride bytes apart, each starting with the clip-space position; setup reads
  // positions from them, carries no varying planes, and EMIT writes vary_addr.
  uint64_t vsrec_addr;
  uint32_t vstride;
  uint32_t pad0_;
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
  uint64_t vary_addr;      // rast_vary_t[P]                  (out, with vsrec_addr)
} pipe_arg_t;

// Pass-end multisample colour resolve. A multisample pass renders into a
// sample-interleaved colour plane (a pixel's S samples are contiguous, so a row
// is W*S texels); what the application reads is one texel per pixel. The
// resolve folds the samples with the box filter in gfx_sw.h and writes a dense
// single-sample plane the host copy can read at its natural size.
//
// It writes a SEPARATE destination rather than compacting in place: a thread
// resolving pixel i writes at i*bpp while a thread resolving pixel j>i is still
// reading from (j*S)*bpp, and with every pixel in flight at once the two
// overlap. One thread per pixel, grid-strided.
typedef struct {
  uint64_t src_addr;      // sample-interleaved colour plane          (in)
  uint64_t dst_addr;      // dense single-sample colour plane         (out)
  uint32_t width, height; // pixels
  uint32_t samples;       // samples per pixel
  uint32_t color_format;  // VX_OM_COLOR_FORMAT_*
} resolve_arg_t;

// Image blit (vkCmdBlitImage): every destination pixel of the box samples the
// source level through the software sampler at its centre's mapped position and
// is encoded into the destination format. The destination is a dense staging
// plane of the box; the host moves it to and from the resource. One thread per
// pixel, grid-strided.
typedef struct {
  uint64_t texstate_addr; // gfx_sw_texstate_t of the source level        (in)
  uint64_t dst_addr;      // dst_w x dst_h staging plane, dst_pitch bytes (in/out)
  uint32_t dst_w, dst_h, dst_pitch;
  uint32_t dst_format;    // VX_OM_COLOR_FORMAT_*
  uint32_t dst_bgra;      // 8-bit formats: blue in the low byte
  uint32_t dst_mask;      // per-channel write mask, RGBA in bits 0..3
  uint32_t filter;        // VX_TEX_FILTER_POINT / _BILINEAR
  uint32_t layer;         // source array layer
  float src_x0, src_y0;   // source texel position of the box's top-left corner
  float scale_x, scale_y; // source texels per destination pixel (negative: flip)
  float inv_src_w, inv_src_h;
  uint32_t src_3d;        // 1: a 3D source, sampled at depth coordinate src_r
  int32_t src_r;          // normalized depth of the slice, TEX_FXD_FRAC fixed point
} blit_arg_t;

// Raw nearest blit (blit_raw_k, gfx_blit_k.h): each texel of a dst_w x dst_h
// destination staging plane copies the bytes of its nearest source texel, under
// `mask`, without decoding them. It serves the blits whose formats match and
// must not be filtered -- integer and depth/stencil -- one source slice and
// destination slice per launch. One thread per pixel, grid-strided.
typedef struct {
  uint64_t src_addr;      // source slice, texel (0,0) at its start          (in)
  uint64_t dst_addr;      // dst_w x dst_h staging plane, dst_pitch bytes (in/out)
  uint32_t texel_bytes;   // 1..16
  uint32_t src_w, src_h, src_pitch; // source slice extent (clamped to) and pitch
  uint32_t dst_w, dst_h, dst_pitch;
  float src_x0, src_y0;   // source texel position of the box's top-left corner
  float scale_x, scale_y; // source texels per destination pixel (negative: flip)
  uint32_t mask[4];       // the texel bits to write
} blit_raw_arg_t;

// Texel copy (texel_copy_k, gfx_copy_k.h): for every texel of a w x h x d box,
// `bytes` bytes starting at byte src_byte of the source texel go to byte
// dst_byte of the destination texel, under `mask`. The two boxes have their own
// texel sizes and pitches, so one aspect of a depth/stencil texel can move to or
// from a buffer holding only that aspect. One thread per texel, grid-strided.
typedef struct {
  uint64_t src_addr;      // source span, texel (0,0,0) at its start       (in)
  uint64_t dst_addr;      // destination span, texel (0,0,0) at its start (in/out)
  uint32_t w, h, d;
  uint32_t src_texel, src_row, src_slice;
  uint32_t dst_texel, dst_row, dst_slice;
  uint32_t src_byte, dst_byte, bytes;  // bytes <= 16
  uint32_t mask[4];       // per copied byte: the bits to write
} texel_copy_arg_t;

// Masked texel fill (clear_k, gfx_clear_k.h). Every texel of a w x h x d box of
// the staging span takes `value` where `mask` is set and keeps its other bits,
// so one kernel serves colour clears, depth- or stencil-only clears of a
// combined format, and buffer fills. One thread per texel, grid-strided.
typedef struct {
  uint64_t dst_addr;      // staging span, texel (0,0,0) at its start      (in/out)
  uint32_t texel_bytes;   // 1..16
  uint32_t w, h, d;
  uint32_t row_pitch, slice_pitch;
  uint32_t value[4];      // the texel, packed in the destination format
  uint32_t mask[4];       // the texel bits to write
} clear_arg_t;
