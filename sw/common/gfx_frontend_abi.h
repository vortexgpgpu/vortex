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
#include <VX_types.h>

// Near-plane clip + triangle-setup depth range. The device clip stage uses
// these directly (see pipe_clip_and_setup), so they are part of the contract.
#ifndef SETUP_NEAR
#define SETUP_NEAR     0.0f
#endif
#ifndef SETUP_FAR
#define SETUP_FAR      1.0f
#endif

// Front-end counters at pipe_arg_t::meta_addr: { P, keys, nb, stipple }, the
// pass's kept-primitive count, its (bin, primitive) key count, its non-empty bin
// count, and the counter a stippled strip carries into the next pass of the draw.
#define PIPE_META_WORDS 4

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

// One slot of a draw's viewport array (VkPipelineViewportStateCreateInfo): the
// transform, and the window rectangle a primitive drawn through it is confined
// to (the viewport's rect, the slot's scissor and the render target,
// intersected), packed as rast_prim_t's scissor words. A primitive selects its
// slot by the ViewportIndex its provoking vertex wrote.
#define SETUP_MAX_VIEWPORTS 16
typedef struct {
  setup_viewport_t vp;
  uint32_t scissor_x, scissor_y;
} setup_vp_slot_t;

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

// The same depth plane as attribs.z, in float, for a float depth target to
// evaluate in fp32 with none of the Q7.24 plane's range or precision limits:
// gradients and the biased depth at an origin (x, y, z) the second plane holds
// in integer pixel coordinates, so z = z0 + a*(px - x0) + b*(py - y0). Relative
// to a vertex, the constant does not cancel against large gradient terms.
#define SETUP_PRIM_ZFLOAT_PLANE(prim)  ((prim).attribs.w0)
#define SETUP_PRIM_ZFLOAT_ORIGIN(prim) ((prim).attribs.w1)

#define SETUP_PRIM_TRIANGLE 0
#define SETUP_PRIM_LINE     1
#define SETUP_PRIM_POINT    2
#define SETUP_PRIM_POLYGON  3

#define SETUP_POLY_FILL  0
#define SETUP_POLY_LINE  1
#define SETUP_POLY_POINT 2

// A line is a rectangle along it (VkLineRasterizationModeKHR RECTANGULAR, the
// strictLines default), a parallelogram offset along the minor axis
// (BRESENHAM: the diamond-exit rule's coverage for one-pixel lines), or a
// rectangle half a pixel larger on every side whose fragments take their
// coverage into alpha (RECTANGULAR_SMOOTH).
#define SETUP_LINE_RECT      0
#define SETUP_LINE_BRESENHAM 1
#define SETUP_LINE_SMOOTH    2

// A point primitive's record carries its centre (window x, y) and size, for
// gl_PointCoord, in another plane a coverage-only record leaves unused; the
// three words are floats.
#define SETUP_PRIM_POINT_PLANE(prim) ((prim).attribs.g)

// A line primitive's record carries its stipple counter as a window-space plane
// (x, y, constant; floats): the counter at a pixel centre is the segment's start
// count plus its distance along the line -- Euclidean for a rectangular line,
// along the major axis for a Bresenham one.
#define SETUP_PRIM_STIPPLE_PLANE(prim) ((prim).attribs.b)

// A smooth line's record carries what its coverage is computed from, as floats:
// the window-space planes of the signed distance across the line and along it
// from its midpoint, and (half width, half length, length) of the drawn
// rectangle. A fragment's coverage is
//   sat(half_width - |across|) * min(sat(half_length - |along|), length).
#define SETUP_PRIM_SMOOTH_ACROSS(prim) ((prim).attribs.a)
#define SETUP_PRIM_SMOOTH_ALONG(prim)  ((prim).attribs.u)
#define SETUP_PRIM_SMOOTH_SIZE(prim)   ((prim).attribs.v)

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

// Primitives of a draw one front-end pass takes. Each emits up to SETUP_MAX_SUB
// records, and a pass's records must fit one RASTER pass (GFX_RASTER_MAX_PRIMS,
// 1 << VX_RASTER_PID_BITS), so a larger draw runs as consecutive passes over
// primitive ranges starting at pipe_arg_t::prim_base.
#define PIPE_PASS_MAX_PRIMS ((1u << VX_RASTER_PID_BITS) / SETUP_MAX_SUB)

// Varying source of one emitted primitive. Varyings are not set up per plane:
// the fragment stage interpolates them straight out of the VS output records
// with perspective-correct barycentrics, beta_i = F_i / (F_0 + F_1 + F_2), where
// F_i = bary[i][0]*x + bary[i][1]*y + bary[i][2] at integer pixel (x,y) (the
// half-pixel centre offset is folded into bary[i][2]), and a smooth input is
// sum(beta_i * record vrec[i]). For a triangle the planes are the 2DH edge
// functions of the SOURCE triangle, so the barycentrics stay exact for every
// visible pixel of a near-clipped triangle's subtriangles, and clipping never
// touches a varying. A line keeps two planes (start, end) and a point one
// constant plane; a plane that is zero leaves its record free to be the one a
// flat input reads, which is vrec[0] or vrec[2] by the provoking-vertex mode.
typedef struct {
  float    bary[3][3];
  uint32_t vrec[3];
  uint32_t vp_layer;   // RAST_VARY_VP_LAYER: the primitive's viewport and layer
  uint32_t prim_id;    // gl_PrimitiveID
} rast_vary_t;

// ViewportIndex and Layer of a primitive as the fragment stage reads them, after
// setup's clamp: viewport in the low half, layer in the high half.
#define RAST_VARY_VP_LAYER(vp, layer) (((uint32_t)(vp) & 0xffffu) | ((uint32_t)(layer) << 16))

// Input assembly: how a draw's index positions make its primitives. The vertex
// kernel's thread for vertex k of assembled primitive p reads the index at the
// position this places there, straight from the application's index buffer.
// Position `last` ends a primitive; `run` is where its run begins (0, or one
// past the restart index that began it). With adjacency a primitive's vertices
// are in geometry-shader input order, adjacent vertices interleaved:
//   LIST        last-per+1 .. last (point, line and triangle lists, and line and
//               triangle lists with adjacency: per 4 and 6)
//   LINE_STRIP  last-per+1 .. last (line strips, and with adjacency: per 4)
//   TRI_STRIP   last-2, last-1, last, the first two swapped on an odd triangle of
//               its run when the provoking vertex is last, else the last two
//   TRI_FAN     run, last-1, last when the provoking vertex is last, else
//               last-1, last, run
//   TRI_STRIP_ADJ  triangle i of its run, from b = last-5 = run+2i: b, b-2 (b+1
//               for the first), b+2, b+6 (b+5 for the last), b+4, b+3; an odd
//               one permuted to (2, 1, 0, 5, 4, 3) of those when the provoking
//               vertex is last, else (0, 5, 4, 3, 2, 1)
#define GFX_IA_LIST          0u
#define GFX_IA_LINE_STRIP    1u
#define GFX_IA_TRI_STRIP     2u
#define GFX_IA_TRI_FAN       3u
#define GFX_IA_TRI_STRIP_ADJ 4u

// One assembled primitive of a draw whose restart index cuts its runs, written
// by ia_k: the position of its last vertex and the run it belongs to.
typedef struct {
  uint32_t last;
  uint32_t run;
} gfx_ia_prim_t;

// A draw's input assembly, in device memory. The draw parameters are those of
// an indexed draw command; `prims` is the primitives each instance assembles --
// the closed form of the draw for the host to write, or, when `restart` is set,
// the count ia_k finds. Every later stage takes the draw's counts from here,
// and the launches are only sized to bound them, so a device writer of the
// draw parameters (an indirect draw) only has to fill the closed form too.
typedef struct {
  uint64_t index_addr;    // the bound index buffer, from its index 0
  uint64_t map_addr;      // gfx_ia_prim_t[prims] when `restart` is set (ia_k out)
  uint64_t query_block;   // GFX_QC_* counters ia_k counts the primitives into; 0 = none
  uint32_t index_size;    // bytes per index: 1, 2 or 4; 0: the vertex is the position
  uint32_t index_limit;   // indices the bound buffer holds; an index past it reads as 0
  uint32_t first;         // the draw's first index
  uint32_t count;         // the draw's index positions
  int32_t  base_vertex;   // added to every index read
  uint32_t instances;
  uint32_t first_instance;
  uint32_t topology;      // GFX_IA_*
  uint32_t per;           // vertices per primitive: 1, 2, 3, 4 or 6
  uint32_t restart;       // `restart_index` (masked to index_size) cuts the runs
  uint32_t restart_index;
  uint32_t prims;         // primitives per instance
  uint32_t query_scale;   // draws each counted primitive stands for (instances x views)
  uint32_t query_xfb;     // also count them as needed by a stream query
  // A geometry stage's invocations per primitive, 0 = none: the primitives
  // are then its input, counted as its invocations rather than as primitives
  // the rasterizer and a stream query see, which its own scan counts.
  uint32_t gs_invocations;
} gfx_ia_t;

// The three ia_k launches of a draw with restart, in CP order: COUNT and EMIT
// over the whole grid, each thread a contiguous chunk of positions; SCAN on one
// CTA, between them. Its partials hold three words per COUNT thread. A draw
// short enough for one CTA runs the three in one launch of that CTA (ALL).
#define IA_STAGE_COUNT 0u
#define IA_STAGE_SCAN  1u
#define IA_STAGE_EMIT  2u
#define IA_STAGE_ALL   3u
typedef struct {
  uint64_t ia_addr;       // gfx_ia_t                          (in; out: prims)
  uint64_t part_addr;     // uint32[3 * threads]               (scratch)
  uint64_t tsum_addr;     // uint32[2 * SCAN's blockDim.x]     (scratch)
  uint32_t stage;         // IA_STAGE_*
  uint32_t threads;       // threads of the COUNT and EMIT launches
} ia_arg_t;

// Per-launch front-end argument block. Addresses are device byte addresses;
// counts/config are runtime values the kernels read (so launch dims and pool
// addresses can stay static while the work scales with the in-memory counts).
typedef struct {
  uint32_t num_tris;      // primitives of this pass, of prim_type
  // The draw's first primitive in this pass: pass-local primitive t is draw
  // primitive prim_base + t, which selects its VS records, strip restarts and
  // instance. The pass's records, bins and pids stay pass-local.
  uint32_t prim_base;
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
  // What one primitive is (SETUP_PRIM_*): triangle t reads records 3t..3t+2, line
  // t records 2t, 2t+1, point t record t, and a polygon drawn in line or point
  // mode takes three primitives per triangle, edge or vertex t%3 of triangle t/3.
  // An instanced draw numbers primitives and records within each instance.
  uint32_t prim_type;
  uint32_t line_mode;     // SETUP_LINE_*
  uint32_t poly_front, poly_back; // SETUP_POLY_* for SETUP_PRIM_POLYGON, by facing
  uint32_t provoke_last;  // flat inputs take a primitive's last vertex
  // gl_PointSize in the VS records (byte offset), or 0 for point_size: offset 0
  // holds the position.
  uint32_t psize_off;
  float    line_width, point_size;
  float    point_size_max;  // a point's size is clamped to [0, point_size_max]
  // 1: the lines are the consecutive segments of one stippled strip, so the
  // stipple counter runs on from one segment to the next instead of restarting.
  uint32_t stipple_strip;
  // Primitives and VS records per instance of an instanced draw; 0: one instance.
  uint32_t inst_prims;
  // The draw's viewport array (setup_vp_slot_t[vp_count]) at vp_table_addr; a
  // vp_table_addr of 0 selects the one viewport above, unscissored. vpi_off and layer_off are the byte offsets of
  // ViewportIndex and Layer in the VS records (0 = not written). With layer_off,
  // layer_max clamps a primitive's layer; every layer is drawn in the one pass,
  // each fragment going to its primitive's layer (rast_vary_t::vp_layer).
  uint32_t vp_count;
  uint32_t vpi_off;
  uint32_t layer_off;
  uint32_t layer_max;
  uint64_t vp_table_addr;
  uint64_t verts_addr;     // setup_vertex_t[3*num_tris]      (in, when vsrec_addr is 0)
  // VS output records (in): when set, a primitive's vertices are records as
  // prim_type says, vstride bytes apart, each starting with the clip-space
  // position; setup reads positions from them, carries no varying planes, and
  // EMIT writes vary_addr.
  uint64_t vsrec_addr;
  uint32_t vstride;
  uint32_t inst_verts;
  // The draw's gfx_ia_t, read by setup_ia_k (setup_k takes the counts above as
  // exact). Its counts are the draw's and the ones above
  // only bound them: a primitive past what the assembly made is skipped, and
  // an instance's records are as many as its primitives take. Under a restart
  // index a stippled strip's segment that begins a run (its map entry's
  // last == run + per - 1) restarts the counter; otherwise every instance's strip is
  // one run.
  uint64_t ia_addr;
  uint64_t slot_prim_addr; // rast_prim_t[num_tris*MAX_SUB]   (scratch)
  uint64_t slot_bbox_addr; // setup_bbox_t[num_tris*MAX_SUB]  (scratch)
  uint64_t keep_addr;      // uint32[num_tris]                (scratch)
  uint64_t offset_addr;    // uint32[num_tris + 1]            (scratch)
  uint64_t tsum_addr;      // uint32[T]                       (scratch)
  uint64_t stipple_addr;   // uint32[num_tris]: steps, then start (scratch, with stipple_strip)
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
  uint64_t meta_addr;      // uint32[PIPE_META_WORDS]         (out)
  uint64_t vary_addr;      // rast_vary_t[P]                  (out, with vsrec_addr)
  // An indexed primitive list (pipe_list_entry_t[]), the output of a shader
  // stage that makes its own primitives: draw primitive t reads its records
  // from entry t instead of by the implicit rule above, and a polygon drawn as
  // lines or points takes entry t / 3. 0 = implicit records.
  uint64_t prim_index_addr;
  // The list's length on the device (uint32), which this pass clamps to
  // num_tris primitives from prim_base; 0 = num_tris is the count.
  uint64_t num_tris_addr;
  uint32_t list_aux;       // PIPE_LIST_AUX_*: what an entry's aux word holds
  // 1: the implicit records come in adjacency groups -- four per line, six per
  // triangle -- of which the primitive is records 1 and 2, or 0, 2 and 4.
  uint32_t adjacency;
} pipe_arg_t;

// One entry of an indexed primitive list. A triangle names its three records in
// winding order with its provoking vertex where flat inputs read it, a line its
// two ends in rec[0..1], a point its record in rec[0]. `aux` is per-primitive
// data in the meaning pipe_arg_t::list_aux gives it, and its top bit marks a
// line that begins a strip, where a stipple counter restarts.
typedef struct {
  uint32_t rec[3];
  uint32_t aux;
} pipe_list_entry_t;

#define PIPE_LIST_AUX_NONE   0u
#define PIPE_LIST_AUX_PRIMID 1u   // aux holds the primitive's gl_PrimitiveID
#define PIPE_LIST_HEAD       0x80000000u
#define PIPE_LIST_AUX_MASK   0x7fffffffu

// A capturing draw's transform feedback, as the stage that captures reads it
// and, when a geometry stage's counts are found on the device, as the scan that
// orders them commits it (gfx_xfb_commit). A primitive is written only when
// every captured buffer of its vertex stream has room for all of it, so a
// stream writes its first W primitives and none after. The filled sizes are
// device words: the draw reads them at off_rd and leaves the advanced ones at
// off_wr, a different four, so no thread reads a word another writes.
typedef struct {
  uint64_t base[4];          // device address of each bound range
  uint32_t size[4];          // bytes in each bound range
  uint64_t off_rd;           // device address of the uint32_t[4] filled sizes
  uint64_t off_wr;           // ... and of the four the draw leaves behind
  uint64_t query_block;      // GFX_QC_* counters, 0 when no query counts
  uint32_t prims;            // vertex stage: primitives in the draw, every instance
  uint32_t per;              // vertices per primitive: 1, 2 or 3
  uint32_t mask;             // buffers captured into: written by the shader, bound
  uint32_t streams;          // geometry stage: the vertex streams it captures
  uint32_t stride[4];        // bytes per captured vertex in each buffer
  uint32_t stream_prims[4];  // geometry stage, static counts: each stream's primitives
  uint8_t  buffer_stream[4]; // the vertex stream each buffer captures
  // Vertex stage: records per primitive, `per` or with adjacency 4 or 6, of
  // which only the primitive's own vertices are captured.
  uint32_t group;
} gfx_xfb_desc_t;

// Device-wide exclusive prefix sum (scan_k, gfx_scan_k.h) over n elements of
// `lanes` uint32 lanes each, lane by lane, in three launches the command
// processor drains one after the other, so no CTA waits on another:
//   UP    `parts` CTAs, CTA j summing elements [j * chunk, (j + 1) * chunk) of
//         each lane into part[j]
//   TOP   one CTA: the partials scanned in place, the totals written
//   DOWN  `parts` CTAs again, each writing its elements' exclusive prefixes from
//         its partial on
// Every step inside a CTA is a log-depth scan of its threads' sums.
#define PIPE_SCAN_UP   0u
#define PIPE_SCAN_TOP  1u
#define PIPE_SCAN_DOWN 2u
#define PIPE_SCAN_MAX_LANES 4u

typedef struct {
  uint32_t stage;
  uint32_t n;              // elements, or a bound on them with n_addr
  uint32_t lanes;          // 1 .. PIPE_SCAN_MAX_LANES
  uint32_t chunk;          // elements per CTA in UP and DOWN
  uint32_t parts;          // CTAs of UP and DOWN
  // TOP also adds the totals to the query counter block (GFX_QC_*) when one
  // is given: every lane's to counter query_sum, and lane k's to the counters
  // named in byte k of each query_lane word (PIPE_SCAN_QC_NONE: none).
  uint32_t query_sum;
  uint32_t query_lane[3];
  uint32_t streams;        // the vertex streams the lanes scan, lane k the k-th
  uint64_t n_addr;         // the element count on the device (uint32), 0 = n
  uint64_t in_addr;        // uint32[n * lanes]
  uint64_t out_addr;       // uint32[n * lanes]: the exclusive prefixes
  uint64_t part_addr;      // uint32[parts * lanes]
  uint64_t total_addr;     // uint32[lanes]
  uint64_t scratch_addr;   // uint32[parts * 2 * blockDim]
  uint64_t query_addr;     // the query counter block, 0 = none
  // A geometry stage's transform feedback (gfx_xfb_desc_t), which TOP commits
  // from the totals, lane k being the k-th stream it emits to; 0 = none.
  uint64_t xfb_addr;
} pipe_scan_arg_t;

#define PIPE_SCAN_QC_NONE 0xffu

// Multisample colour resolve. A multisample image keeps one whole image per
// sample, src_sample_stride bytes apart; the resolve folds a pixel's samples with
// the box filter in gfx_sw.h into one texel of the single-sample destination.
// One thread per pixel, grid-strided.
typedef struct {
  uint64_t src_addr;          // sample 0's image                     (in)
  uint64_t dst_addr;          // single-sample image                  (out)
  uint32_t width, height;     // pixels
  uint32_t samples;           // samples per pixel
  uint32_t color_format;      // VX_OM_COLOR_FORMAT_*
  uint32_t src_pitch;         // bytes per row of each sample's image
  uint32_t src_sample_stride; // bytes between the sample images
  uint32_t dst_pitch;         // bytes per row of the destination
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
//
// With samples > 1 the source is a multisample image's `samples` planes,
// sample_stride bytes apart, and the depth field of each texel's first word --
// depth_bits bits at depth_shift, an f32 when depth_float -- is their average
// (a depth resolve); the other masked bytes come from sample 0.
typedef struct {
  uint64_t src_addr;      // source span, texel (0,0,0) at its start       (in)
  uint64_t dst_addr;      // destination span, texel (0,0,0) at its start (in/out)
  uint32_t w, h, d;
  uint32_t src_texel, src_row, src_slice;
  uint32_t dst_texel, dst_row, dst_slice;
  uint32_t src_byte, dst_byte, bytes;  // bytes <= 16
  uint32_t mask[4];       // per copied byte: the bits to write
  uint32_t samples, sample_stride;     // 0 or 1: a plain copy
  uint32_t depth_shift, depth_bits, depth_float;
} texel_copy_arg_t;

// Texture pack (tex_pack_k, gfx_tex_pack_k.h): the TEX unit's copy of a sampled
// texture. Every texel of every level and layer of the source, read where the
// texture lives through its software-sampler descriptor, is stored as the unit
// reads it (`mode`) in the packed chain the unit addresses: each layer's levels
// back to back (a 3D level holds its slices back to back), level l at
// dst_mip_off[l] within its layer, layers dst_layer_stride bytes apart. One
// thread per texel, grid-strided.
#define TEX_PACK_ARGB8      0u  // colour, decoded to A8R8G8B8
#define TEX_PACK_COPY16     1u  // the texel's first two bytes (D16)
#define TEX_PACK_COPY32     2u  // the texel's first word (D32F, D32F_S8's depth)
#define TEX_PACK_D24_TO_F32 3u  // a D24 texel's depth as an f32 (D32F)
typedef struct {
  uint64_t texstate_addr;   // gfx_sw_texstate_t of the source's level 0, layer 0 (in)
  uint64_t dst_addr;        // packed chain                                    (out)
  uint32_t mode;            // TEX_PACK_*
  uint32_t levels;          // levels to pack
  uint32_t layers;          // array layers (cube faces included); 1 for 3D
  uint32_t depth;           // a 3D texture's slices at level 0; 0 otherwise
  uint32_t dst_layer_stride;
  uint32_t dst_mip_off[VX_TEX_LOD_MAX + 1];
} tex_pack_arg_t;

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
