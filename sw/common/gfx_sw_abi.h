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

// C ABI for the gfx_v2 on-device software fallback.
// The vortexpipe FS (NIR→LLVM, C) emits calls to these entry points when a unit
// is routed to software; the host builds the POD descriptors from the bound
// pipeline state and passes their resident device pointers via the kernel arg.
// The implementations (gfx_sw_abi.cpp) are thin wrappers over the C++ single-
// source-of-truth headers (gfx_frag_tex.h / gfx_sw.h / gfx_frag_rast.h), so the SW path
// the driver runs is bit-identical to the FF model and the unit tests.
//
// This header is plain C (no C++), so the C mesa driver can include it. The POD
// descriptors mirror gfx_sw::TexState / gfx_sw::om_state_t exactly; gfx_sw_abi.cpp
// static_asserts the layouts match.

#pragma once

#include <stdint.h>
#include <VX_types.h>   // VX_TEX_LOD_MAX

#ifdef __cplusplus
extern "C" {
#endif

// texstate.filter bit above the mip sub-field: the bound sampler has a mip
// chain. When clear, a fragment shader forces LOD 0 — a non-mipmapped
// NEAREST/LINEAR sampler always reads the base level, whatever the coordinate
// gradients. (The mag/min and mip-linear bits mirror the VX_TEX_FILTER_* enum.)
#define GFX_SW_TEX_FILTER_MIP_ENABLE (1u << 2)

// texstate.filter bit 3: the min (minification) tap filter, 0=point 1=bilinear.
// bit 0 is the mag (magnification) tap; a fragment shader picks between them by the
// sign of its computed LOD (minified -> min, magnified -> mag). The HW TEX unit has
// one per-draw filter DCR, so a mipmapped sampler routes to the SW sampler, which
// takes the shader-resolved filter explicitly.
#define GFX_SW_TEX_FILTER_MIN_BILINEAR (1u << 3)

// texstate.filter bit 4: the bound texture has non-power-of-two dimensions. The FF
// vx_tex4 unit is POT-only, so a HW-TEX fragment shader routes an NPOT texture to
// the SW sampler (which addresses it by width/height), the same runtime branch a
// mipmapped sampler takes. Set per draw from the bound texture's mip-0 dims.
#define GFX_SW_TEX_FILTER_NPOT (1u << 4)

// texstate.filter bit 5: the bound texture's format is above the FF set
// (VX_TEX_FORMAT_FF_MAX). The FF vx_tex4 unit decodes only the FF formats and its
// stride table covers only 1/2/4-byte texels, so an extended-format texture must
// sample in software — the same runtime branch a mipmapped or NPOT texture takes.
// Set per draw from the bound texture's resolved VX format.
#define GFX_SW_TEX_FILTER_EXT_FORMAT (1u << 5)

// Resident per-stage texture descriptor (mirror of gfx_sw::TexState).
typedef struct {
  uint64_t base;                          // mip 0 base (TEX_ADDR << 6)
  uint32_t mip_off[VX_TEX_LOD_MAX + 1];   // per-LOD byte offset from base
  uint32_t logdim;                        // {log_h << 16 | log_w} of mip 0
  uint32_t format;                        // VX_TEX_FORMAT_*
  uint32_t filter;                        // mag tap (bit0) | mip-linear (bit1) | mip-enable (bit2) | min tap (bit3) | NPOT (bit4) | ext format (bit5)
  uint32_t wrap;                          // {wrap_v << 16 | wrap_u}
  uint32_t width;                         // mip-0 integer width  (0 => POT via logdim)
  uint32_t height;                        // mip-0 integer height (0 => POT via logdim)
  uint32_t border;                        // ARGB8888 border colour (WRAP_BORDER)
  uint32_t layer_stride;                  // bytes per array layer / cube face (0 => single 2D)
  uint32_t compare_func;                  // shadow compare op (VX_OM_DEPTH_FUNC_*); 0 => none
  uint32_t swizzle;                       // view component map: r|g<<3|b<<6|a<<9 (0..3=RGBA, 4=0, 5=1)
  uint32_t min_lod;                       // sampler LOD clamp lower bound, Q(VX_TEX_LOD_FRAC_BITS)
  uint32_t max_lod;                       // sampler LOD clamp upper bound, Q(VX_TEX_LOD_FRAC_BITS)
  int32_t  lod_bias;                      // sampler LOD bias, signed Q(VX_TEX_LOD_FRAC_BITS)
  uint32_t depth;                         // sampler3D: mip-0 depth-slice count; samplerCubeArray: cube count; 0 otherwise
  uint32_t wrap_w;                        // VX_TEX_WRAP_* for the 3D depth (r) axis
} gfx_sw_texstate_t;

// Resident output-merger descriptor (mirror of gfx_sw::om_state_t).
typedef struct {
  uint32_t depth_func;
  uint32_t stencil_func[2], stencil_zpass[2], stencil_zfail[2], stencil_fail[2];
  uint32_t stencil_ref[2], stencil_mask[2], stencil_writemask[2];
  uint32_t depth_writemask;
  uint32_t blend_mode_rgb, blend_mode_a;
  uint32_t blend_src_rgb, blend_src_a, blend_dst_rgb, blend_dst_a;
  uint32_t blend_const, logic_op;
  uint64_t zbuf_base, cbuf_base;
  uint32_t zbuf_pitch, cbuf_pitch;
  uint32_t cbuf_writemask4;
  uint32_t color_format;               // VX_OM_COLOR_FORMAT_*
  uint32_t depth_format;               // VX_OM_DEPTH_FORMAT_*
  uint32_t depth_enabled, stencil_enabled[2], blend_enabled;
  uint32_t cbuf_writemask;
  uint32_t color_read, color_write;
} gfx_sw_omstate_t;

// Per-attachment colour descriptor for MRT (mirror of gfx_sw::om_color_t). The
// depth/stencil state stays in gfx_sw_omstate_t (shared); one of these describes
// each colour attachment's base/pitch/blend/write-mask.
typedef struct {
  uint64_t cbuf_base;
  uint32_t cbuf_pitch;
  uint32_t blend_mode_rgb, blend_mode_a;
  uint32_t blend_src_rgb, blend_src_a, blend_dst_rgb, blend_dst_a;
  uint32_t blend_const, logic_op;
  uint32_t cbuf_writemask4;
  uint32_t color_format;               // VX_OM_COLOR_FORMAT_*
  uint32_t blend_enabled;
  uint32_t cbuf_writemask;
  uint32_t color_read, color_write;
} gfx_sw_omcolor_t;

// Sample the resident texture (software fallback for vx_tex4). `lod` is integer
// for point/bilinear, fixed-point when the mip-linear filter bit is set. `filter`
// (tap in bit0, mip-linear in bit1) is resolved by the caller — the fragment
// shader picks the min or mag tap per fragment from the sign of its LOD.
uint32_t gfx_tex_sample_sw(const gfx_sw_texstate_t* st,
                           int32_t u, int32_t v, uint32_t lod, uint32_t filter);

// texelFetch: the four channels of the exact texel at integer (x,y) of integer
// `lod`, as floats in RGBA order -- no wrap/filter/mip. Floats because a
// float-format texture's values lie outside [0,1]; a non-float format yields the
// same [0,1] channels the shader unpack would.
void gfx_tex_fetch_f32(const gfx_sw_texstate_t* st,
                       int32_t x, int32_t y, uint32_t lod, float* out);

// texelFetch on a 2D array: the texel at (x,y) of integer `layer` and `lod`
// (see gfx_tex_fetch_f32).
void gfx_tex_fetch_array_f32(const gfx_sw_texstate_t* st,
                             int32_t x, int32_t y, uint32_t layer, uint32_t lod,
                             float* out);

// textureGather: channel `comp` of the 2x2 footprint at (x,y), base level, packed
// in GL gather order as bytes x | y<<8 | z<<16 | w<<24.
uint32_t gfx_tex_gather_sw(const gfx_sw_texstate_t* st,
                           int32_t x, int32_t y, uint32_t comp);

// textureGatherCmp: compare each of the 2x2 depth taps at (x,y) against `ref_bits`
// with st->compare_func; pack the 0/1 results (0xff/0x00) in GL gather order.
uint32_t gfx_tex_gather_cmp_sw(const gfx_sw_texstate_t* st,
                               int32_t x, int32_t y, uint32_t ref_bits);

// sampler2DArrayShadow textureGatherCmp on the `layer` slice.
uint32_t gfx_tex_gather_cmp_array_sw(const gfx_sw_texstate_t* st,
                                     int32_t x, int32_t y, uint32_t ref_bits,
                                     uint32_t layer);

// sampler2DShadow: sample the depth texture at (x,y), compare each tap against the
// reference `ref_bits` (a float bit-pattern) using st->compare_func, and return the
// result as a float bit-pattern in [0,1] (0/1 for a point sampler, a PCF fraction
// for a bilinear one). `lod` selects the mip level (integer level for mip-nearest,
// Q(LOD_FRAC) for a mip-linear blend), matching gfx_tex_sample_*.
uint32_t gfx_tex_shadow_sw(const gfx_sw_texstate_t* st,
                           int32_t x, int32_t y, uint32_t ref_bits, uint32_t filter,
                           uint32_t lod);

// sampler2DArrayShadow: like gfx_tex_shadow_sw, but the integer `layer` selects
// the array slice (base + layer*layer_stride) before the depth compare.
uint32_t gfx_tex_shadow_array_sw(const gfx_sw_texstate_t* st,
                                 int32_t x, int32_t y, uint32_t layer,
                                 uint32_t ref_bits, uint32_t filter, uint32_t lod);

// samplerCubeShadow: pick the cube face from the (sc,tc,rc) direction, project,
// and compare against `ref_bits` at that face's slice.
uint32_t gfx_tex_shadow_cube_sw(const gfx_sw_texstate_t* st,
                                float sc, float tc, float rc,
                                uint32_t ref_bits, uint32_t filter, uint32_t lod);

// 2D-array view: sample integer `layer` of the bound array texture.
uint32_t gfx_tex_sample_array_sw(const gfx_sw_texstate_t* st,
                                 int32_t u, int32_t v, uint32_t layer, uint32_t lod);

// Cube view: sample the face selected by the major axis of the (sc,tc,rc)
// direction vector supplied by the FS.
uint32_t gfx_tex_sample_cube_sw(const gfx_sw_texstate_t* st,
                                float sc, float tc, float rc, uint32_t lod);

// Cube-array view: array_index selects the cube, (sc,tc,rc) the face; the slice is
// array_index*6 + face.
uint32_t gfx_tex_sample_cube_array_sw(const gfx_sw_texstate_t* st,
                                      float sc, float tc, float rc,
                                      uint32_t array_index, uint32_t lod);

// samplerCubeArrayShadow: like gfx_tex_sample_cube_array_sw, but compare against
// ref_bits at slice array_index*6 + face.
uint32_t gfx_tex_shadow_cube_array_sw(const gfx_sw_texstate_t* st,
                                      float sc, float tc, float rc,
                                      uint32_t array_index,
                                      uint32_t ref_bits, uint32_t filter,
                                      uint32_t lod);

// 3D view: (u, v) are S.23 fixed-point in-slice coords, `w` the S.23 depth coord;
// `filter` carries the resolved tap (bit0 linear => blend the two bracketing
// depth slices, else the nearest slice).
uint32_t gfx_tex_sample_3d_sw(const gfx_sw_texstate_t* st,
                              int32_t u, int32_t v, int32_t w, uint32_t lod, uint32_t filter);

// Merge one fragment (software fallback for vx_om4): depth/stencil test + blend
// + ROP at pixel (x, y) for face (0=front, 1=back) using the resident om state.
// `covered` is the sub-pixel's coverage bit; uncovered fragments are dropped
// here so the SIMT caller stays straight-line (the divergent branch lives in
// this compilation unit, under the device toolchain's divergence lowering).
void gfx_om_fragment_sw(const gfx_sw_omstate_t* st, uint32_t covered,
                        uint32_t x, uint32_t y, uint32_t face,
                        uint32_t color, uint32_t depth);

// MRT software fallback: one shared depth/stencil op against `st`, then a
// per-attachment blend + colour write of colors[k] for each of the `num_color`
// attachments described by `rt`. `covered` drops uncovered fragments here so the
// SIMT caller stays straight-line.
void gfx_om_fragment_mrt_sw(const gfx_sw_omstate_t* st, const gfx_sw_omcolor_t* rt,
                            uint32_t num_color, uint32_t covered,
                            uint32_t x, uint32_t y, uint32_t face,
                            const uint32_t* colors, uint32_t depth);

// One covered 2x2 quad produced by the SW fine-rasterizer (fallback for the FF
// RASTER producer). Layout matches the FF frag payload the FS wrapper reads:
// pos_mask (cov_mask[3:0] | quad_x | quad_y | face@31) and the per-fragment edge
// values bcoords[axis*4 + corner] (axis 0..2, corner 0..3; fixed_t<16> raw).
typedef struct {
  uint32_t pos_mask;
  int32_t  bcoords[12];
} gfx_rast_quad_t;

// Walk one primitive over one screen tile (origin tx,ty, side 1<<tile_logsize),
// appending every covered quad to out[0..max). Returns the count (capped at max).
// `prim` points at the resident rast_prim_t for `pid`. Used by the SW-raster FS
// wrapper variant (one thread per tile iterates all prims in draw order).
uint32_t gfx_rast_walk_tile_sw(const void* prim, uint32_t pid,
                               uint32_t tx, uint32_t ty, uint32_t tile_logsize,
                               uint32_t scissor_w, uint32_t scissor_h,
                               gfx_rast_quad_t* out, uint32_t max);

#ifdef __cplusplus
}
#endif
