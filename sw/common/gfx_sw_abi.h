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

// The standard 4x sample positions (Vulkan's standardSampleLocations), as offsets
// from the pixel centre in 1/16 px: sample k sits at (8 + DX16[k], 8 + DY16[k])/16
// from the pixel's top-left corner -- (0.375, 0.125), (0.875, 0.375),
// (0.125, 0.625), (0.625, 0.875).
#define GFX_MSAA4_SAMPLE_DX16 { -2,  6, -6,  2 }
#define GFX_MSAA4_SAMPLE_DY16 { -6, -2,  2,  6 }

// texstate.filter bit above the mip sub-field: the bound sampler has a mip
// chain. When clear, a fragment shader forces LOD 0 — a non-mipmapped
// NEAREST/LINEAR sampler always reads the base level, whatever the coordinate
// gradients. (The mag/min and mip-linear bits mirror the VX_TEX_FILTER_* enum.)
#define GFX_SW_TEX_FILTER_MIP_ENABLE (1u << 2)

// texstate.filter bit 3: the min (minification) tap filter, 0=point 1=bilinear.
// bit 0 is the mag (magnification) tap; a fragment shader picks between them by the
// sign of its computed LOD (minified -> min, magnified -> mag), and hands the
// resolved tap to the TEX unit in its operand or to the SW sampler explicitly.
#define GFX_SW_TEX_FILTER_MIN_BILINEAR (1u << 3)

// texstate.filter bit 4: the sampler takes unnormalized (texel) coordinates; the
// shader divides them by the level-0 dims before sampling.
#define GFX_SW_TEX_FILTER_UNNORM (1u << 4)

// texstate.filter bit 7: cube filtering is seamless -- a bilinear tap past a face
// edge reads the adjacent face (Vulkan's default; VK_EXT_non_seamless_cube_map
// clears it).
#define GFX_SW_TEX_FILTER_SEAMLESS (1u << 7)

// texstate.filter bit 5: the bound texture's format is above the FF set
// (VX_TEX_FORMAT_FF_MAX). The FF vx_tex4 unit decodes only the FF formats and its
// stride table covers only 1/2/4-byte texels, so an extended-format texture must
// sample in software.
// Set per draw from the bound texture's resolved VX format.
#define GFX_SW_TEX_FILTER_EXT_FORMAT (1u << 5)

// textureOffset's texel offset, carried in the caller-resolved filter word as
// three signed 4-bit fields (x, y, z) from this bit; the sampler applies it in
// texels of each level it reads.
#define GFX_SW_TEX_FILTER_OFFSET_SHIFT 16

// Either axis wraps with CLAMP_TO_BORDER, which the FF unit decodes as REPEAT.
// Routing only: the software sampler keys off the wrap value itself.
#define GFX_SW_TEX_FILTER_BORDER (1u << 6)

// Resident per-stage texture descriptor (mirror of gfx_sw::TexState).
typedef struct {
  uint64_t base;                          // mip 0 base (TEX_ADDR << 6)
  uint32_t mip_off[VX_TEX_LOD_MAX + 1];   // per-LOD byte offset from base
  uint32_t logdim;                        // {log_h << 16 | log_w} of mip 0
  uint32_t format;                        // VX_TEX_FORMAT_*
  uint32_t filter;                        // mag tap (bit0) | mip-linear (bit1) | mip-enable (bit2) | min tap (bit3) | unnormalized (bit4) | ext format (bit5) | border (bit6) | seamless (bit7) | reduction (bits 9:8) | offset (bits 16+)
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
  uint32_t levels;                        // mip levels the view exposes
  uint32_t res_page;                      // sparse page log2 dims: w | h << 8 | d << 16
  uint32_t res_layer_bits;                // residency bits per array layer
  uint32_t res_level[VX_TEX_LOD_MAX + 1]; // first residency bit of each level
  uint64_t residency;                     // sparse page bitmap; 0 => fully resident
  uint32_t border_t[4];                   // border colour R,G,B,A as the format's channel type: float bits, or integers
  uint32_t aniso;                         // sampler's max anisotropy; 0 or 1 => isotropic
  uint32_t samples;                       // samples per texel of a multisample image; 0 or 1 otherwise
  uint32_t sample_stride;                 // bytes between a multisample image's sample planes
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
  // Depth bounds test. min/max are pre-encoded host-side into the bound depth
  // format's value range (inclusive), so the device compares integers.
  uint32_t depth_bounds_enable;
  uint32_t depth_bounds_min, depth_bounds_max;
  uint32_t stencil_ref_export;
  uint32_t blend_constf[4];            // float bits, RGBA
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
  uint32_t blend_constf[4];            // float bits, RGBA
} gfx_sw_omcolor_t;

// Sample the resident texture (software fallback for vx_tex4). `lod` is integer
// for point/bilinear, fixed-point when the mip-linear filter bit is set. `filter`
// (tap in bit0, mip-linear in bit1) is resolved by the caller — the fragment
// shader picks the min or mag tap per fragment from the sign of its LOD.
uint32_t gfx_tex_sample_sw(const gfx_sw_texstate_t* st,
                           int32_t u, int32_t v, uint32_t lod, uint32_t filter);

// The same sample delivered as four float channels in RGBA order. A float-format
// texture holds values outside [0,1], so it must be decoded and filtered in float
// rather than through the 8-bit working space; every other format delegates to
// gfx_tex_sample_sw and expands the packed word, yielding exactly the [0,1]
// channels the shader unpack would. One entry point for all formats, so the caller
// needs no format test.
void gfx_tex_sample_f32(const gfx_sw_texstate_t* st,
                        int32_t u, int32_t v, uint32_t lod, uint32_t filter,
                        float* out);
// An anisotropic sample: `n` probes of slice `layer` at level `lod`, spread evenly
// along (du, dv) -- the footprint's major axis, S.23 -- around (u, v), averaged.
// The caller picks n and the level from the footprint.
void gfx_tex_sample_aniso_f32(const gfx_sw_texstate_t* st, int32_t u, int32_t v,
                              int32_t du, int32_t dv, uint32_t n, uint32_t layer,
                              uint32_t lod, uint32_t filter, float* out);

// 2D-array sample as four float channels: gfx_tex_sample_array_sw's float twin,
// for a float-format texture (see gfx_tex_sample_f32).
void gfx_tex_sample_array_f32(const gfx_sw_texstate_t* st,
                              int32_t u, int32_t v, uint32_t layer, uint32_t lod,
                              uint32_t filter, float* out);

// Cube sample as four float channels: gfx_tex_sample_cube_sw's float twin, for a
// float-format texture (see gfx_tex_sample_f32).
void gfx_tex_sample_cube_f32(const gfx_sw_texstate_t* st,
                             float sc, float tc, float rc, uint32_t lod, uint32_t filter,
                             float* out);

// Cube-array sample as four float channels: gfx_tex_sample_cube_array_sw's float
// twin, for a float-format texture (see gfx_tex_sample_f32).
void gfx_tex_sample_cube_array_f32(const gfx_sw_texstate_t* st,
                                   float sc, float tc, float rc, uint32_t array_index,
                                   uint32_t lod, uint32_t filter, float* out);

// 3D sample as four float channels: gfx_tex_sample_3d_sw's float twin, for a
// float-format texture (see gfx_tex_sample_f32).
void gfx_tex_sample_3d_f32(const gfx_sw_texstate_t* st,
                           int32_t u, int32_t v, int32_t w, uint32_t lod,
                           uint32_t filter, float* out);

// texelFetch: the four channels of the exact texel at integer (x,y) of integer
// `lod`, as floats in RGBA order -- no wrap/filter/mip. Floats because a
// float-format texture's values lie outside [0,1]; a non-float format yields the
// same [0,1] channels the shader unpack would.
void gfx_tex_fetch_f32(const gfx_sw_texstate_t* st,
                       int32_t x, int32_t y, uint32_t lod, float* out);

// texelFetch for an integer sampler: the four channels of the texel at (x,y) of
// `layer` and `lod` as the stored integers, sign-extended for a signed format.
void gfx_tex_fetch_i32(const gfx_sw_texstate_t* st,
                       int32_t x, int32_t y, uint32_t lod, uint32_t layer,
                       int32_t* out);

// Typeless image read (an input attachment, or a storage image read without a
// format): the texel at `base + layer*img_stride + y*row_stride + x*texel size`,
// sized and decoded by `format` (VX_TEX_FORMAT_*, taken from the image's
// descriptor), as four lanes in RGBA order -- floats for a normalized or float
// format, the stored integers for an integer one.
void gfx_image_load_sw(const void* base, uint32_t row_stride, uint32_t img_stride,
                       int32_t x, int32_t y, uint32_t layer, uint32_t format,
                       uint32_t* out);

// Typeless image write (a storage image written without a format): the four
// lanes encoded by `format` into the texel gfx_image_load_sw would read. Only
// the format's own channels are written.
void gfx_image_store_sw(void* base, uint32_t row_stride, uint32_t img_stride,
                        int32_t x, int32_t y, uint32_t layer, uint32_t format,
                        const uint32_t* lanes);

// texelFetch of a multisample image (texelFetch with a sample index): the texel
// at (x,y) of `layer` in sample plane `sample`, as floats or as the stored
// integers. Sample k's plane starts sample_stride bytes after sample k-1's.
void gfx_tex_fetch_ms_f32(const gfx_sw_texstate_t* st, int32_t x, int32_t y,
                          uint32_t layer, uint32_t sample, float* out);
void gfx_tex_fetch_ms_i32(const gfx_sw_texstate_t* st, int32_t x, int32_t y,
                          uint32_t layer, uint32_t sample, int32_t* out);

// texelFetch on a 2D array: the texel at (x,y) of integer `layer` and `lod`
// (see gfx_tex_fetch_f32).
void gfx_tex_fetch_array_f32(const gfx_sw_texstate_t* st,
                             int32_t x, int32_t y, uint32_t layer, uint32_t lod,
                             float* out);

// textureGather: channel `comp` of the 2x2 footprint at (x,y), base level, packed
// in GL gather order as bytes x | y<<8 | z<<16 | w<<24.
uint32_t gfx_tex_gather_sw(const gfx_sw_texstate_t* st,
                           int32_t x, int32_t y, uint32_t comp);

// Sparse residency code of a sample (nonzero = resident) at the coordinates and
// level a sample of the same op takes: S.23 (u,v[,w]) and resolved lod/filter, or
// texel coordinates and an integer lod when GFX_SW_TEX_RES_FETCH is set; `z` is
// the array layer of a layered view.
#define GFX_SW_TEX_RES_3D    (1u << 0)
#define GFX_SW_TEX_RES_FETCH (1u << 1)
// (u, v, w) are a cube direction's float bits and `z` the cube index.
#define GFX_SW_TEX_RES_CUBE  (1u << 2)
uint32_t gfx_tex_residency_sw(const gfx_sw_texstate_t* st, int32_t u, int32_t v, int32_t w,
                              uint32_t z, uint32_t lod, uint32_t filter, uint32_t flags);

// textureGather at full precision: the four taps of channel `comp` at (x,y) of
// `layer`, in GL gather order, as float bits (float and normalized formats) or the
// stored integers (integer formats). A constant swizzle gives 0 or `one_bits`.
void gfx_tex_gather_f32(const gfx_sw_texstate_t* st, int32_t x, int32_t y, uint32_t comp,
                        uint32_t layer, uint32_t one_bits, uint32_t* out);

// Cube (or cube-array) textureGather: the face the (sc,tc,rc) direction selects,
// in cube `array_index`, as gfx_tex_gather_f32.
void gfx_tex_gather_cube_f32(const gfx_sw_texstate_t* st, float sc, float tc, float rc,
                             uint32_t array_index, uint32_t comp, uint32_t one_bits,
                             uint32_t* out);

// Cube (or cube-array) textureGatherCmp, as gfx_tex_gather_cmp_sw on the face the
// direction selects in cube `array_index`.
uint32_t gfx_tex_gather_cmp_cube_sw(const gfx_sw_texstate_t* st, float sc, float tc,
                                    float rc, uint32_t array_index, uint32_t ref_bits);

// sampler2DArray textureGather on the `layer` slice.
uint32_t gfx_tex_gather_array_sw(const gfx_sw_texstate_t* st,
                                 int32_t x, int32_t y, uint32_t comp,
                                 uint32_t layer);

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

// 2D-array view: sample integer `layer` of the bound array texture. `filter` is
// the tap and mip mode the shader resolved, as for gfx_tex_sample_sw.
uint32_t gfx_tex_sample_array_sw(const gfx_sw_texstate_t* st,
                                 int32_t u, int32_t v, uint32_t layer, uint32_t lod,
                                 uint32_t filter);

// Cube view: sample the face selected by the major axis of the (sc,tc,rc)
// direction vector supplied by the FS.
uint32_t gfx_tex_sample_cube_sw(const gfx_sw_texstate_t* st,
                                float sc, float tc, float rc, uint32_t lod, uint32_t filter);

// Cube-array view: array_index selects the cube, (sc,tc,rc) the face; the slice is
// array_index*6 + face.
uint32_t gfx_tex_sample_cube_array_sw(const gfx_sw_texstate_t* st,
                                      float sc, float tc, float rc,
                                      uint32_t array_index, uint32_t lod, uint32_t filter);

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
// `sref` is the fragment's exported stencil reference, read only when the
// state's stencil_ref_export is set; every merge entry point takes it, and
// returns how many of the fragment's samples passed its depth/stencil test --
// all it covers when that test is off -- which is what an occlusion query
// counts.
uint32_t gfx_om_fragment_sw(const gfx_sw_omstate_t* st, uint32_t covered,
                            uint32_t x, uint32_t y, uint32_t face,
                            uint32_t color, uint32_t depth, uint32_t sref);

// Multisample merge of pixel (x, y) into every attachment. Bit k of
// `sample_mask` is sample k's final coverage -- raster coverage with the
// shader's discard, the sample masks and alpha-to-coverage already folded in --
// and an all-zero mask drops the fragment here so the SIMT caller stays
// straight-line. Each covered sample tests and writes its own depth/stencil slot
// at depth[k], then merges `lanes` (the lane layout of gfx_om_fragment_mrt_wide_sw,
// four per attachment) into its own slot of each of the `num_color` attachments
// `rt` describes, or of `st`'s own attachment when `rt` is null. With `dual` the
// single attachment's second source is lanes 4-7.
//
// Every plane's pitch must already be the multisample row stride (W*samples*bpp),
// which is what msaa_color_addr/msaa_depth_addr index against.
uint32_t gfx_om_fragment_ms_sw(const gfx_sw_omstate_t* st, const gfx_sw_omcolor_t* rt,
                               uint32_t num_color, uint32_t samples, uint32_t sample_mask,
                               uint32_t x, uint32_t y, uint32_t face, const uint32_t* lanes,
                               const uint32_t* depth, uint32_t sref, uint32_t dual);

// Early fragment tests: the depth/stencil test and write against `st` of the
// samples in `sample_mask` of pixel (x, y), each at its own depth[k], ahead of
// the fragment shader (bit 0 alone at one sample). Returns the samples that
// passed; the shader runs for those and the merge after it has its depth/stencil
// stage off.
uint32_t gfx_ds_early_sw(const gfx_sw_omstate_t* st, uint32_t samples, uint32_t sample_mask,
                         uint32_t x, uint32_t y, uint32_t face, const uint32_t* depth);

// MRT software fallback: one shared depth/stencil op against `st`, then a
// per-attachment blend + colour write of colors[k] for each of the `num_color`
// attachments described by `rt`. `covered` drops uncovered fragments here so the
// SIMT caller stays straight-line.
uint32_t gfx_om_fragment_mrt_sw(const gfx_sw_omstate_t* st, const gfx_sw_omcolor_t* rt,
                                uint32_t num_color, uint32_t covered,
                                uint32_t x, uint32_t y, uint32_t face,
                                const uint32_t* colors, uint32_t depth, uint32_t sref);

// Merge for a pass with a wide colour attachment (VX_OM_COLOR_FORMAT_R16F and
// above). `lanes` holds four 32-bit lanes per attachment: a wide attachment takes
// them as its RGBA channels (float bits, or integers for an integer format); any
// other attachment takes lane 0 as the packed ARGB8 word gfx_om_fragment_mrt_sw
// would receive. The single-attachment form merges with `st`'s own colour state.
uint32_t gfx_om_fragment_wide_sw(const gfx_sw_omstate_t* st, uint32_t covered,
                                 uint32_t x, uint32_t y, uint32_t face,
                                 const uint32_t* lanes, uint32_t depth, uint32_t sref);
uint32_t gfx_om_fragment_mrt_wide_sw(const gfx_sw_omstate_t* st, const gfx_sw_omcolor_t* rt,
                                     uint32_t num_color, uint32_t covered,
                                     uint32_t x, uint32_t y, uint32_t face,
                                     const uint32_t* lanes, uint32_t depth, uint32_t sref);

// Dual-source merge into `st`'s single attachment, in the lane layout above:
// lanes 0-3 are the first source and lanes 4-7 the second.
uint32_t gfx_om_fragment_dual_sw(const gfx_sw_omstate_t* st, uint32_t covered,
                                 uint32_t x, uint32_t y, uint32_t face,
                                 const uint32_t* lanes, uint32_t depth, uint32_t sref);

// One covered 2x2 quad produced by the SW fine-rasterizer (fallback for the FF
// RASTER producer). Layout matches the FF frag payload the FS wrapper reads:
// pos_mask (cov_mask[3:0] | quad_x | quad_y | face@31) and the per-fragment edge
// values bcoords[axis*4 + corner] (axis 0..2, corner 0..3; fixed_t<16> raw).
typedef struct {
  uint32_t pos_mask;
  int32_t  bcoords[12];
} gfx_rast_quad_t;

// The multisample twin of gfx_rast_quad_t. sample_masks[c] is corner c's
// per-sample coverage (bit k = sample k), which is what the merge consumes;
// pos_mask carries the quad position only, because the multisample leaf does not
// compute a pixel-centre coverage mask and the two are not interchangeable --
// multisample coverage is "any sample inside", single-sample is "centre inside",
// and they differ exactly at the edges the whole feature exists to improve.
typedef struct {
  uint32_t pos_mask;
  uint32_t sample_masks[4];
  int32_t  bcoords[12];
} gfx_rast_msaa_quad_t;

// The multisample twin of gfx_rast_walk_tile_sw: same traversal, but each quad
// carries per-sample coverage instead of a pixel-centre coverage mask. A quad is
// emitted when any sample of any of its four pixels is covered.
uint32_t gfx_rast_walk_tile_msaa_sw(const void* prim, uint32_t pid,
                                    uint32_t tx, uint32_t ty, uint32_t tile_logsize,
                                    uint32_t scissor_left, uint32_t scissor_top,
                                    uint32_t scissor_right, uint32_t scissor_bottom,
                                    gfx_rast_msaa_quad_t* out, uint32_t max);

// Walk one primitive over one screen tile (origin tx,ty, side 1<<tile_logsize),
// appending every covered quad to out[0..max). Returns the count (capped at max).
// `prim` points at the resident rast_prim_t for `pid`. Used by the SW-raster FS
// wrapper variant (one thread per tile iterates all prims in draw order).
//
// The scissor is the full rect the coverage walk is confined to, not just a
// width/height: it carries the app's VkScissor intersected with the viewport
// rect, which has a non-zero origin whenever the app scissors a sub-rectangle.
// Passing only an extent would silently pin the rect's corner to (0,0) and clip
// the wrong side.
uint32_t gfx_rast_walk_tile_sw(const void* prim, uint32_t pid,
                               uint32_t tx, uint32_t ty, uint32_t tile_logsize,
                               uint32_t scissor_left, uint32_t scissor_top,
                               uint32_t scissor_right, uint32_t scissor_bottom,
                               gfx_rast_quad_t* out, uint32_t max);

#ifdef __cplusplus
}
#endif
