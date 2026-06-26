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

// C ABI for the gfx_v2 on-device software fallback (gfx_v2 §5 driver routing).
// The vortexpipe FS (NIR→LLVM, C) emits calls to these entry points when a unit
// is routed to software; the host builds the POD descriptors from the bound
// pipeline state and passes their resident device pointers via the kernel arg.
// The implementations (gfx_sw_abi.cpp) are thin wrappers over the C++ single-
// source-of-truth headers (tex_sample.h / gfx_sw.h / rast_sw.h), so the SW path
// the driver runs is bit-identical to the FF model and the unit tests (§7).
//
// This header is plain C (no C++), so the C mesa driver can include it. The POD
// descriptors mirror gfx_sw::TexState / gfx_sw::om_state_t exactly; gfx_sw_abi.cpp
// static_asserts the layouts match.

#ifndef _GFX_SW_ABI_H_
#define _GFX_SW_ABI_H_

#include <stdint.h>
#include <VX_types.h>   // VX_TEX_LOD_MAX

#ifdef __cplusplus
extern "C" {
#endif

// Resident per-stage texture descriptor (mirror of gfx_sw::TexState).
typedef struct {
  uint64_t base;                          // mip 0 base (TEX_ADDR << 6)
  uint32_t mip_off[VX_TEX_LOD_MAX + 1];   // per-LOD byte offset from base
  uint32_t logdim;                        // {log_h << 16 | log_w} of mip 0
  uint32_t format;                        // VX_TEX_FORMAT_*
  uint32_t filter;                        // mag/min (bit 0) | mip-linear (bit 1)
  uint32_t wrap;                          // {wrap_v << 16 | wrap_u}
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
  uint32_t depth_enabled, stencil_enabled[2], blend_enabled;
  uint32_t cbuf_writemask;
  uint32_t color_read, color_write;
} gfx_sw_omstate_t;

// Sample the resident texture (software fallback for vx_tex4). `lod` is integer
// for point/bilinear, fixed-point when the mip-linear filter bit is set.
uint32_t gfx_tex_sample_sw(const gfx_sw_texstate_t* st,
                           int32_t u, int32_t v, uint32_t lod);

// Merge one fragment (software fallback for vx_om4): depth/stencil test + blend
// + ROP at pixel (x, y) for face (0=front, 1=back) using the resident om state.
void gfx_om_fragment_sw(const gfx_sw_omstate_t* st,
                        uint32_t x, uint32_t y, uint32_t face,
                        uint32_t color, uint32_t depth);

#ifdef __cplusplus
}
#endif

#endif // _GFX_SW_ABI_H_
