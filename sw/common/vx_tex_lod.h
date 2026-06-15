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

// Shared, bit-exact hardware-LOD formula for vx_tex4 quad mode (gfx_v2 P2).
//
// One thread owns a 2x2 fragment quad; its four texture coords are laid out as
//   frag 0 = (x,  y),  frag 1 = (x+1, y),  frag 2 = (x, y+1),  frag 3 = (x+1,y+1)
// in S.23 normalized fixed-point (VX_TEX_FXD_FRAC = 23 fractional bits). The
// integer mip LOD is floor(log2(rho)) where rho is the per-pixel texel-space
// gradient (max over the four partial derivatives, each scaled by its axis's
// log2 dimension). This is the SINGLE source of truth replicated bit-for-bit by
// VX_tex_unit (RTL, via VX_lzc), the SimX TEX model, and the validation kernel.
//
// Integer-mip only (P2): the fractional part / trilinear blend is intentionally
// not produced here — RTL has no two-mip blend datapath (see gfx_v2_tex4_p2.md).

#ifndef VX_TEX_LOD_H
#define VX_TEX_LOD_H

#include <stdint.h>
#include <VX_types.h>

// floor(log2(x)) for x > 0 (position of the leading 1 bit). The RTL computes
// the same value as (WIDTH-1 - VX_lzc(rho)).
static inline uint32_t vx_tex_msb64(uint64_t x) {
  uint32_t r = 0;
  while (x > 1) { x >>= 1; ++r; }
  return r;
}

static inline uint32_t vx_tex_absdiff32(int32_t a, int32_t b) {
  int32_t d = a - b;
  return (uint32_t)(d < 0 ? -d : d);
}

// Integer mip LOD from a 2x2 quad of S.23 normalized coords. logw/logh are the
// log2 texture dimensions (low/high halves of VX_DCR_TEX_LOGDIM).
static inline uint32_t vx_tex_quad_lod(const int32_t u[4], const int32_t v[4],
                                       uint32_t logw, uint32_t logh) {
  // Texel-space partial derivatives = |dcoord| << log2(dim). dx uses frags 0,1;
  // dy uses frags 0,2.
  uint64_t gux = (uint64_t)vx_tex_absdiff32(u[1], u[0]) << logw;  // du/dx
  uint64_t guy = (uint64_t)vx_tex_absdiff32(u[2], u[0]) << logw;  // du/dy
  uint64_t gvx = (uint64_t)vx_tex_absdiff32(v[1], v[0]) << logh;  // dv/dx
  uint64_t gvy = (uint64_t)vx_tex_absdiff32(v[2], v[0]) << logh;  // dv/dy
  uint64_t rho = gux;
  if (guy > rho) rho = guy;
  if (gvx > rho) rho = gvx;
  if (gvy > rho) rho = gvy;
  if (rho == 0) return 0;
  // rho carries VX_TEX_FXD_FRAC fractional bits, so log2(texels/pixel) =
  // floor(log2(rho)) - VX_TEX_FXD_FRAC.
  int32_t lod = (int32_t)vx_tex_msb64(rho) - VX_TEX_FXD_FRAC;
  if (lod < 0) lod = 0;
  if (lod > VX_TEX_LOD_MAX) lod = VX_TEX_LOD_MAX;
  return (uint32_t)lod;
}

#endif // VX_TEX_LOD_H
