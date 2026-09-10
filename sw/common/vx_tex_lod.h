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

// The single source of truth for texture LOD selection, shared bit-for-bit by
// the software sampler, the host reference, and the fragment shader.
//
// A quad's four texture coords are laid out as
//   frag 0 = (x,  y),  frag 1 = (x+1, y),  frag 2 = (x, y+1),  frag 3 = (x+1,y+1)
// in S.23 normalized fixed-point (TEX_FXD_FRAC = 23 fractional bits). The
// integer mip LOD is floor(log2(rho)), where rho is the per-pixel texel-space
// gradient: the max over the four partial derivatives, each scaled by its axis's
// log2 dimension.
//
// Integer mip only. The texture unit samples one level and has no inter-level
// blend, so a fractional LOD has nowhere to go; software that wants mip-linear
// samples two levels and lerps them itself.

#pragma once

#include <stdint.h>
#include <VX_types.h>
#include "vx_gfx_abi.h"

// floor(log2(x)) for x > 0 (position of the leading 1 bit). The RTL computes
// the same value as (WIDTH-1 - VX_lzc(rho)).
//
// Branchless binary search rather than a shift-down loop: this runs per quad in
// the fragment shader's inner loop, and rv32imaf has no Zbb clz to lower to.
// Bit-identical to the loop for all x (both return 0 at x == 0).
static inline uint32_t vx_tex_msb64(uint64_t x) {
  uint32_t r = 0;
  if (x >> 32) { r += 32; x >>= 32; }
  if (x >> 16) { r += 16; x >>= 16; }
  if (x >> 8)  { r += 8;  x >>= 8;  }
  if (x >> 4)  { r += 4;  x >>= 4;  }
  if (x >> 2)  { r += 2;  x >>= 2;  }
  if (x >> 1)  { r += 1; }
  return r;
}

static inline uint32_t vx_tex_absdiff32(int32_t a, int32_t b) {
  int32_t d = a - b;
  return (uint32_t)(d < 0 ? -d : d);
}

// The reduction, shared by every caller. The four texel-space partial
// derivatives are already scaled by their axis's log2 dimension; the LOD is
// floor(log2(max)) with the fixed-point bias removed. Whoever holds the four
// coords -- one thread with a quad, or four lanes with a pixel each -- produces
// the same four gradients and lands on the same LOD.
static inline uint32_t vx_tex_lod_from_grads(uint64_t gux, uint64_t guy,
                                             uint64_t gvx, uint64_t gvy) {
  uint64_t rho = gux;
  if (guy > rho) rho = guy;
  if (gvx > rho) rho = gvx;
  if (gvy > rho) rho = gvy;
  if (rho == 0) {
    return 0;
  }
  // rho carries TEX_FXD_FRAC fractional bits, so log2(texels/pixel) =
  // floor(log2(rho)) - TEX_FXD_FRAC.
  int32_t lod = (int32_t)vx_tex_msb64(rho) - TEX_FXD_FRAC;
  if (lod < 0) {
    lod = 0;
  }
  if (lod > VX_TEX_LOD_MAX) {
    lod = VX_TEX_LOD_MAX;
  }
  return (uint32_t)lod;
}

// Integer mip LOD from a 2x2 quad of S.23 normalized coords held by ONE caller.
// The software sampler and the host reference own four pixels at once, so this
// form stays. logw/logh are the log2 texture dimensions (low/high halves of
// VX_DCR_TEX_LOGDIM). dx uses frags 0,1; dy uses frags 0,2.
static inline uint32_t vx_tex_quad_lod(const int32_t u[4], const int32_t v[4],
                                       uint32_t logw, uint32_t logh) {
  return vx_tex_lod_from_grads(
    (uint64_t)vx_tex_absdiff32(u[1], u[0]) << logw,   // du/dx
    (uint64_t)vx_tex_absdiff32(u[2], u[0]) << logw,   // du/dy
    (uint64_t)vx_tex_absdiff32(v[1], v[0]) << logh,   // dv/dx
    (uint64_t)vx_tex_absdiff32(v[2], v[0]) << logh);  // dv/dy
}

#ifdef __VORTEX__
#include <vx_intrinsics.h>

// Integer mip LOD from the lane's OWN coords plus its quad neighbours'. The four
// gradients are the same four numbers as the quad form above; only who holds
// them moves, so the two are bit-identical by construction.
//
// Every lane of the quad must be active, or the neighbour reads return the
// reader's own value and the LOD collapses to 0.
static inline uint32_t vx_tex_auto_lod(int32_t u, int32_t v,
                                       uint32_t logw, uint32_t logh) {
  return vx_tex_lod_from_grads(
    (uint64_t)vx_tex_absdiff32(vx_quad_ddx_i32(u), 0) << logw,   // du/dx
    (uint64_t)vx_tex_absdiff32(vx_quad_ddy_i32(u), 0) << logw,   // du/dy
    (uint64_t)vx_tex_absdiff32(vx_quad_ddx_i32(v), 0) << logh,   // dv/dx
    (uint64_t)vx_tex_absdiff32(vx_quad_ddy_i32(v), 0) << logh);  // dv/dy
}
#endif
