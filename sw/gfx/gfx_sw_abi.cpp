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

// gfx_v2 software-fallback C ABI — thin wrappers over the C++ single-source-
// of-truth headers. Built to LLVM bitcode with the libgfx_sw divergence flags so
// the vortexpipe FS can llvm-link + inline these (om_fragment's divergent merge
// needs the Vortex divergence pass over the whole FS kernel; see libgfx_sw.mk).

#include "gfx_sw_abi.h"
#include "gfx_sw.h"      // gfx_sw::TexState / om_state_t / tex_sample_sw / om_fragment
#include "gfx_frag_rast.h"     // gfx_rast::rast_walk_primitive — SW fine-rasterizer
#include "vx_gfx_abi.h"  // vortex::graphics::rast_prim_t
#include <cstring>
#include <type_traits>

// The C ABI descriptors must be layout-identical to the C++ state structs so the
// reinterpret_cast below is sound (and the host can fill either form).
static_assert(sizeof(gfx_sw_texstate_t) == sizeof(gfx_sw::TexState),
              "gfx_sw_texstate_t must mirror gfx_sw::TexState");
static_assert(sizeof(gfx_sw_omstate_t) == sizeof(gfx_sw::om_state_t),
              "gfx_sw_omstate_t must mirror gfx_sw::om_state_t");
static_assert(sizeof(gfx_sw_omcolor_t) == sizeof(gfx_sw::om_color_t),
              "gfx_sw_omcolor_t must mirror gfx_sw::om_color_t");
static_assert(std::is_trivially_copyable<gfx_sw::TexState>::value &&
              std::is_trivially_copyable<gfx_sw::om_state_t>::value,
              "SW state structs must be POD for the C ABI");

extern "C" uint32_t gfx_tex_sample_sw(const gfx_sw_texstate_t* st,
                                      int32_t u, int32_t v, uint32_t lod,
                                      uint32_t filter) {
  return gfx_sw::tex_sample_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), u, v, lod, filter);
}

extern "C" uint32_t gfx_tex_fetch_sw(const gfx_sw_texstate_t* st,
                                     int32_t x, int32_t y, uint32_t lod) {
  return gfx_sw::tex_fetch_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y, lod);
}

extern "C" uint32_t gfx_tex_gather_sw(const gfx_sw_texstate_t* st,
                                      int32_t x, int32_t y, uint32_t comp) {
  return gfx_sw::tex_gather_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y, comp);
}

extern "C" uint32_t gfx_tex_fetch_array_sw(const gfx_sw_texstate_t* st,
                                           int32_t x, int32_t y, uint32_t layer,
                                           uint32_t lod) {
  return gfx_sw::tex_fetch_array_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y,
                                    layer, lod);
}

extern "C" uint32_t gfx_tex_shadow_sw(const gfx_sw_texstate_t* st,
                                      int32_t x, int32_t y, uint32_t ref_bits,
                                      uint32_t filter) {
  return gfx_sw::tex_shadow_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y,
                               ref_bits, filter);
}

extern "C" uint32_t gfx_tex_shadow_array_sw(const gfx_sw_texstate_t* st,
                                            int32_t x, int32_t y, uint32_t layer,
                                            uint32_t ref_bits, uint32_t filter) {
  return gfx_sw::tex_shadow_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y,
                               ref_bits, filter, layer);
}

extern "C" uint32_t gfx_tex_sample_array_sw(const gfx_sw_texstate_t* st,
                                            int32_t u, int32_t v, uint32_t layer, uint32_t lod) {
  return gfx_sw::tex_sample_sw_array(*reinterpret_cast<const gfx_sw::TexState*>(st), u, v, layer, lod);
}

extern "C" uint32_t gfx_tex_sample_cube_sw(const gfx_sw_texstate_t* st,
                                           float sc, float tc, float rc, uint32_t lod) {
  return gfx_sw::tex_sample_sw_cube(*reinterpret_cast<const gfx_sw::TexState*>(st), sc, tc, rc, lod);
}

extern "C" uint32_t gfx_tex_sample_3d_sw(const gfx_sw_texstate_t* st,
                                         int32_t u, int32_t v, int32_t w, uint32_t lod, uint32_t filter) {
  return gfx_sw::tex_sample_sw_3d(*reinterpret_cast<const gfx_sw::TexState*>(st), u, v, w, lod, filter);
}

extern "C" void gfx_om_fragment_sw(const gfx_sw_omstate_t* st, uint32_t covered,
                                   uint32_t x, uint32_t y, uint32_t face,
                                   uint32_t color, uint32_t depth) {
  if (!covered) {
    return;
  }
  gfx_sw::om_fragment(*reinterpret_cast<const gfx_sw::om_state_t*>(st), x, y, face, color, depth);
}

extern "C" void gfx_om_fragment_mrt_sw(const gfx_sw_omstate_t* st,
                                       const gfx_sw_omcolor_t* rt, uint32_t num_color,
                                       uint32_t covered, uint32_t x, uint32_t y,
                                       uint32_t face, const uint32_t* colors,
                                       uint32_t depth) {
  if (!covered) {
    return;
  }
  gfx_sw::om_fragment_mrt(*reinterpret_cast<const gfx_sw::om_state_t*>(st),
                          reinterpret_cast<const gfx_sw::om_color_t*>(rt), num_color,
                          x, y, face, colors, depth);
}

extern "C" uint32_t gfx_rast_walk_tile_sw(const void* prim, uint32_t pid,
                                          uint32_t tx, uint32_t ty, uint32_t tile_logsize,
                                          uint32_t scissor_w, uint32_t scissor_h,
                                          gfx_rast_quad_t* out, uint32_t max) {
  using namespace gfx_rast;
  const auto* p = reinterpret_cast<const vortex::graphics::rast_prim_t*>(prim);
  RastConfig cfg{ tile_logsize, 0, 0, scissor_w, scissor_h };
  uint32_t count = 0;
  auto emit = [&](uint32_t pos_mask, const vec3e_t* bc, uint32_t) {
    if (count >= max) return;
    gfx_rast_quad_t& q = out[count++];
    q.pos_mask = pos_mask;
    // Pack as the FF frag payload: bcoords[axis*4 + corner].
    for (uint32_t c = 0; c < 4; ++c) {
      q.bcoords[0 * 4 + c] = bc[c].x.data();
      q.bcoords[1 * 4 + c] = bc[c].y.data();
      q.bcoords[2 * 4 + c] = bc[c].z.data();
    }
  };
  // Iterative (non-recursive) leaf walk — SIMT divergence-safe. The SSOT
  // rast_walk_primitive uses a Morton-DFS recursion whose per-lane depth diverges
  // when adjacent tiles cover the primitive differently; the Vortex SIMT
  // reconvergence then drops fragments on the partially-active warp. Visiting the
  // fixed 2x2-quad leaf grid with a uniform-trip loop keeps every lane in lockstep
  // and emits the identical quads (rast_emit_quad does the per-quad trivial-reject
  // + coverage the recursion's leaves would). Order differs (row-major vs Morton)
  // but quads within a primitive don't overlap, so the merged image is identical.
  delta_t delta{
    { p->edges[0].x, p->edges[1].x, p->edges[2].x },
    { p->edges[0].y, p->edges[1].y, p->edges[2].y },
    { CalcEdgeExtents(p->edges[0]), CalcEdgeExtents(p->edges[1]), CalcEdgeExtents(p->edges[2]) }
  };
  const uint32_t leaves = 1u << (tile_logsize - 1);   // 2x2-quad leaves per side
  for (uint32_t ly = 0; ly < leaves; ++ly) {
    for (uint32_t lx = 0; lx < leaves; ++lx) {
      const uint32_t qx = tx + (lx << 1), qy = ty + (ly << 1);
      vec3e_t value{
        EvalEdgeFunction(p->edges[0], (int)qx, (int)qy),
        EvalEdgeFunction(p->edges[1], (int)qx, (int)qy),
        EvalEdgeFunction(p->edges[2], (int)qx, (int)qy)
      };
      rast_emit_quad(cfg, qx, qy, pid, value, delta, emit);
    }
  }
  return count;
}
