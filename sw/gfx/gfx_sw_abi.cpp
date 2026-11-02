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
#include <cstddef>
#include <cstring>
#include <type_traits>

// The C ABI descriptors must be layout-identical to the C++ state structs so the
// reinterpret_cast below is sound (and the host can fill either form).
static_assert(sizeof(gfx_sw_texstate_t) == sizeof(gfx_sw::TexState),
              "gfx_sw_texstate_t must mirror gfx_sw::TexState");
static_assert(offsetof(gfx_sw_texstate_t, sparse) == offsetof(gfx_sw::TexState, sparse),
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

extern "C" void gfx_tex_sample_f32(const gfx_sw_texstate_t* st,
                                   int32_t u, int32_t v, uint32_t lod,
                                   uint32_t filter, float* out) {
  gfx_sw::tex_sample_f32_layer(*reinterpret_cast<const gfx_sw::TexState*>(st), u, v, lod,
                               0, filter, out);
}

extern "C" void gfx_tex_sample_aniso_f32(const gfx_sw_texstate_t* st, int32_t u, int32_t v,
                                         int32_t du, int32_t dv, uint32_t n, uint32_t layer,
                                         uint32_t lod, uint32_t filter, float* out) {
  gfx_sw::tex_sample_aniso_f32(*reinterpret_cast<const gfx_sw::TexState*>(st), u, v, du, dv,
                               n, layer, lod, filter, out);
}

extern "C" void gfx_tex_sample_array_f32(const gfx_sw_texstate_t* st,
                                         int32_t u, int32_t v, uint32_t layer,
                                         uint32_t lod, uint32_t filter, float* out) {
  gfx_sw::tex_sample_f32_array(*reinterpret_cast<const gfx_sw::TexState*>(st), u, v,
                               layer, lod, filter, out);
}

extern "C" void gfx_tex_sample_cube_f32(const gfx_sw_texstate_t* st,
                                        float sc, float tc, float rc, uint32_t lod,
                                        uint32_t filter, float* out) {
  gfx_sw::tex_sample_f32_cube(*reinterpret_cast<const gfx_sw::TexState*>(st), sc, tc, rc,
                              lod, filter, out);
}

extern "C" void gfx_tex_sample_cube_array_f32(const gfx_sw_texstate_t* st,
                                              float sc, float tc, float rc,
                                              uint32_t array_index, uint32_t lod,
                                              uint32_t filter, float* out) {
  gfx_sw::tex_sample_f32_cube_array(*reinterpret_cast<const gfx_sw::TexState*>(st),
                                    sc, tc, rc, array_index, lod, filter, out);
}

extern "C" void gfx_tex_sample_3d_f32(const gfx_sw_texstate_t* st,
                                      int32_t u, int32_t v, int32_t w, uint32_t lod,
                                      uint32_t filter, float* out) {
  gfx_sw::tex_sample_f32_3d(*reinterpret_cast<const gfx_sw::TexState*>(st), u, v, w, lod,
                            filter, out);
}

extern "C" void gfx_tex_fetch_f32(const gfx_sw_texstate_t* st,
                                  int32_t x, int32_t y, uint32_t lod, float* out) {
  gfx_sw::tex_fetch_f32(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y, lod, out);
}

extern "C" void gfx_tex_fetch_array_f32(const gfx_sw_texstate_t* st,
                                        int32_t x, int32_t y, uint32_t layer,
                                        uint32_t lod, float* out) {
  gfx_sw::tex_fetch_array_f32(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y,
                              layer, lod, out);
}

// A sample past the image's reads the zero texel, as a texel outside it does:
// the fetch is steered past the view's last level.
static inline gfx_sw::TexState tex_ms_plane(const gfx_sw_texstate_t* st, uint32_t sample,
                                            uint32_t* lod) {
  gfx_sw::TexState s = *reinterpret_cast<const gfx_sw::TexState*>(st);
  const uint32_t samples = s.samples ? s.samples : 1u;
  *lod = sample < samples ? 0u : (uint32_t)VX_TEX_LOD_MAX + 1u;
  s.base += (uint64_t)(sample < samples ? sample : 0u) * s.sample_stride;
  return s;
}

extern "C" void gfx_tex_fetch_ms_f32(const gfx_sw_texstate_t* st, int32_t x, int32_t y,
                                     uint32_t layer, uint32_t sample, float* out) {
  uint32_t lod;
  const gfx_sw::TexState s = tex_ms_plane(st, sample, &lod);
  gfx_sw::tex_fetch_array_f32(s, x, y, layer, lod, out);
}

extern "C" void gfx_tex_fetch_ms_i32(const gfx_sw_texstate_t* st, int32_t x, int32_t y,
                                     uint32_t layer, uint32_t sample, int32_t* out) {
  uint32_t lod;
  const gfx_sw::TexState s = tex_ms_plane(st, sample, &lod);
  gfx_sw::tex_fetch_i32(s, x, y, lod, layer, out);
}

extern "C" void gfx_tex_fetch_i32(const gfx_sw_texstate_t* st,
                                  int32_t x, int32_t y, uint32_t lod,
                                  uint32_t layer, int32_t* out) {
  gfx_sw::tex_fetch_i32(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y, lod,
                        layer, out);
}

extern "C" void gfx_tex_gather_f32(const gfx_sw_texstate_t* st, int32_t x, int32_t y,
                                   uint32_t comp, uint32_t layer, uint32_t one_bits,
                                   uint32_t* out) {
  gfx_sw::tex_gather_f32(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y, comp, layer,
                         one_bits, out);
}

extern "C" void gfx_tex_gather_cube_f32(const gfx_sw_texstate_t* st, float sc, float tc,
                                        float rc, uint32_t array_index, uint32_t comp,
                                        uint32_t one_bits, uint32_t* out) {
  gfx_sw::tex_gather_cube_f32(*reinterpret_cast<const gfx_sw::TexState*>(st), sc, tc, rc,
                              array_index, comp, one_bits, out);
}

extern "C" uint32_t gfx_tex_gather_cmp_cube_sw(const gfx_sw_texstate_t* st, float sc,
                                               float tc, float rc, uint32_t array_index,
                                               uint32_t ref_bits) {
  return gfx_sw::tex_gather_cmp_cube_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), sc, tc,
                                        rc, array_index, ref_bits);
}

extern "C" uint32_t gfx_tex_residency_sw(const gfx_sw_texstate_t* st, int32_t u, int32_t v,
                                         int32_t w, uint32_t z, uint32_t lod, uint32_t filter,
                                         uint32_t flags) {
  const gfx_sw::TexState& s = *reinterpret_cast<const gfx_sw::TexState*>(st);
  if (flags & GFX_SW_TEX_RES_CUBE) {
    float sc, tc, rc;
    __builtin_memcpy(&sc, &u, 4);
    __builtin_memcpy(&tc, &v, 4);
    __builtin_memcpy(&rc, &w, 4);
    int32_t fu, fv;
    const uint32_t face = gfx_sw::cube_face_uv(sc, tc, rc, &fu, &fv);
    return gfx_sw::tex_residency(s, fu, fv, 0, gfx_sw::cube_array_clamp(s, z) * 6u + face, lod,
                                 filter, false, false);
  }
  return gfx_sw::tex_residency(s, u, v, w, z, lod, filter, (flags & GFX_SW_TEX_RES_3D) != 0,
                               (flags & GFX_SW_TEX_RES_FETCH) != 0);
}

extern "C" uint32_t gfx_tex_gather_sw(const gfx_sw_texstate_t* st,
                                      int32_t x, int32_t y, uint32_t comp) {
  return gfx_sw::tex_gather_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y, comp);
}

extern "C" uint32_t gfx_tex_gather_array_sw(const gfx_sw_texstate_t* st,
                                            int32_t x, int32_t y, uint32_t comp,
                                            uint32_t layer) {
  return gfx_sw::tex_gather_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y,
                               comp, layer);
}

extern "C" uint32_t gfx_tex_shadow_sw(const gfx_sw_texstate_t* st,
                                      int32_t x, int32_t y, uint32_t ref_bits,
                                      uint32_t filter, uint32_t lod) {
  return gfx_sw::tex_shadow_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y,
                               ref_bits, filter, 0u, lod);
}

extern "C" uint32_t gfx_tex_shadow_array_sw(const gfx_sw_texstate_t* st,
                                            int32_t x, int32_t y, uint32_t layer,
                                            uint32_t ref_bits, uint32_t filter,
                                            uint32_t lod) {
  return gfx_sw::tex_shadow_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), x, y,
                               ref_bits, filter, layer, lod);
}

extern "C" uint32_t gfx_tex_shadow_cube_sw(const gfx_sw_texstate_t* st,
                                           float sc, float tc, float rc,
                                           uint32_t ref_bits, uint32_t filter,
                                           uint32_t lod) {
  return gfx_sw::tex_shadow_cube_sw(*reinterpret_cast<const gfx_sw::TexState*>(st),
                                    sc, tc, rc, ref_bits, filter, lod);
}

extern "C" uint32_t gfx_tex_gather_cmp_sw(const gfx_sw_texstate_t* st,
                                          int32_t x, int32_t y, uint32_t ref_bits) {
  return gfx_sw::tex_gather_cmp_sw(*reinterpret_cast<const gfx_sw::TexState*>(st),
                                   x, y, ref_bits);
}

extern "C" uint32_t gfx_tex_gather_cmp_array_sw(const gfx_sw_texstate_t* st,
                                                int32_t x, int32_t y, uint32_t ref_bits,
                                                uint32_t layer) {
  return gfx_sw::tex_gather_cmp_sw(*reinterpret_cast<const gfx_sw::TexState*>(st),
                                   x, y, ref_bits, layer);
}

extern "C" uint32_t gfx_tex_sample_cube_array_sw(const gfx_sw_texstate_t* st,
                                                 float sc, float tc, float rc,
                                                 uint32_t array_index, uint32_t lod,
                                                 uint32_t filter) {
  return gfx_sw::tex_sample_cube_array_sw(
      *reinterpret_cast<const gfx_sw::TexState*>(st), sc, tc, rc, array_index, lod, filter);
}

extern "C" uint32_t gfx_tex_shadow_cube_array_sw(const gfx_sw_texstate_t* st,
                                                 float sc, float tc, float rc,
                                                 uint32_t array_index,
                                                 uint32_t ref_bits, uint32_t filter,
                                                 uint32_t lod) {
  return gfx_sw::tex_shadow_cube_array_sw(
      *reinterpret_cast<const gfx_sw::TexState*>(st), sc, tc, rc, array_index,
      ref_bits, filter, lod);
}

extern "C" uint32_t gfx_tex_sample_array_sw(const gfx_sw_texstate_t* st,
                                            int32_t u, int32_t v, uint32_t layer, uint32_t lod,
                                            uint32_t filter) {
  return gfx_sw::tex_sample_sw_array(*reinterpret_cast<const gfx_sw::TexState*>(st), u, v,
                                     layer, lod, filter);
}

extern "C" uint32_t gfx_tex_sample_cube_sw(const gfx_sw_texstate_t* st,
                                           float sc, float tc, float rc, uint32_t lod,
                                           uint32_t filter) {
  return gfx_sw::tex_sample_sw_cube(*reinterpret_cast<const gfx_sw::TexState*>(st), sc, tc,
                                    rc, lod, filter);
}

extern "C" uint32_t gfx_tex_sample_3d_sw(const gfx_sw_texstate_t* st,
                                         int32_t u, int32_t v, int32_t w, uint32_t lod, uint32_t filter) {
  return gfx_sw::tex_sample_sw_3d(*reinterpret_cast<const gfx_sw::TexState*>(st), u, v, w, lod, filter);
}

extern "C" uint32_t gfx_om_fragment_sw(const gfx_sw_omstate_t* st, uint32_t covered,
                                       uint32_t x, uint32_t y, uint32_t face,
                                       uint32_t color, uint32_t depth, uint32_t sref) {
  if (!covered) {
    return 0;
  }
  return gfx_sw::om_fragment(*reinterpret_cast<const gfx_sw::om_state_t*>(st), x, y, face, color, depth,
                             sref);
}

extern "C" uint32_t gfx_om_fragment_ms_sw(const gfx_sw_omstate_t* st, const gfx_sw_omcolor_t* rt,
                                          uint32_t num_color, uint32_t samples, uint32_t sample_mask,
                                          uint32_t x, uint32_t y, uint32_t face, const uint32_t* lanes,
                                          const uint32_t* depth, uint32_t sref, uint32_t dual) {
  if (!sample_mask) {
    return 0;
  }
  return gfx_sw::om_fragment_ms(*reinterpret_cast<const gfx_sw::om_state_t*>(st),
                                reinterpret_cast<const gfx_sw::om_color_t*>(rt), num_color, samples,
                                sample_mask, x, y, face, lanes, depth, sref, dual != 0);
}

extern "C" uint32_t gfx_ds_early_sw(const gfx_sw_omstate_t* st, uint32_t samples,
                                    uint32_t sample_mask, uint32_t x, uint32_t y,
                                    uint32_t face, const uint32_t* depth) {
  if (!sample_mask) {
    return 0;
  }
  return gfx_sw::ds_early(*reinterpret_cast<const gfx_sw::om_state_t*>(st), samples, x, y,
                          face, sample_mask, depth);
}

extern "C" uint32_t gfx_om_fragment_mrt_sw(const gfx_sw_omstate_t* st,
                                           const gfx_sw_omcolor_t* rt, uint32_t num_color,
                                           uint32_t covered, uint32_t x, uint32_t y,
                                           uint32_t face, const uint32_t* colors,
                                           uint32_t depth, uint32_t sref) {
  if (!covered) {
    return 0;
  }
  return gfx_sw::om_fragment_mrt(*reinterpret_cast<const gfx_sw::om_state_t*>(st),
                                 reinterpret_cast<const gfx_sw::om_color_t*>(rt), num_color,
                                 x, y, face, colors, depth, sref);
}

extern "C" uint32_t gfx_om_fragment_wide_sw(const gfx_sw_omstate_t* st, uint32_t covered,
                                            uint32_t x, uint32_t y, uint32_t face,
                                            const uint32_t* lanes, uint32_t depth, uint32_t sref) {
  if (!covered) {
    return 0;
  }
  return gfx_sw::om_fragment_wide(*reinterpret_cast<const gfx_sw::om_state_t*>(st), x, y, face,
                                  lanes, depth, sref);
}

extern "C" uint32_t gfx_om_fragment_dual_sw(const gfx_sw_omstate_t* st, uint32_t covered,
                                            uint32_t x, uint32_t y, uint32_t face,
                                            const uint32_t* lanes, uint32_t depth, uint32_t sref) {
  if (!covered) {
    return 0;
  }
  return gfx_sw::om_fragment_dual(*reinterpret_cast<const gfx_sw::om_state_t*>(st), x, y, face,
                                  lanes, depth, sref);
}

extern "C" uint32_t gfx_om_fragment_mrt_wide_sw(const gfx_sw_omstate_t* st,
                                                const gfx_sw_omcolor_t* rt, uint32_t num_color,
                                                uint32_t covered, uint32_t x, uint32_t y,
                                                uint32_t face, const uint32_t* lanes,
                                                uint32_t depth, uint32_t sref) {
  if (!covered) {
    return 0;
  }
  return gfx_sw::om_fragment_mrt_wide(*reinterpret_cast<const gfx_sw::om_state_t*>(st),
                                      reinterpret_cast<const gfx_sw::om_color_t*>(rt), num_color,
                                      x, y, face, lanes, depth, sref);
}

extern "C" uint32_t gfx_rast_walk_tile_msaa_sw(const void* prim, uint32_t pid,
                                               uint32_t tx, uint32_t ty,
                                               uint32_t tile_logsize,
                                               uint32_t scissor_left, uint32_t scissor_top,
                                               uint32_t scissor_right, uint32_t scissor_bottom,
                                               gfx_rast_msaa_quad_t* out, uint32_t max) {
  using namespace gfx_rast;
  const auto* p = reinterpret_cast<const vortex::graphics::rast_prim_t*>(prim);
  const RastConfig cfg = rast_prim_config(
      { tile_logsize, scissor_left, scissor_top, scissor_right, scissor_bottom },
      p->scissor_x, p->scissor_y);
  uint32_t count = 0;
  auto emit = [&](uint32_t pos_mask, const vec3e_t* bc, const uint32_t* smask, uint32_t) {
    if (count >= max) {
      return;
    }
    gfx_rast_msaa_quad_t& q = out[count++];
    q.pos_mask = pos_mask;
    for (uint32_t c = 0; c < 4; ++c) {
      q.sample_masks[c] = smask[c];
      // Pack as the FF frag payload: bcoords[axis*4 + corner].
      q.bcoords[0 * 4 + c] = bc[c].x.data();
      q.bcoords[1 * 4 + c] = bc[c].y.data();
      q.bcoords[2 * 4 + c] = bc[c].z.data();
    }
  };
  // Same iterative leaf walk as the single-sample entry point above, and for the
  // same reason: the SSOT Morton-DFS recursion diverges per lane and the SIMT
  // reconvergence then drops fragments on a partially-active warp.
  delta_t delta{
    { p->edges[0].x, p->edges[1].x, p->edges[2].x },
    { p->edges[0].y, p->edges[1].y, p->edges[2].y },
    { CalcEdgeExtents(p->edges[0]), CalcEdgeExtents(p->edges[1]), CalcEdgeExtents(p->edges[2]) }
  };
  const uint32_t leaves = 1u << (tile_logsize - 1);
  for (uint32_t ly = 0; ly < leaves; ++ly) {
    for (uint32_t lx = 0; lx < leaves; ++lx) {
      const uint32_t qx = tx + (lx << 1), qy = ty + (ly << 1);
      vec3e_t value{
        EvalEdgeFunction(p->edges[0], (int)qx, (int)qy),
        EvalEdgeFunction(p->edges[1], (int)qx, (int)qy),
        EvalEdgeFunction(p->edges[2], (int)qx, (int)qy)
      };
      rast_emit_quad_msaa(cfg, qx, qy, pid, value, delta, emit);
    }
  }
  return count;
}

extern "C" uint32_t gfx_rast_walk_tile_sw(const void* prim, uint32_t pid,
                                          uint32_t tx, uint32_t ty, uint32_t tile_logsize,
                                          uint32_t scissor_left, uint32_t scissor_top,
                                          uint32_t scissor_right, uint32_t scissor_bottom,
                                          gfx_rast_quad_t* out, uint32_t max) {
  using namespace gfx_rast;
  const auto* p = reinterpret_cast<const vortex::graphics::rast_prim_t*>(prim);
  const RastConfig cfg = rast_prim_config(
      { tile_logsize, scissor_left, scissor_top, scissor_right, scissor_bottom },
      p->scissor_x, p->scissor_y);
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
