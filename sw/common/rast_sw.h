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

// gfx_v2 rasterizer coverage math — the single source of truth shared by the
// host fixed-function RASTER model (sw/common/gfx_render.cpp Rasterizer) and the
// on-device SIMT software fallback (sw/common/gfx_sw.h rast_walk_sw). The
// recursive tile→quad edge-equation walk here produces the exact same covered
// quads (pos_mask + per-fragment edge values) on both sides, so the SW path
// matches the FF unit bit-for-bit because it IS the same code
// (gfx_v2_software_fallback.md §4.1 / §7).
//
// Freestanding: edge coefficients are vx_gfx_abi.h fixed_t<16> (self-contained,
// no <cmath>), so this compiles for the baremetal device. The emit callback is a
// template so the FF model forwards to its ShaderCB while the device path stages
// quads into the graphics register window — no shared allocation/ABI coupling.

#ifndef _RAST_SW_H_
#define _RAST_SW_H_

#include <stdint.h>
#include <VX_types.h>
#include <vx_gfx_abi.h>   // vec3e_t / FloatE (fixed_t<16>)

namespace gfx_rast {

using vortex::graphics::vec3e_t;
using vortex::graphics::FloatE;

// Cached scissor + tile geometry for one primitive walk (the SW mirror of
// RasterDCRS scissor + the RASTER tile/block log sizes).
struct RastConfig {
  uint32_t tile_logsize;     // top-level tile = 1 << tile_logsize pixels per side
  uint32_t scissor_left;
  uint32_t scissor_top;
  uint32_t scissor_right;
  uint32_t scissor_bottom;
};

// Per-edge dx/dy steps and trivial-reject extents for a primitive.
struct delta_t {
  vec3e_t dx;
  vec3e_t dy;
  vec3e_t extents;
};

static inline FloatE fx_zero() { return FloatE(0); }

// Edge function value at pixel (x, y): e.x*x + e.y*y + e.z (fixed-point).
static inline FloatE EvalEdgeFunction(const vec3e_t& e, int x, int y) {
  return (e.x * x) + (e.y * y) + e.z;
}

// Largest positive corner contribution of one edge over a unit step — used to
// trivially reject a tile when even its best corner is outside the edge.
static inline FloatE CalcEdgeExtents(const vec3e_t& e) {
  FloatE z = fx_zero();
  return (e.y >= z) ? ((e.x >= z) ? (e.x + e.y) : e.y)
                    : ((e.x >= z) ? e.x : z);
}

static inline FloatE ShiftLeft(const FloatE& value, uint32_t dist) {
  return value << static_cast<int>(dist);
}

// Emit one covered 2x2 quad. mask = per-fragment coverage (frag p = j*2+i);
// bcoords[p] = edge values at that fragment. quad origin in pixels = (x, y).
template <typename Emit>
static inline void rast_emit_quad(const RastConfig& cfg, uint32_t x, uint32_t y,
                                  uint32_t pid, const vec3e_t& edges,
                                  const delta_t& delta, Emit&& emit) {
  FloatE z = fx_zero();
  // Trivial-reject the quad.
  if ((edges.x + ShiftLeft(delta.extents.x, 1)) < z
   || (edges.y + ShiftLeft(delta.extents.y, 1)) < z
   || (edges.z + ShiftLeft(delta.extents.z, 1)) < z)
    return;

  vec3e_t bcoords[4];
  uint32_t mask = 0;
  for (uint32_t j = 0; j < 2; ++j) {
    for (uint32_t i = 0; i < 2; ++i) {
      FloatE ee0 = edges.x + (delta.dx.x * int(i)) + (delta.dy.x * int(j));
      FloatE ee1 = edges.y + (delta.dx.y * int(i)) + (delta.dy.y * int(j));
      FloatE ee2 = edges.z + (delta.dx.z * int(i)) + (delta.dy.z * int(j));
      bool coverage = (ee0 >= z && ee1 >= z && ee2 >= z
                    && (x + i) >= cfg.scissor_left && (x + i) < cfg.scissor_right
                    && (y + j) >= cfg.scissor_top  && (y + j) < cfg.scissor_bottom);
      uint32_t p = j * 2 + i;
      mask |= (uint32_t(coverage) << p);
      bcoords[p].x = ee0;
      bcoords[p].y = ee1;
      bcoords[p].z = ee2;
    }
  }
  if (mask) {
    uint32_t quad_x = x / 2;
    uint32_t quad_y = y / 2;
    uint32_t pos_mask = (quad_y << (4 + VX_RASTER_DIM_BITS - 1)) | (quad_x << 4) | mask;
    emit(pos_mask, bcoords, pid);
  }
}

// ── MSAA (§6): 4x multisample coverage ───────────────────────────────────────
// 4x rotated-grid sample pattern (the D3D/GL-standard 4X positions), expressed
// as pixel-relative sub-sample offsets in fixed_t<16> (1/16-pixel grid). The
// offsets are added to the integer sample point the single-sample path uses, so
// sample evaluation reuses the per-pixel edge values from the walk.
//   s0 (6,2) s1 (14,6) s2 (2,10) s3 (10,14)  [in 1/16 of a pixel]
static constexpr int RAST_MSAA_SAMPLES = 4;

struct SampleOff { FloatE dx, dy; };

static inline const SampleOff* rast_msaa_offsets() {
  // n/16 in fixed_t<16> is raw n<<12.
  static const SampleOff k[RAST_MSAA_SAMPLES] = {
    { FloatE::make( 6 << 12), FloatE::make( 2 << 12) },
    { FloatE::make(14 << 12), FloatE::make( 6 << 12) },
    { FloatE::make( 2 << 12), FloatE::make(10 << 12) },
    { FloatE::make(10 << 12), FloatE::make(14 << 12) },
  };
  return k;
}

// Per-pixel MSAA coverage mask: bit k set iff sample k is inside all three
// edges. `edges` are the primitive's edge coefficients ({a,b,c} each); `base`
// is the edge-function values at the pixel's integer sample point (i.e. the
// bcoords entry the walk emits). The sub-sample value is
// base_i + a_i*dx + b_i*dy = f_i(x+dx, y+dy).
static inline uint32_t rast_sample_mask(const vec3e_t edges[3], const vec3e_t& base) {
  FloatE z = fx_zero();
  const SampleOff* off = rast_msaa_offsets();
  uint32_t mask = 0;
  for (int k = 0; k < RAST_MSAA_SAMPLES; ++k) {
    FloatE e0 = base.x + edges[0].x * off[k].dx + edges[0].y * off[k].dy;
    FloatE e1 = base.y + edges[1].x * off[k].dx + edges[1].y * off[k].dy;
    FloatE e2 = base.z + edges[2].x * off[k].dx + edges[2].y * off[k].dy;
    if (e0 >= z && e1 >= z && e2 >= z)
      mask |= (1u << k);
  }
  return mask;
}

// MSAA leaf: emit a quad whenever ANY sample of ANY fragment is covered (so a
// pixel whose center is outside the triangle but whose samples are inside is not
// dropped — the §6 conservative-coverage requirement). Calls
// emit(pos_mask, bcoords[4], sample_masks[4], pid) — sample_masks[p] = the 4-bit
// MSAA coverage of fragment p (0 for uncovered fragments).
template <typename Emit>
static inline void rast_emit_quad_msaa(const RastConfig& cfg, uint32_t x, uint32_t y,
                                       uint32_t pid, const vec3e_t& edges,
                                       const delta_t& delta, Emit&& emit) {
  FloatE z = fx_zero();
  // Conservative trivial-reject (footprint [x,x+2) covers all sample positions).
  if ((edges.x + ShiftLeft(delta.extents.x, 1)) < z
   || (edges.y + ShiftLeft(delta.extents.y, 1)) < z
   || (edges.z + ShiftLeft(delta.extents.z, 1)) < z)
    return;

  // Reconstruct the edge coefficients ({a,b}) from the deltas for sampling.
  vec3e_t ecoef[3] = {
    { delta.dx.x, delta.dy.x, z },
    { delta.dx.y, delta.dy.y, z },
    { delta.dx.z, delta.dy.z, z }
  };

  vec3e_t bcoords[4];
  uint32_t sample_masks[4];
  uint32_t any = 0;
  for (uint32_t j = 0; j < 2; ++j) {
    for (uint32_t i = 0; i < 2; ++i) {
      uint32_t p = j * 2 + i;
      vec3e_t base{ edges.x + (delta.dx.x * int(i)) + (delta.dy.x * int(j)),
                    edges.y + (delta.dx.y * int(i)) + (delta.dy.y * int(j)),
                    edges.z + (delta.dx.z * int(i)) + (delta.dy.z * int(j)) };
      bcoords[p] = base;
      bool in_scissor = (x + i) >= cfg.scissor_left && (x + i) < cfg.scissor_right
                     && (y + j) >= cfg.scissor_top  && (y + j) < cfg.scissor_bottom;
      uint32_t m = in_scissor ? rast_sample_mask(ecoef, base) : 0u;
      sample_masks[p] = m;
      any |= m;
    }
  }
  if (any) {
    uint32_t quad_x = x / 2, quad_y = y / 2;
    uint32_t pos_mask = (quad_y << (4 + VX_RASTER_DIM_BITS - 1)) | (quad_x << 4);
    emit(pos_mask, bcoords, sample_masks, pid);
  }
}

// Recursive Morton-DFS tile subdivision: reject tiles outside the primitive,
// recurse to four sub-tiles, and run `leaf(cfg, x, y, pid, edges, delta)` at each
// 2x2-quad leaf. The leaf decides per-quad coverage (single-sample or MSAA).
template <typename Leaf>
static inline void rast_walk_tile(const RastConfig& cfg, uint32_t tileLogSize,
                                  uint32_t x, uint32_t y, uint32_t pid,
                                  const vec3e_t& edges, const delta_t& delta,
                                  Leaf&& leaf) {
  FloatE z = fx_zero();
  // Trivial-reject the whole tile.
  if ((edges.x + ShiftLeft(delta.extents.x, tileLogSize)) < z
   || (edges.y + ShiftLeft(delta.extents.y, tileLogSize)) < z
   || (edges.z + ShiftLeft(delta.extents.z, tileLogSize)) < z)
    return;

  if (tileLogSize > 1) {
    --tileLogSize;
    uint32_t subTileSize = 1u << tileLogSize;
    // top-left
    rast_walk_tile(cfg, tileLogSize, x, y, pid, edges, delta, leaf);
    // top-right
    {
      vec3e_t s{ edges.x + ShiftLeft(delta.dx.x, tileLogSize),
                 edges.y + ShiftLeft(delta.dx.y, tileLogSize),
                 edges.z + ShiftLeft(delta.dx.z, tileLogSize) };
      rast_walk_tile(cfg, tileLogSize, x + subTileSize, y, pid, s, delta, leaf);
    }
    // bottom-left
    {
      vec3e_t s{ edges.x + ShiftLeft(delta.dy.x, tileLogSize),
                 edges.y + ShiftLeft(delta.dy.y, tileLogSize),
                 edges.z + ShiftLeft(delta.dy.z, tileLogSize) };
      rast_walk_tile(cfg, tileLogSize, x, y + subTileSize, pid, s, delta, leaf);
    }
    // bottom-right
    {
      vec3e_t s{ edges.x + ShiftLeft(delta.dx.x, tileLogSize) + ShiftLeft(delta.dy.x, tileLogSize),
                 edges.y + ShiftLeft(delta.dx.y, tileLogSize) + ShiftLeft(delta.dy.y, tileLogSize),
                 edges.z + ShiftLeft(delta.dx.z, tileLogSize) + ShiftLeft(delta.dy.z, tileLogSize) };
      rast_walk_tile(cfg, tileLogSize, x + subTileSize, y + subTileSize, pid, s, delta, leaf);
    }
  } else {
    leaf(cfg, x, y, pid, edges, delta);
  }
}

// Walk one primitive: build the per-edge deltas + start values at the tile
// origin (x, y), then recurse. `edges[i]` = {a, b, c} of edge i. `emit` is
// called once per covered quad as emit(pos_mask, const vec3e_t bcoords[4], pid).
template <typename Emit>
static inline void rast_walk_primitive(const RastConfig& cfg, uint32_t x, uint32_t y,
                                       uint32_t pid, const vec3e_t edges[3], Emit&& emit) {
  delta_t delta{
    { edges[0].x, edges[1].x, edges[2].x },
    { edges[0].y, edges[1].y, edges[2].y },
    { CalcEdgeExtents(edges[0]), CalcEdgeExtents(edges[1]), CalcEdgeExtents(edges[2]) }
  };
  vec3e_t value{
    EvalEdgeFunction(edges[0], x, y),
    EvalEdgeFunction(edges[1], x, y),
    EvalEdgeFunction(edges[2], x, y)
  };
  rast_walk_tile(cfg, cfg.tile_logsize, x, y, pid, value, delta,
    [&](const RastConfig& c, uint32_t qx, uint32_t qy, uint32_t qpid,
        const vec3e_t& e, const delta_t& d) {
      rast_emit_quad(c, qx, qy, qpid, e, d, emit);
    });
}

// MSAA walk: same tile recursion, but the leaf emits per-sample coverage (a quad
// survives if any sample is covered). `emit` is called as
// emit(pos_mask, const vec3e_t bcoords[4], const uint32_t sample_masks[4], pid).
template <typename Emit>
static inline void rast_walk_primitive_msaa(const RastConfig& cfg, uint32_t x, uint32_t y,
                                            uint32_t pid, const vec3e_t edges[3], Emit&& emit) {
  delta_t delta{
    { edges[0].x, edges[1].x, edges[2].x },
    { edges[0].y, edges[1].y, edges[2].y },
    { CalcEdgeExtents(edges[0]), CalcEdgeExtents(edges[1]), CalcEdgeExtents(edges[2]) }
  };
  vec3e_t value{
    EvalEdgeFunction(edges[0], x, y),
    EvalEdgeFunction(edges[1], x, y),
    EvalEdgeFunction(edges[2], x, y)
  };
  rast_walk_tile(cfg, cfg.tile_logsize, x, y, pid, value, delta,
    [&](const RastConfig& c, uint32_t qx, uint32_t qy, uint32_t qpid,
        const vec3e_t& e, const delta_t& d) {
      rast_emit_quad_msaa(c, qx, qy, qpid, e, d, emit);
    });
}

} // namespace gfx_rast

#endif // _RAST_SW_H_
