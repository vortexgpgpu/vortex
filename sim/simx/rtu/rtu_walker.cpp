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

#include "rtu_walker.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include <VX_types.h>      // VX_RT_FLAG_*, VX_RT_CB_TYPE_*

#include "rtu_types.h"       // RtuReq, SceneView, LaneState, PerfStats,
                             // scene-format constants
#include "rtu_bvh.h"         // CW-BVH node/leaf/instance layouts
#include "rtu_isect.h"       // ray_triangle, ray_aabb_intersect,
                             // world_to_object_ray
#include "rtu_classifier.h"  // classify_tri_hit, finalise_lane

namespace vortex { namespace rtu {

namespace {

// ────────────────────────────────────────────────────────────────────
// Walker-local helpers.
// ────────────────────────────────────────────────────────────────────

// Read `len` bytes of scene at offset `off` through the context's line set,
// crossing line boundaries as needed. A line the context does not hold is a
// miss: record the first one and zero-fill, so the caller can unwind. Every
// read site must test sv.miss before trusting what it read.
void read_scene_bytes(SceneView& sv, uint32_t off, uint32_t len, uint8_t* out) {
  uint32_t pos = sv.byte_off + off;
  uint32_t done = 0;
  while (done < len) {
    uint64_t addr = sv.base_line
                  + uint64_t(pos / VX_CFG_MEM_BLOCK_SIZE) * VX_CFG_MEM_BLOCK_SIZE;
    uint32_t bo = pos % VX_CFG_MEM_BLOCK_SIZE;
    uint32_t n  = std::min(len - done, VX_CFG_MEM_BLOCK_SIZE - bo);
    auto it = sv.lines->find(addr);
    if (it == sv.lines->end()) {
      if (!sv.miss) {
        sv.miss = true;
        sv.miss_line = addr;
      }
      std::memset(out + done, 0, n);
    } else {
      std::memcpy(out + done, it->second.data() + bo, n);
    }
    done += n;
    pos  += n;
  }
}

// CW-BVH: reconstruct a child AABB from quantized representation.
//   real = origin + qaabb * 2^exp (per axis)
inline void reconstruct_child_aabb(const float origin[3], const int8_t exp[3],
                                   const uint8_t qmin[3], const uint8_t qmax[3],
                                   float out_mn[3], float out_mx[3]) {
  for (int i = 0; i < 3; ++i) {
    float scale = std::ldexp(1.0f, exp[i]);
    out_mn[i] = origin[i] + static_cast<float>(qmin[i]) * scale;
    out_mx[i] = origin[i] + static_cast<float>(qmax[i]) * scale;
  }
}

// Copy a 3-vector (object-space ray capture helper).
inline void vcopy3(float dst[3], const float src[3]) {
  dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

// Per-lane BVH traversal accumulator. Shared across recursive sub-tree walks so
// a BLAS hit can update the same best_t that culls later TLAS-side AABB tests.
struct WalkCtx {
  float tmin, tmax;
  uint32_t ray_flags;
  uint32_t ray_cull_mask;  // Vulkan instanceCullMask gate
  bool     terminated;     // TERMINATE_ON_FIRST_HIT fired
  float best_t, best_u, best_v;
  uint32_t best_prim;
  uint32_t best_instance;
  uint32_t best_custom;    // VK_INSTANCE_CUSTOM_INDEX of the committed instance
  uint32_t best_geom;      // gl_GeometryIndexEXT of the committed leaf
  bool     best_kv;        // an opaque hit committed in this walk
  uint64_t best_order;     // (instance order << 32) | triangle order of it
  uint32_t best_blas_tab;  // its BLAS's visit-order table (0: none)
  float    best_tri_box[6];  // its triangle's vertex min/max
  uint32_t tlas_tab;       // the TLAS's visit-order table (0: none)
  float    world_o[3], world_d[3];
  bool any_hit;
  bool yield_pending;
  float yield_t, yield_u, yield_v;
  uint32_t yield_prim;
  uint32_t yield_sbt;
  uint32_t yield_cb_type;
  uint32_t yield_instance;
  uint32_t yield_custom;   // VK_INSTANCE_CUSTOM_INDEX of the yield candidate
  uint32_t yield_geom;     // gl_GeometryIndexEXT of the yield candidate
  // Object-space ray of the committed hit (best_obj_*) and the yield candidate
  // (yield_obj_*). Set to {ro,rd} at the leaf that wins; equals the world ray
  // at the top level (no instance).
  float best_obj_o[3],  best_obj_d[3];
  float yield_obj_o[3], yield_obj_d[3];
  // Candidates are offered one at a time in ascending (t, key) order, key =
  // (instance_id << 32) | record offset -- unique per primitive per instance. A
  // resumed walk skips everything at or below the last decided candidate's key.
  uint64_t yield_key;
  bool     has_floor;
  float    floor_t;
  uint64_t floor_key;
};

inline uint64_t cand_key(uint32_t instance_id, uint32_t record_off) {
  return (uint64_t(instance_id) << 32) | record_off;
}

// Whether a candidate at (t, key) replaces the pending one: nearer than the
// committed hit, above the resume floor, and first in (t, key) order.
inline bool cand_takes(const WalkCtx& ctx, float t, uint64_t key) {
  if (!(t < ctx.best_t)) return false;
  if (ctx.has_floor
   && (t < ctx.floor_t || (t == ctx.floor_t && key <= ctx.floor_key)))
    return false;
  if (!ctx.yield_pending) return true;
  return t < ctx.yield_t || (t == ctx.yield_t && key < ctx.yield_key);
}

// The geometry word's back-facing bit, after the instance's FLIP_FACING:
// what gl_HitKindEXT reports.
inline uint32_t hit_facing_bit(bool back_facing, uint32_t inst_flags) {
  if (inst_flags & kRtuInstanceFlagTriFlip) back_facing = !back_facing;
  return back_facing ? VX_RT_HIT_BACK_FACING : 0u;
}

inline uint64_t hit_order(uint32_t inst_order, uint32_t tri_order) {
  return (uint64_t(inst_order) << 32) | tri_order;
}

inline uint32_t scene_u32(SceneView& sv, uint32_t off) {
  uint32_t v = 0;
  read_scene_bytes(sv, off, sizeof(v), reinterpret_cast<uint8_t*>(&v));
  return v;
}

// A source-BVH child box tested exactly as the reference traversal tests it
// (F32, (bound - origin) * 1/dir, 1/0 -> FLT_MAX): `lo` is its entry distance
// (the child-order key), and the box is visited iff it passes against the
// committed t at the time.
struct SrcBox { float lo, hi; };

SrcBox src_box(const float box[6], const float o[3], const float inv[3]) {
  if (std::isnan(box[0])) return { INFINITY, -INFINITY };
  float b0[3], b1[3];
  for (int i = 0; i < 3; ++i) {
    b0[i] = (box[i] - o[i]) * inv[i];
    b1[i] = (box[3 + i] - o[i]) * inv[i];
  }
  const float lo = std::fmax(std::fmax(std::fmin(b0[0], b1[0]),
                                       std::fmin(b0[1], b1[1])),
                             std::fmin(b0[2], b1[2]));
  const float hi = std::fmin(std::fmin(std::fmax(b0[0], b1[0]),
                                       std::fmax(b0[1], b1[1])),
                             std::fmax(b0[2], b1[2]));
  return { lo, hi };
}

inline bool src_box_hit(const SrcBox& b) { return b.hi >= std::fmax(0.f, b.lo); }
inline float src_box_key(const SrcBox& b) { return src_box_hit(b) ? b.lo : INFINITY; }
inline bool src_box_passes(const SrcBox& b, float tmax) {
  return src_box_hit(b) && b.lo < tmax;
}

// The reference's ray in a table's space.
struct SrcRay { float o[3], d[3], inv[3]; };

SrcRay src_ray(const float o[3], const float d[3]) {
  SrcRay r;
  for (int i = 0; i < 3; ++i) {
    r.o[i] = o[i]; r.d[i] = d[i];
    r.inv[i] = (d[i] == 0.f) ? FLT_MAX : 1.0f / d[i];
  }
  return r;
}

// Visit-order tables: the source BVH's binary tree, which the reference walks
// depth-first, nearer child box first, child 0 on equal entry distances, each
// box tested against the hit committed when its parent is visited.
// TLAS table at `tab` (leaves = instances, indexed by visit rank):
//   { n_leaves, n_nodes, nodes_off, leaf_stride }
//   leaf[i] at tab + 16 + i*leaf_stride: { parent << 1 | side, _, _, _,
//     world->object 3x4 }
//   node[j] at tab + nodes_off + j*64: { child box 0 min/max, child box 1
//     min/max, parent << 1 | side (~0: root), depth }
// BLAS table at `tab` (leaves = triangles, each naming its parent << 1 | side
// in its leaf header; a triangle's box is its vertices' min/max):
//   { n_nodes, _, nodes_off, 32 }
//   node[j] at tab + nodes_off + j*32: { own box min/max, parent << 1 | side
//     (~0: root), depth }
constexpr uint32_t kSrcRoot          = 0xffffffffu;
constexpr uint32_t kSrcTabHdr        = 16;
constexpr uint32_t kSrcLeafWto       = 16;
constexpr uint32_t kSrcTlasNodeBytes = 64;
constexpr uint32_t kSrcTlasNodeParent = 48;
constexpr uint32_t kSrcTlasNodeDepth  = 52;
constexpr uint32_t kSrcBlasNodeBytes = 32;
constexpr uint32_t kSrcBlasNodeParent = 24;
constexpr uint32_t kSrcBlasNodeDepth  = 28;

// One leaf's climb towards the root. `box` is the box of the child the climb
// is at (tested when its parent was visited); each box the climb leaves below
// the lowest common ancestor's child was tested after the other leaf's
// subtree, had that come first, so the climb notes whether one of them fails
// against `tmax` (the other hit's t).
struct SrcClimb {
  uint32_t ps;      // parent << 1 | side of the current child
  uint32_t depth;   // depth of node(ps)
  float    box[6];
  bool     culled;
};

struct SrcTab {
  SceneView& sv;
  uint32_t tab, nodes_off, node_bytes, parent_off, depth_off;
  bool tlas;
  uint32_t node(uint32_t ps) const { return nodes_off + (ps >> 1) * node_bytes; }
  uint32_t parent(uint32_t ps) const { return scene_u32(sv, node(ps) + parent_off); }
  uint32_t depth(uint32_t ps) const { return scene_u32(sv, node(ps) + depth_off); }
  // Box of the child at ps: a TLAS node holds its children's boxes, a BLAS
  // node its own.
  void child_box(uint32_t ps, uint32_t below, float box[6]) const {
    if (tlas)
      read_scene_bytes(sv, node(ps) + 24 * (ps & 1u), 24, reinterpret_cast<uint8_t*>(box));
    else
      read_scene_bytes(sv, node(below), 24, reinterpret_cast<uint8_t*>(box));
  }
};

SrcTab src_tab(SceneView& sv, uint32_t tab, bool tlas) {
  return { sv, tab, tab + scene_u32(sv, tab + 8),
           tlas ? kSrcTlasNodeBytes : kSrcBlasNodeBytes,
           tlas ? kSrcTlasNodeParent : kSrcBlasNodeParent,
           tlas ? kSrcTlasNodeDepth : kSrcBlasNodeDepth, tlas };
}

void src_up(const SrcTab& t, SrcClimb& c, const SrcRay& r, float tmax) {
  if (!src_box_passes(src_box(c.box, r.o, r.inv), tmax)) c.culled = true;
  const uint32_t below = c.ps;
  c.ps = t.parent(c.ps);
  --c.depth;
  t.child_box(c.ps, below, c.box);
}

// Climb both leaves to their lowest common ancestor; true when a's side is
// visited first there. a.culled / b.culled report the boxes left on the way.
bool src_lca(const SrcTab& t, SrcClimb& a, SrcClimb& b, const SrcRay& r,
             float ta, float tb) {
  while (!t.sv.miss && a.depth > b.depth) src_up(t, a, r, tb);
  while (!t.sv.miss && b.depth > a.depth) src_up(t, b, r, ta);
  while (!t.sv.miss && (a.ps >> 1) != (b.ps >> 1)) {
    src_up(t, a, r, tb);
    src_up(t, b, r, ta);
  }
  const float ka = src_box_key(src_box(a.box, r.o, r.inv));
  const float kb = src_box_key(src_box(b.box, r.o, r.inv));
  const float d0 = (a.ps & 1u) ? kb : ka;
  const float d1 = (a.ps & 1u) ? ka : kb;
  const uint32_t first = (d1 < d0) ? 1u : 0u;
  return (a.ps & 1u) == first;
}

SrcClimb src_blas_leaf(const SrcTab& t, uint32_t ps, const float box[6]) {
  SrcClimb c;
  c.ps = ps;
  c.depth = (ps == kSrcRoot) ? 0 : t.depth(ps) + 1;
  std::memcpy(c.box, box, sizeof(c.box));
  c.culled = false;
  return c;
}

// Whether a box on the leaf's whole BLAS path fails against tmax: every box
// below the BLAS root is tested after the instance is entered.
bool src_blas_path_culled(const SrcTab& t, uint32_t ps, const float box[6],
                          const SrcRay& r, float tmax) {
  SrcClimb c = src_blas_leaf(t, ps, box);
  while (!t.sv.miss && c.ps != kSrcRoot) src_up(t, c, r, tmax);
  return c.culled;
}

// The reference's object-space ray for TLAS leaf `inst`, from its
// world->object matrix in the table.
SrcRay src_object_ray(SceneView& sv, uint32_t tlas_tab, uint32_t inst,
                      const float wo[3], const float wd[3]) {
  const uint32_t leaf_stride = scene_u32(sv, tlas_tab + 12);
  float m[12];
  read_scene_bytes(sv, tlas_tab + kSrcTabHdr + inst * leaf_stride + kSrcLeafWto,
                   sizeof(m), reinterpret_cast<uint8_t*>(m));
  float o[3], d[3];
  world_to_object_ray(m, wo, wd, o, d);
  return src_ray(o, d);
}

inline void tri_box(const float* v, float box[6]) {
  for (int a = 0; a < 3; ++a) {
    box[a]     = std::fmin(v[a], std::fmin(v[3 + a], v[6 + a]));
    box[3 + a] = std::fmax(v[a], std::fmax(v[3 + a], v[6 + a]));
  }
}

// An opaque triangle hit as the source traversal sees it.
struct SrcHit {
  float    t;
  uint32_t inst;      // TLAS visit rank
  uint32_t ps;        // parent << 1 | side in its BLAS table
  uint32_t blas_tab;
  float    box[6];
};

// Which of two opaque hits within a few ulps of each other the source
// traversal keeps: it commits a hit only nearer than the committed one, and
// tests each box against the committed t, so the first-reached of two equal-t
// hits wins, and a nearer hit reached second is lost when a box on its way
// (below where the two paths part) enters at or past the first one's t.
bool src_keeps_a(SceneView& sv, uint32_t tlas_tab, const float wo[3],
                 const float wd[3], const SrcHit& a, const SrcHit& b) {
  bool a_first;
  bool a_culled, b_culled;
  if (a.inst == b.inst) {
    const SrcTab t = src_tab(sv, a.blas_tab, false);
    const SrcRay r = src_object_ray(sv, tlas_tab, a.inst, wo, wd);
    if (a.ps == kSrcRoot || b.ps == kSrcRoot) return a.t <= b.t;
    SrcClimb ca = src_blas_leaf(t, a.ps, a.box);
    SrcClimb cb = src_blas_leaf(t, b.ps, b.box);
    a_first = src_lca(t, ca, cb, r, a.t, b.t);
    a_culled = ca.culled; b_culled = cb.culled;
  } else {
    const SrcTab tt = src_tab(sv, tlas_tab, true);
    const SrcRay rw = src_ray(wo, wd);
    const uint32_t stride = scene_u32(sv, tlas_tab + 12);
    auto tlas_leaf = [&](uint32_t inst) {
      SrcClimb c;
      c.ps = scene_u32(sv, tlas_tab + kSrcTabHdr + inst * stride);
      c.depth = (c.ps == kSrcRoot) ? 0 : tt.depth(c.ps) + 1;
      tt.child_box(c.ps, 0, c.box);
      c.culled = false;
      return c;
    };
    SrcClimb ca = tlas_leaf(a.inst), cb = tlas_leaf(b.inst);
    if (sv.miss) return false;
    a_first = src_lca(tt, ca, cb, rw, a.t, b.t);
    a_culled = ca.culled
            || src_blas_path_culled(src_tab(sv, a.blas_tab, false), a.ps, a.box,
                                    src_object_ray(sv, tlas_tab, a.inst, wo, wd), b.t);
    b_culled = cb.culled
            || src_blas_path_culled(src_tab(sv, b.blas_tab, false), b.ps, b.box,
                                    src_object_ray(sv, tlas_tab, b.inst, wo, wd), a.t);
  }
  if (a.t == b.t) return a_first;
  if (a.t < b.t) return !(!a_first && a_culled);
  return a_first && b_culled;
}

// A small relative bound past the committed t: within it, a box's slab entry
// (a few ulps of rounding a triangle's watertight t does not carry) can land
// on either side of the committed t, so which of two such hits the source
// traversal keeps depends on its visit order. Beyond it the nearer hit wins.
inline float near_t(float t) { return t + std::fabs(t) * 0x1p-19f; }

// The bound a child box's entry is culled against: kept open by near_t, so
// every hit the source traversal might keep over the committed one is reached.
inline float box_cull_t(const WalkCtx& ctx) {
  return ctx.best_kv ? near_t(ctx.best_t) : ctx.best_t;
}

// Whether an opaque hit at t replaces the committed one. Within near_t of an
// opaque hit committed in this walk, the scene's visit-order tables settle it
// as the source traversal would; without tables, an exact tie goes to the
// lowest (instance, geometry, primitive) and otherwise the nearer hit wins.
bool commit_takes(SceneView& sv, const WalkCtx& ctx, float t, uint64_t order,
                  uint32_t blas_tab, const float* tri, uint32_t instance_id,
                  uint32_t geom, uint32_t prim) {
  const bool near = ctx.best_kv
                 && (t == ctx.best_t
                  || (t < ctx.best_t && near_t(t) >= ctx.best_t)
                  || (t > ctx.best_t && t <= near_t(ctx.best_t)));
  if (near && ctx.tlas_tab && tri && blas_tab && ctx.best_blas_tab
   && order != ctx.best_order) {
    SrcHit a, b;
    a.t = t; a.inst = uint32_t(order >> 32); a.ps = uint32_t(order);
    a.blas_tab = blas_tab; tri_box(tri, a.box);
    b.t = ctx.best_t; b.inst = uint32_t(ctx.best_order >> 32);
    b.ps = uint32_t(ctx.best_order); b.blas_tab = ctx.best_blas_tab;
    std::memcpy(b.box, ctx.best_tri_box, sizeof(b.box));
    const bool keeps_new = src_keeps_a(sv, ctx.tlas_tab, ctx.world_o,
                                       ctx.world_d, a, b);
    return !sv.miss && keeps_new;
  }
  if (t < ctx.best_t) return true;
  if (!(ctx.best_kv && t == ctx.best_t)) return false;
  const uint32_t best_geom = ctx.best_geom & VX_RT_HIT_GEOMETRY_MASK;
  geom &= VX_RT_HIT_GEOMETRY_MASK;
  if (instance_id != ctx.best_instance) return instance_id < ctx.best_instance;
  if (geom != best_geom) return geom < best_geom;
  return prim < ctx.best_prim;
}

// Depth-first walker for one BVH sub-tree under the supplied (object-space)
// ray. Recurses on LeafInst so each instance's BLAS gets walked with its
// transformed ray. ctx accumulates hits/yields across the whole call tree.
//
// Returns as soon as sv.miss is raised: the caller discards the partial state
// and replays the walk once the missing line has arrived.
//
// The functional traversal stack is unbounded so the oracle never misses a hit;
// the HW short-stack (VX_CFG_RTU_STACK_DEPTH) overflow is counted as a
// trail-based RESTART and charged in the cost model.
void walk_bvh4_subtree(SceneView& sv,
                       const float ro[3], const float rd[3],
                       uint32_t root_off, uint32_t instance_id,
                       uint32_t custom_id, uint32_t inst_flags,
                       uint32_t inst_order, uint32_t blas_tab,
                       WalkCtx& ctx, PerfStats& perf) {
  auto visit_leaf_tri = [&](uint32_t leaf_off, uint32_t count) {
    uint8_t hdr_buf[kVxBvhLeafHeaderBytes];
    read_scene_bytes(sv, leaf_off, sizeof(hdr_buf), hdr_buf);
    if (sv.miss) return;
    const VxBvhLeafHeader* hdr =
        reinterpret_cast<const VxBvhLeafHeader*>(hdr_buf);
    uint32_t leaf_geom = hdr->geometry_index;
    uint32_t leaf_prim_base = hdr->prim_base;  // Vulkan gl_PrimitiveID base
    uint32_t leaf_order = hdr->flags;          // LeafTri: source parent/side
    uint32_t tris_off = leaf_off + kVxBvhLeafHeaderBytes;
    for (uint32_t i = 0; i < count; ++i) {
      if (ctx.terminated) return;
      uint8_t tri_buf[kVxBvhTriStride];
      read_scene_bytes(sv, tris_off + i * kVxBvhTriStride,
                       kVxBvhTriStride, tri_buf);
      if (sv.miss) return;
      const float* tri = reinterpret_cast<const float*>(tri_buf);
      uint32_t tri_flags = 0;
      std::memcpy(&tri_flags, tri_buf + kPhase2TriFlagsOff,
                  sizeof(uint32_t));

      float t_hit = 0.f, u = 0.f, v = 0.f;
      bool back_facing = false;
      ++perf.bvh_tri_tests;
      const bool tri_hit = ray_triangle(ro, rd, &tri[0], &tri[3], &tri[6],
                        ctx.tmin, ctx.tmax,
                        t_hit, u, v, back_facing);
      if (!tri_hit) continue;

      TriClassify cls = classify_tri_hit(ctx.ray_flags, tri_flags,
                                          inst_flags, back_facing);
      const uint32_t hit_geom = (leaf_geom & VX_RT_HIT_GEOMETRY_MASK)
                              | hit_facing_bit(back_facing, inst_flags);
      if (cls.action == TriAction::Ignore) continue;

      if (cls.action == TriAction::Commit) {
        const uint64_t order = hit_order(inst_order, leaf_order + i);
        const bool takes = commit_takes(sv, ctx, t_hit, order, blas_tab, tri,
                                        instance_id, leaf_geom,
                                        leaf_prim_base + i);
        if (sv.miss) return;
        if (takes) {
          ctx.best_t = t_hit; ctx.best_u = u; ctx.best_v = v;
          ctx.best_order = order;
          ctx.best_blas_tab = blas_tab;
          tri_box(tri, ctx.best_tri_box);
          ctx.best_prim = leaf_prim_base + i;
          ctx.best_instance = instance_id;
          ctx.best_custom = custom_id;
          ctx.best_geom = hit_geom;
          ctx.any_hit = true;
          ctx.best_kv = true;
          vcopy3(ctx.best_obj_o, ro);   // object-space ray of this BLAS
          vcopy3(ctx.best_obj_d, rd);
          if (ctx.yield_pending && ctx.yield_t >= ctx.best_t) {
            ctx.yield_pending = false;
            ctx.yield_t = ctx.tmax;
          }
          if (cls.terminate_on_first_hit) {
            ctx.terminated = true;
            return;
          }
        }
      } else {  // TriAction::Yield
        uint64_t key = cand_key(instance_id, tris_off + i * kVxBvhTriStride);
        if (cand_takes(ctx, t_hit, key)) {
          ctx.yield_pending = true;
          ctx.yield_key = key;
          ctx.yield_t = t_hit; ctx.yield_u = u; ctx.yield_v = v;
          ctx.yield_prim = leaf_prim_base + i;
          ctx.yield_instance = instance_id;
          ctx.yield_custom = custom_id;
          ctx.yield_geom = hit_geom;
          ctx.yield_sbt = cls.yield_sbt_idx;
          ctx.yield_cb_type = cls.yield_cb_type;
          vcopy3(ctx.yield_obj_o, ro);  // object-space ray for AHS/IS
          vcopy3(ctx.yield_obj_d, rd);
        }
      }
    }
  };

  // Procedural-AABB leaf. Each record is a custom primitive's bounding box; a
  // ray-AABB hit yields an IS callback so the kernel's intersection shader
  // computes the real hit. The candidate t is the AABB entry parameter (a lower
  // bound); the IS supplies the true t via the CONTINUE operands.
  auto visit_leaf_proc = [&](uint32_t leaf_off, uint32_t count) {
    uint8_t hdr_buf[kVxBvhLeafHeaderBytes];
    read_scene_bytes(sv, leaf_off, sizeof(hdr_buf), hdr_buf);
    if (sv.miss) return;
    const VxBvhLeafHeader* hdr =
        reinterpret_cast<const VxBvhLeafHeader*>(hdr_buf);
    uint32_t leaf_sbt =
        (hdr->flags >> kVxBvhLeafSbtIdxShift) & kVxBvhLeafSbtIdxMask;
    uint32_t aabbs_off = leaf_off + kVxBvhLeafHeaderBytes;
    for (uint32_t i = 0; i < count; ++i) {
      if (ctx.terminated) return;
      uint8_t rec_buf[sizeof(VxBvhProcAabb)];
      read_scene_bytes(sv, aabbs_off + i * uint32_t(sizeof(VxBvhProcAabb)),
                       sizeof(rec_buf), rec_buf);
      if (sv.miss) return;
      const VxBvhProcAabb* rec =
          reinterpret_cast<const VxBvhProcAabb*>(rec_buf);
      float t_near = 0.f;
      ++perf.bvh_box_tests;
      if (!ray_aabb_intersect(ro, rd, rec->aabb_min, rec->aabb_max,
                              ctx.tmin, ctx.best_t, t_near)) {
        continue;
      }
      // Procedural primitives are inherently non-opaque (the IS decides the
      // hit), so always stage an IS yield for the closest candidate.
      uint64_t key = cand_key(instance_id,
                              aabbs_off + i * uint32_t(sizeof(VxBvhProcAabb)));
      if (cand_takes(ctx, t_near, key)) {
        ctx.yield_pending = true;
        ctx.yield_key = key;
        ctx.yield_t = t_near; ctx.yield_u = 0.f; ctx.yield_v = 0.f;
        ctx.yield_prim = hdr->prim_base + i;   // gl_PrimitiveID, as for LEAF_TRI
        ctx.yield_instance = instance_id;
        ctx.yield_custom = custom_id;
        ctx.yield_geom = hdr->geometry_index & VX_RT_HIT_GEOMETRY_MASK;
        ctx.yield_sbt = leaf_sbt;
        ctx.yield_cb_type = VX_RT_CB_TYPE_PROC;
        vcopy3(ctx.yield_obj_o, ro);
        vcopy3(ctx.yield_obj_d, rd);
      }
    }
  };

  auto visit_leaf_inst = [&](uint32_t leaf_off, uint32_t count) {
    uint8_t hdr_buf[kVxBvhLeafHeaderBytes];
    read_scene_bytes(sv, leaf_off, sizeof(hdr_buf), hdr_buf);
    if (sv.miss) return;
    // LeafInst header: geometry_index = the BLAS's visit-order table,
    // flags = the instance's visit order, prim_base = the TLAS's table.
    const VxBvhLeafHeader* ihdr =
        reinterpret_cast<const VxBvhLeafHeader*>(hdr_buf);
    const uint32_t leaf_order = ihdr->flags;
    ctx.tlas_tab = ihdr->prim_base;
    uint32_t insts_off = leaf_off + kVxBvhLeafHeaderBytes;
    for (uint32_t i = 0; i < count; ++i) {
      uint8_t inst_buf[kVxBvhInstanceStride];
      read_scene_bytes(sv, insts_off + i * kVxBvhInstanceStride,
                       kVxBvhInstanceStride, inst_buf);
      if (sv.miss) return;
      const VxBvhInstance* inst =
          reinterpret_cast<const VxBvhInstance*>(inst_buf);
      // Vulkan instanceCullMask: skip the instance entirely if its mask byte
      // and the ray's cull_mask have no bits in common. Both default to 0xff in
      // the no-culling path.
      if ((inst->cull_mask & ctx.ray_cull_mask & 0xffu) == 0) continue;
      // VkGeometryInstanceFlagBits packed into cull_mask bits 15..8.
      uint32_t inst_flags2 =
          (inst->cull_mask >> kRtuInstanceFlagsShift) & kRtuInstanceFlagsMask;
      float obj_ro[3], obj_rd[3];
      world_to_object_ray(inst->xform, ro, rd, obj_ro, obj_rd);
      ++perf.bvh_instance_descents;
      walk_bvh4_subtree(sv, obj_ro, obj_rd,
                        inst->blas_root_byte_offset,
                        inst->instance_id,
                        inst->custom_id, inst_flags2,
                        leaf_order + i, ihdr->geometry_index,
                        ctx, perf);
      if (sv.miss) return;
    }
  };

  // SimX is the correctness oracle: keep an UNBOUNDED traversal stack so deep
  // sub-trees are never dropped. The HW short-stack is only
  // VX_CFG_RTU_STACK_DEPTH deep and trail-restarts to re-find subtrees it had
  // to evict — it visits the same leaves, just at extra cost.
  std::vector<uint32_t> stack;
  stack.reserve(VX_CFG_RTU_STACK_DEPTH);
  uint32_t current = root_off;
  bool have_current = true;

  // Safety backstop: a malformed/cyclic acceleration structure (a child offset
  // pointing back at an ancestor) would otherwise descend forever with the
  // unbounded stack. The ceiling is far above any well-formed scene's node
  // count, so it never truncates a legitimate walk.
  constexpr uint64_t kMaxNodeVisits = 1ull << 22;
  uint64_t node_visits = 0;

  while (have_current) {
    if (ctx.terminated) break;
    if (++node_visits > kMaxNodeVisits) break;
    uint8_t kind_buf[4];
    read_scene_bytes(sv, current, sizeof(kind_buf), kind_buf);
    if (sv.miss) return;
    uint32_t kind_word = 0;
    std::memcpy(&kind_word, kind_buf, sizeof(uint32_t));
    uint32_t kind  = kind_word & kVxBvhKindMask;
    uint32_t count = (kind_word >> kVxBvhCountShift) & kVxBvhCountMask;

    if (kind == kVxBvhKindLeafTri) {
      ++perf.bvh_leaves_fetched;
      if (!(ctx.ray_flags & VX_RT_FLAG_SKIP_TRIANGLES)) {
        visit_leaf_tri(current, count);
        if (sv.miss) return;
      }
    } else if (kind == kVxBvhKindLeafInst) {
      ++perf.bvh_leaves_fetched;
      visit_leaf_inst(current, count);
      if (sv.miss) return;
    } else if (kind == kVxBvhKindLeafProc) {
      ++perf.bvh_leaves_fetched;
      // SKIP_AABBS: symmetric gate with SKIP_TRIANGLES. Otherwise ray-test each
      // procedural AABB and yield IS for the closest hit.
      if (!(ctx.ray_flags & VX_RT_FLAG_SKIP_AABBS)) {
        visit_leaf_proc(current, count);
        if (sv.miss) return;
      }
    } else if (kind == kVxBvhKindInternal) {
      ++perf.bvh_nodes_fetched;
      // Width-generic decode: CW-BVH4 (64 B) or CW-BVH6 (96 B). Both decode
      // into VxBvhNodeView so the box-test loop and the box-PE cycle model are
      // fan-out independent.
      VxBvhNodeView nv;
#if VX_CFG_RTU_BVH_WIDTH == 6
      uint8_t node_buf[sizeof(VxBvh6InternalNode)];
      read_scene_bytes(sv, current, sizeof(node_buf), node_buf);
      if (sv.miss) return;
      decode_bvh6_node(
          reinterpret_cast<const VxBvh6InternalNode*>(node_buf), count, nv);
#else
      uint8_t node_buf[sizeof(VxBvhInternalNode)];
      read_scene_bytes(sv, current, sizeof(node_buf), node_buf);
      if (sv.miss) return;
      decode_bvh4_node(
          reinterpret_cast<const VxBvhInternalNode*>(node_buf), count, nv);
#endif

      struct ChildHit { uint32_t offset; float t_near; };
      ChildHit hits[kVxBvhMaxWidth];
      uint32_t hit_count = 0;
      for (uint32_t i = 0; i < nv.n_children; ++i) {
        uint32_t off_word  = nv.child_offsets[i];
        uint32_t child_off = off_word & kVxBvhChildOffsetMask;
        if (off_word == kVxBvhChildEmpty) continue;
        float mn[3], mx[3];
        reconstruct_child_aabb(nv.origin, nv.exp,
                                nv.qaabb_min[i], nv.qaabb_max[i],
                                mn, mx);
        float t_near = 0.f;
        ++perf.bvh_box_tests;
        if (!ray_aabb_intersect(ro, rd, mn, mx,
                                ctx.tmin, box_cull_t(ctx), t_near)) {
          continue;
        }
        hits[hit_count++] = { child_off, t_near };
      }
      // Insertion-sort children by t_near (nearest-first traversal).
      for (uint32_t i = 1; i < hit_count; ++i) {
        ChildHit h = hits[i];
        uint32_t j = i;
        while (j > 0 && hits[j-1].t_near > h.t_near) {
          hits[j] = hits[j-1]; --j;
        }
        hits[j] = h;
      }
      if (hit_count > 0) {
        for (uint32_t i = hit_count; i-- > 1; ) {
          // Each push past the HW short-stack depth is one subtree the HW would
          // have to re-descend for via restart — count it for the cost model,
          // but keep it (unbounded) so the functional walk never misses a hit.
          if (stack.size() >= VX_CFG_RTU_STACK_DEPTH)
            ++perf.bvh_stack_restarts;
          stack.push_back(hits[i].offset);
        }
        current = hits[0].offset;
        have_current = true;
        continue;
      }
    }

    if (stack.empty()) {
      have_current = false;
    } else {
      current = stack.back();
      stack.pop_back();
    }
  }
}

// End-of-lane finalise: translates the accumulated walk state into LaneState
// writes. Returns true iff a CB_YIELD should be queued for this lane (the queue
// push itself is the orchestrator's, once the slot's PE work has drained).
bool emit_lane_result(const RtuReq& req, LaneState& l, uint32_t t,
                      const WalkCtx& ctx) {
  l.hit       = ctx.any_hit;
  l.hit_t     = ctx.best_t;
  l.hit_u     = ctx.best_u;
  l.hit_v     = ctx.best_v;
  l.hit_prim  = ctx.best_prim;
  l.hit_instance_id = ctx.any_hit ? ctx.best_instance
                                  : (ctx.yield_pending ? ctx.yield_instance : 0u);
  l.hit_instance_custom = ctx.any_hit ? ctx.best_custom
                                      : (ctx.yield_pending ? ctx.yield_custom : 0u);
  l.hit_geometry  = ctx.best_geom;
  l.cand_geometry = ctx.yield_geom;
  vcopy3(l.hit_obj_o,  ctx.best_obj_o);
  vcopy3(l.hit_obj_d,  ctx.best_obj_d);
  vcopy3(l.cand_obj_o, ctx.yield_obj_o);
  vcopy3(l.cand_obj_d, ctx.yield_obj_d);

  LaneAction action = finalise_lane(req.flags[t], ctx.any_hit,
                                     ctx.yield_pending, ctx.yield_cb_type);
  switch (action) {
  case LaneAction::TerminalHit:
  case LaneAction::TerminalMiss:
    return false;
  case LaneAction::YieldAhs:
  case LaneAction::YieldIs:
    l.cb_pending = true;
    l.cb_type    = ctx.yield_cb_type;
    l.sbt_idx    = ctx.yield_sbt;
    l.cand_t     = ctx.yield_t;
    l.cand_u     = ctx.yield_u;
    l.cand_v     = ctx.yield_v;
    l.cand_prim  = ctx.yield_prim;
    l.cand_instance = ctx.yield_instance;
    l.cand_custom   = ctx.yield_custom;
    l.cand_key      = ctx.yield_key;
    return true;
  case LaneAction::YieldChs:
    l.cb_pending = true;
    l.cb_type    = VX_RT_CB_TYPE_CHS;
    l.sbt_idx    = 0;
    l.cand_t     = ctx.best_t;
    l.cand_u     = ctx.best_u;
    l.cand_v     = ctx.best_v;
    l.cand_prim  = ctx.best_prim;
    l.cand_instance = ctx.best_instance;
    l.cand_custom   = ctx.best_custom;
    // A CHS candidate IS the committed hit, so gl_GeometryIndexEXT reads the
    // committed leaf's geometry (not the stale yield_geom).
    l.cand_geometry = ctx.best_geom;
    return true;
  case LaneAction::YieldMiss:
    l.cb_pending = true;
    l.cb_type    = VX_RT_CB_TYPE_MISS;
    l.sbt_idx    = 0;
    l.cand_t     = 0.f;
    l.cand_u     = 0.f;
    l.cand_v     = 0.f;
    l.cand_prim  = 0;
    l.cand_instance = 0;
    l.cand_custom   = 0;
    return true;
  }
  return false;  // unreachable
}

// Common init of the traversal accumulator from the ray, and -- for a walk
// resumed after a callback verdict -- from the lane's committed hit and floor.
WalkCtx init_ctx(const RtuReq& req, uint32_t t,
                 const float ro[3], const float rd[3], const LaneState& l) {
  WalkCtx ctx;
  ctx.tmin = req.tmin[t];
  ctx.tmax = req.tmax[t];
  ctx.ray_flags = req.flags[t];
  ctx.ray_cull_mask = req.cull_mask[t];
  ctx.terminated = false;
  ctx.best_t = ctx.tmax;
  ctx.best_u = 0.f; ctx.best_v = 0.f;
  ctx.best_prim = 0; ctx.best_instance = 0; ctx.best_custom = 0;
  ctx.best_geom = 0;
  ctx.any_hit = false;
  ctx.best_kv = false;
  ctx.best_order = 0;
  ctx.best_blas_tab = 0;
  ctx.tlas_tab = 0;
  vcopy3(ctx.world_o, ro); vcopy3(ctx.world_d, rd);
  ctx.yield_pending = false;
  ctx.yield_t = ctx.tmax; ctx.yield_u = 0.f; ctx.yield_v = 0.f;
  ctx.yield_prim = 0; ctx.yield_sbt = 0;
  ctx.yield_cb_type = VX_RT_CB_TYPE_ANYHIT;
  ctx.yield_instance = 0; ctx.yield_custom = 0; ctx.yield_geom = 0;
  // Default object ray = world ray (overwritten at a BLAS leaf if the hit is
  // under an instance).
  vcopy3(ctx.best_obj_o, ro);  vcopy3(ctx.best_obj_d, rd);
  vcopy3(ctx.yield_obj_o, ro); vcopy3(ctx.yield_obj_d, rd);
  ctx.yield_key = 0;
  ctx.has_floor = l.has_floor;
  ctx.floor_t   = l.floor_t;
  ctx.floor_key = l.floor_key;
  if (l.has_floor && l.hit) {
    ctx.any_hit = true;
    ctx.best_t = l.hit_t; ctx.best_u = l.hit_u; ctx.best_v = l.hit_v;
    ctx.best_prim = l.hit_prim;
    ctx.best_instance = l.hit_instance_id;
    ctx.best_custom = l.hit_instance_custom;
    ctx.best_geom = l.hit_geometry;
    vcopy3(ctx.best_obj_o, l.hit_obj_o);
    vcopy3(ctx.best_obj_d, l.hit_obj_d);
  }
  return ctx;
}

}  // namespace

// ════════════════════════════════════════════════════════════════════
// FlatWalker
// ════════════════════════════════════════════════════════════════════

WalkResult FlatWalker::walk_lane(const RtuReq& req, uint32_t t, SceneView& sv,
                                  LaneState& l, PerfStats& perf) {
  // The scene header is the first thing the ray reads, so it is also the first
  // line it fetches.
  uint8_t hdr[kRtuSceneHeaderBytes];
  read_scene_bytes(sv, 0, sizeof(hdr), hdr);
  if (sv.miss) return {true, false};
  uint32_t primary_count = 0;
  std::memcpy(&primary_count, hdr, sizeof(uint32_t));

  const float ro[3] = { req.origin_x[t], req.origin_y[t], req.origin_z[t] };
  const float rd[3] = { req.dir_x[t],    req.dir_y[t],    req.dir_z[t]   };
  WalkCtx ctx = init_ctx(req, t, ro, rd, l);

  // TLAS scenes walk one or more instances; each instance points at a BLAS (a
  // triangle list) through its world→object affine transform.
  uint32_t num_instances  = 1;
  uint32_t triangle_count = 0;
#ifdef VX_CFG_RTU_TLAS_ENABLE
  num_instances = std::min(primary_count, kRtuMaxInstancesPerTlas);
  if (num_instances == 0) {
    l.hit = false;
    return {false, false};
  }
#else
  triangle_count = std::min(primary_count, kRtuMaxTrisPerScene);
  if (triangle_count == 0) {
    l.hit = false;
    return {false, false};
  }
#endif

  uint8_t tri_buf[kPhase2TriStride];

  // Scan ALL instances (not stopping at the first with a yield candidate): a
  // later instance may hold a NEARER opaque hit (which occludes an alpha-tested
  // candidate) or a nearer non-opaque candidate. The inner logic already keeps
  // the single closest opaque + closest non-opaque via the best_t / yield_t
  // monotone compares, so scanning every instance is order-independent and
  // matches the Bvh4Walker whole-tree walk. Exception: a terminate-on-first-hit
  // commit halts the entire walk, so no later instance can substitute a
  // different reported hit.
  for (uint32_t inst_idx = 0; inst_idx < num_instances && !ctx.terminated;
       ++inst_idx) {
    uint32_t blas_tri_off   = kRtuSceneHeaderBytes;
    uint32_t blas_tri_count = triangle_count;
    uint32_t cur_custom     = 0;
    uint32_t cur_inst_flags = 0;
    float ray_o[3] = { ro[0], ro[1], ro[2] };
    float ray_d[3] = { rd[0], rd[1], rd[2] };
#ifdef VX_CFG_RTU_TLAS_ENABLE
    {
      uint32_t inst_off = kRtuSceneHeaderBytes
                        + inst_idx * kRtuInstanceStride;
      uint8_t inst_buf[kRtuInstanceStride];
      read_scene_bytes(sv, inst_off, sizeof(inst_buf), inst_buf);
      if (sv.miss) return {true, false};
      // Vulkan instanceCullMask gate — skip the entire instance (transform +
      // BLAS scan) before doing the affine ray transform when masks don't
      // overlap. Same semantics as the BVH LeafInst gate.
      uint32_t inst_cull_mask = 0;
      std::memcpy(&inst_cull_mask,
                  inst_buf + kRtuInstanceCullMaskOff,
                  sizeof(uint32_t));
      if ((inst_cull_mask & req.cull_mask[t] & 0xffu) == 0) continue;
      // VkGeometryInstanceFlagBits packed into cull_mask bits 15..8.
      cur_inst_flags =
          (inst_cull_mask >> kRtuInstanceFlagsShift) & kRtuInstanceFlagsMask;
      const float* xform = reinterpret_cast<const float*>(inst_buf);
      uint32_t blas_byte_off = 0;
      std::memcpy(&blas_byte_off,
                  inst_buf + kRtuInstanceBlasOffOff,
                  sizeof(uint32_t));
      std::memcpy(&cur_custom,
                  inst_buf + kRtuInstanceCustomIdOff,
                  sizeof(uint32_t));
      // World→object ray transform; the direction is not renormalised, so
      // the BLAS-reported hit_t is also the world hit_t.
      world_to_object_ray(xform, ro, rd, ray_o, ray_d);
      ++perf.bvh_instance_descents;
      uint8_t blas_hdr[4];
      read_scene_bytes(sv, blas_byte_off, sizeof(blas_hdr), blas_hdr);
      if (sv.miss) return {true, false};
      uint32_t bcount = 0;
      std::memcpy(&bcount, blas_hdr, sizeof(uint32_t));
      blas_tri_count = std::min(bcount, kRtuMaxTrisPerScene);
      blas_tri_off   = blas_byte_off + kRtuSceneHeaderBytes;
    }
#endif

    // Walk the *full* triangle list. Track the best opaque hit and the closest
    // non-opaque candidate separately. If a non-opaque candidate ends up closer
    // than the best opaque, yield it; otherwise the opaque commits and no AHS
    // fires (alpha-test fast path).
    //
    // SKIP_TRIANGLES bails the whole leaf-tri scan. (Flat-list scenes only have
    // tri leaves, so SKIP_AABBS is a no-op here.)
    if (ctx.ray_flags & VX_RT_FLAG_SKIP_TRIANGLES) continue;

    for (uint32_t i = 0; i < blas_tri_count; ++i) {
      uint32_t tri_off = blas_tri_off + i * kPhase2TriStride;
      read_scene_bytes(sv, tri_off, kPhase2TriStride, tri_buf);
      if (sv.miss) return {true, false};
      const float* tri = reinterpret_cast<const float*>(tri_buf);
      uint32_t tri_flags = 0;
      std::memcpy(&tri_flags, tri_buf + kPhase2TriFlagsOff,
                  sizeof(uint32_t));

      float t_hit = 0.f, u = 0.f, v = 0.f;
      bool back_facing = false;
      ++perf.bvh_tri_tests;
      // Test against ray.tmax (not best_t) so an opaque hit committed earlier in
      // this walk doesn't pre-cull a non-opaque candidate that might survive an
      // ACCEPT.
      if (!ray_triangle(ray_o, ray_d, &tri[0], &tri[3], &tri[6],
                        ctx.tmin, ctx.tmax,
                        t_hit, u, v, back_facing)) {
        continue;
      }

      TriClassify cls = classify_tri_hit(ctx.ray_flags, tri_flags,
                                          cur_inst_flags, back_facing);
      const uint32_t hit_geom = hit_facing_bit(back_facing, cur_inst_flags);
      if (cls.action == TriAction::Ignore) continue;

      if (cls.action == TriAction::Commit) {
        const bool takes = commit_takes(sv, ctx, t_hit, hit_order(inst_idx, i),
                                        0, nullptr, inst_idx, 0, i);
        if (takes) {
          ctx.best_t = t_hit; ctx.best_u = u; ctx.best_v = v;
          ctx.best_order = hit_order(inst_idx, i);
          ctx.best_prim = i;
          ctx.best_instance = inst_idx;
          ctx.best_custom = cur_custom;
          ctx.best_geom = hit_geom;
          ctx.any_hit = true;
          ctx.best_kv = true;
          vcopy3(ctx.best_obj_o, ray_o);   // this instance's object ray
          vcopy3(ctx.best_obj_d, ray_d);
          if (ctx.yield_pending && ctx.yield_t >= ctx.best_t) {
            ctx.yield_pending = false;
            ctx.yield_t = ctx.tmax;
          }
          if (cls.terminate_on_first_hit) {
            // Halt the whole walk: this hit is committed as the result and no
            // later triangle or instance may replace it.
            ctx.terminated = true;
            break;
          }
        }
      } else {  // TriAction::Yield
        uint64_t key = cand_key(inst_idx, blas_tri_off + i * kPhase2TriStride);
        if (cand_takes(ctx, t_hit, key)) {
          ctx.yield_pending = true;
          ctx.yield_key = key;
          ctx.yield_t = t_hit; ctx.yield_u = u; ctx.yield_v = v;
          ctx.yield_prim = i;
          ctx.yield_sbt = cls.yield_sbt_idx;
          ctx.yield_cb_type = cls.yield_cb_type;
          ctx.yield_instance = inst_idx;
          ctx.yield_custom = cur_custom;
          ctx.yield_geom = hit_geom;
          vcopy3(ctx.yield_obj_o, ray_o);  // object ray for AHS/IS
          vcopy3(ctx.yield_obj_d, ray_d);
        }
      }
    }
  }

  // Flat-list scenes carry no per-geometry split; report geometry 0.
  return {false, emit_lane_result(req, l, t, ctx)};
}

// ════════════════════════════════════════════════════════════════════
// Bvh4Walker
// ════════════════════════════════════════════════════════════════════

WalkResult Bvh4Walker::walk_lane(const RtuReq& req, uint32_t t, SceneView& sv,
                                  LaneState& l, PerfStats& perf) {
  // VxBvhSceneHeader: word0 = root node byte offset. Reading it is the ray's
  // first scene access, and therefore its first fetch.
  uint8_t hdr[kRtuSceneHeaderBytes];
  read_scene_bytes(sv, 0, sizeof(hdr), hdr);
  if (sv.miss) return {true, false};
  uint32_t root_off = 0;
  std::memcpy(&root_off, hdr, sizeof(uint32_t));

  const float ro[3] = { req.origin_x[t], req.origin_y[t], req.origin_z[t] };
  const float rd[3] = { req.dir_x[t],    req.dir_y[t],    req.dir_z[t]   };
  WalkCtx ctx = init_ctx(req, t, ro, rd, l);

  // Top-level (non-instanced) triangles carry no instance flags.
  walk_bvh4_subtree(sv, ro, rd, root_off, 0, 0, 0, 0, 0, ctx, perf);
  if (sv.miss) return {true, false};

  return {false, emit_lane_result(req, l, t, ctx)};
}

}}  // namespace vortex::rtu
