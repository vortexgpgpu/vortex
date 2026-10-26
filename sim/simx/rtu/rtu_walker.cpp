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
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include <VX_types.h>      // VX_RT_FLAG_*, VX_RT_CB_TYPE_*

#include "rtu_types.h"       // RtuReq, SceneView, LaneState, PerfStats,
                             // scene-format constants
#include "rtu_bvh.h"         // CW-BVH node/leaf/instance layouts
#include "rtu_isect.h"       // ray_triangle, ray_box,
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

// The hardware's per-ray traversal stack: VX_CFG_RTU_STACK_DEPTH entries shared
// by the TLAS and every BLAS below it. A walk visits the tree depth first,
// nearer child first, so every node has a rank path (its child rank at each
// level, an instance's index within its leaf) and the walk proceeds in
// ascending rank-path order. A child that does not fit on the stack is dropped;
// the walk then only descends, so when it would next pop it restarts from the
// root instead, following the rank path to the nearest dropped child: every
// node before it has been visited. A tightened best_t only culls a suffix of a
// node's ordered children, so the recorded ranks stay valid across restarts.
struct ShortStack {
  static constexpr uint32_t kLevels = 64;
  struct Entry {
    uint32_t lvl;
    uint32_t rank;
    uint32_t off;
  };
  std::vector<Entry> entries;
  std::array<uint32_t, kLevels> path{};   // rank taken at each level
  bool     ovf      = false;   // a child was dropped since the walk (re)started
  uint32_t nd_lvl   = 0;       // the nearest dropped child
  uint32_t nd_rank  = 0;
  bool     follow   = false;   // restarting: descend along the rank path
  uint32_t trl_lvl  = 0;       // ... down to this dropped child
  uint32_t trl_rank = 0;
  bool     restart  = false;   // the walk must unwind and start over
  uint64_t visits   = 0;

  void set_path(uint32_t lvl, uint32_t rank) {
    if (lvl < kLevels) path[lvl] = rank;
  }
  uint32_t get_path(uint32_t lvl) const {
    return (lvl < kLevels) ? path[lvl] : 0;
  }
};

// Depth-first walker for one BVH sub-tree under the supplied (object-space)
// ray, rooted at level root_lvl. Recurses on LeafInst so each instance's BLAS
// gets walked with its transformed ray on the same short stack. ctx accumulates
// hits/yields across the whole call tree.
//
// Returns as soon as sv.miss is raised (the caller discards the partial state
// and replays the walk once the missing line has arrived) or ss.restart is set
// (the caller restarts the walk from the scene root).
void walk_bvh4_subtree(SceneView& sv,
                       const float ro[3], const float rd[3],
                       uint32_t root_off, uint32_t root_lvl,
                       uint32_t instance_id, uint32_t custom_id,
                       uint32_t inst_flags, WalkCtx& ctx, ShortStack& ss,
                       PerfStats& perf) {
  // The ray setup's reciprocals, once per (object-space) ray.
  const float inv[3] = { ray_recip(rd[0]), ray_recip(rd[1]), ray_recip(rd[2]) };

  auto visit_leaf_tri = [&](uint32_t leaf_off, uint32_t count) {
    uint8_t hdr_buf[kVxBvhLeafHeaderBytes];
    read_scene_bytes(sv, leaf_off, sizeof(hdr_buf), hdr_buf);
    if (sv.miss) return;
    const VxBvhLeafHeader* hdr =
        reinterpret_cast<const VxBvhLeafHeader*>(hdr_buf);
    uint32_t leaf_geom = hdr->geometry_index;
    uint32_t leaf_prim_base = hdr->prim_base;  // Vulkan gl_PrimitiveID base
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
      // The interval shrinks to the committed hit: a hit is only reported
      // strictly nearer than it, so on equal t the first one found stays.
      if (!ray_triangle(ro, rd, &tri[0], &tri[3], &tri[6],
                        ctx.tmin, ctx.best_t,
                        t_hit, u, v, back_facing)) {
        continue;
      }

      TriClassify cls = classify_tri_hit(ctx.ray_flags, tri_flags,
                                          inst_flags, back_facing);
      const uint32_t hit_geom = (leaf_geom & VX_RT_HIT_GEOMETRY_MASK)
                              | hit_facing_bit(back_facing, inst_flags);
      if (cls.action == TriAction::Ignore) continue;

      if (cls.action == TriAction::Commit) {
        ctx.best_t = t_hit; ctx.best_u = u; ctx.best_v = v;
        ctx.best_prim = leaf_prim_base + i;
        ctx.best_instance = instance_id;
        ctx.best_custom = custom_id;
        ctx.best_geom = hit_geom;
        ctx.any_hit = true;
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
      static const float kRawBase[3] = { 0.f, 0.f, 0.f };
      float rel_mn[3], rel_mx[3], t_near = 0.f;
      box_rel(kRawBase, rec->aabb_min, rec->aabb_max, ro, rel_mn, rel_mx);
      ++perf.bvh_box_tests;
      if (!ray_box(rel_mn, rel_mx, inv, ctx.tmin, ctx.best_t, t_near)) {
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

  auto visit_leaf_inst = [&](uint32_t leaf_off, uint32_t count, uint32_t lvl) {
    uint8_t hdr_buf[kVxBvhLeafHeaderBytes];
    read_scene_bytes(sv, leaf_off, sizeof(hdr_buf), hdr_buf);
    if (sv.miss) return;
    uint32_t insts_off = leaf_off + kVxBvhLeafHeaderBytes;
    // A restart following its path resumes at the instance it recorded.
    const uint32_t first = ss.follow ? ss.get_path(lvl + 1) : 0;
    for (uint32_t i = first; i < count; ++i) {
      uint8_t inst_buf[kVxBvhInstanceStride];
      read_scene_bytes(sv, insts_off + i * kVxBvhInstanceStride,
                       kVxBvhInstanceStride, inst_buf);
      if (sv.miss) return;
      const VxBvhInstance* inst =
          reinterpret_cast<const VxBvhInstance*>(inst_buf);
      // Vulkan instanceCullMask: skip the instance entirely if its mask byte
      // and the ray's cull_mask have no bits in common. Both default to 0xff in
      // the no-culling path.
      if ((inst->cull_mask & ctx.ray_cull_mask & 0xffu) == 0) {
        ss.follow = false;
        continue;
      }
      // VkGeometryInstanceFlagBits packed into cull_mask bits 15..8.
      uint32_t inst_flags2 =
          (inst->cull_mask >> kRtuInstanceFlagsShift) & kRtuInstanceFlagsMask;
      float obj_ro[3], obj_rd[3];
      world_to_object_ray(inst->xform, ro, rd, obj_ro, obj_rd);
      ++perf.bvh_instance_descents;
      // The BLAS root sits at the instance level, ranked by its index.
      ss.set_path(lvl + 1, i);
      walk_bvh4_subtree(sv, obj_ro, obj_rd,
                        inst->blas_root_byte_offset, lvl + 1,
                        inst->instance_id,
                        inst->custom_id, inst_flags2,
                        ctx, ss, perf);
      if (sv.miss || ss.restart || ctx.terminated) return;
      ss.follow = false;
    }
  };

  // Safety backstop: a malformed/cyclic acceleration structure (a child offset
  // pointing back at an ancestor) would otherwise descend forever. The ceiling
  // is far above any well-formed scene's node count, so it never truncates a
  // legitimate walk.
  constexpr uint64_t kMaxNodeVisits = 1ull << 22;

  // The stack depth this sub-tree started from: popping back to it ends a BLAS.
  const size_t stack_floor = ss.entries.size();
  uint32_t current = root_off;
  uint32_t lvl = root_lvl;

  while (!ctx.terminated) {
    if (++ss.visits > kMaxNodeVisits) return;
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
      visit_leaf_inst(current, count, lvl);
      if (sv.miss || ss.restart) return;
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
        float mn[3], mx[3], rel_mn[3], rel_mx[3];
        for (int a = 0; a < 3; ++a) {
          mn[a] = quant_corner(nv.qaabb_min[i][a], nv.exp[a]);
          mx[a] = quant_corner(nv.qaabb_max[i][a], nv.exp[a]);
        }
        box_rel(nv.origin, mn, mx, ro, rel_mn, rel_mx);
        float t_near = 0.f;
        ++perf.bvh_box_tests;
        if (!ray_box(rel_mn, rel_mx, inv, ctx.tmin, ctx.best_t, t_near)) {
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

      // A restart descends into the child its path names; nearer children
      // were visited before. Culled by now, that child and every farther one
      // are done.
      uint32_t dsc = 0;
      bool descend = (hit_count != 0);
      if (ss.follow) {
        dsc = (lvl + 1 == ss.trl_lvl) ? ss.trl_rank : ss.get_path(lvl + 1);
        if (hit_count <= dsc) {
          ss.follow = false;
          descend = false;
        }
      }
      if (descend) {
        // Farther children first, so the last one dropped is the nearest.
        for (uint32_t i = hit_count; i-- > dsc + 1; ) {
          if (ss.entries.size() < VX_CFG_RTU_STACK_DEPTH) {
            ss.entries.push_back({lvl + 1, i, hits[i].offset});
          } else {
            ss.ovf     = true;
            ss.nd_lvl  = lvl + 1;
            ss.nd_rank = i;
          }
        }
        current = hits[dsc].offset;
        ++lvl;
        ss.set_path(lvl, dsc);
        if (ss.follow && lvl == ss.trl_lvl) {
          ss.follow = false;
        }
        continue;
      }
    }

    if (ctx.terminated) return;
    if (ss.ovf) {
      ++perf.bvh_stack_restarts;
      ss.restart = true;
      return;
    }
    if (ss.entries.size() == stack_floor) return;
    ss.follow = false;
    const ShortStack::Entry e = ss.entries.back();
    ss.entries.pop_back();
    current = e.off;
    lvl     = e.lvl;
    ss.set_path(e.lvl, e.rank);
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
      if (!ray_triangle(ray_o, ray_d, &tri[0], &tri[3], &tri[6],
                        ctx.tmin, ctx.best_t,
                        t_hit, u, v, back_facing)) {
        continue;
      }

      TriClassify cls = classify_tri_hit(ctx.ray_flags, tri_flags,
                                          cur_inst_flags, back_facing);
      const uint32_t hit_geom = hit_facing_bit(back_facing, cur_inst_flags);
      if (cls.action == TriAction::Ignore) continue;

      if (cls.action == TriAction::Commit) {
        ctx.best_t = t_hit; ctx.best_u = u; ctx.best_v = v;
        ctx.best_prim = i;
        ctx.best_instance = inst_idx;
        ctx.best_custom = cur_custom;
        ctx.best_geom = hit_geom;
        ctx.any_hit = true;
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
  ShortStack ss;
  ss.entries.reserve(VX_CFG_RTU_STACK_DEPTH);
  for (;;) {
    walk_bvh4_subtree(sv, ro, rd, root_off, 0, 0, 0, 0, ctx, ss, perf);
    if (sv.miss) return {true, false};
    if (!ss.restart) break;
    ss.entries.clear();
    ss.restart  = false;
    ss.ovf      = false;
    ss.follow   = true;
    ss.trl_lvl  = ss.nd_lvl;
    ss.trl_rank = ss.nd_rank;
  }

  return {false, emit_lane_result(req, l, t, ctx)};
}

}}  // namespace vortex::rtu
