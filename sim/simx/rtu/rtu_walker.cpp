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
#include <cmath>
#include <cstring>
#include <vector>

#include <VX_types.h>      // VX_RT_FLAG_*, VX_RT_CB_TYPE_*

#include "rtu_types.h"       // Slot, LaneState, PerfStats, QueueEntry,
                             // scene-format constants, SlotState
#include "rtu_bvh.h"         // CW-BVH4 node/leaf/instance layouts
#include "rtu_isect.h"       // ray_triangle, ray_aabb_intersect,
                             // affine_inverse_transform_ray
#include "rtu_classifier.h"  // classify_tri_hit, finalise_lane

namespace vortex { namespace rtu {

namespace {

// ────────────────────────────────────────────────────────────────────
// Walker-local helpers.
// ────────────────────────────────────────────────────────────────────

// Read `len` bytes from the lane's logical scene buffer at offset
// `off` (relative to line 0 + l.line_byte_off), crossing line
// boundaries as needed. Out-of-range reads silently return zeros.
void read_scene_bytes(const LaneState& l, uint32_t off,
                      uint32_t len, uint8_t* out) {
  uint32_t base = l.line_byte_off + off;
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t pos = base + i;
    uint32_t li  = pos / VX_CFG_MEM_BLOCK_SIZE;
    uint32_t bo  = pos % VX_CFG_MEM_BLOCK_SIZE;
    out[i] = (li < kRtuMaxLinesPerLane) ? l.line_data[li][bo] : uint8_t(0);
  }
}

// CW-BVH4: reconstruct a child AABB from quantized representation.
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

// Per-lane BVH4 traversal accumulator. Shared across recursive
// sub-tree walks so a BLAS hit can update the same best_t that culls
// later TLAS-side AABB tests.
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
  // Object-space ray of the committed hit (best_obj_*) and the yield
  // candidate (yield_obj_*). Set to {ro,rd} at the leaf that wins;
  // equals the world ray at the top level (no instance).
  float best_obj_o[3],  best_obj_d[3];
  float yield_obj_o[3], yield_obj_d[3];
};

// Depth-first walker for one BVH4 sub-tree under the supplied
// (object-space) ray. Recurses on LeafInst so each instance's BLAS
// gets walked with its transformed ray. ctx accumulates hits/yields
// across the whole call tree.
//
// The functional traversal stack is unbounded so the oracle never misses a
// hit; the HW short-stack (VX_CFG_RTU_STACK_DEPTH) overflow is counted as a
// trail-based RESTART and charged in the cost model.
// instance_id is the TLAS-assigned ID the caller wants recorded for
// any hit found in this sub-tree.
void walk_bvh4_subtree(LaneState& l,
                       const float ro[3], const float rd[3],
                       uint32_t root_off, uint32_t instance_id,
                       uint32_t custom_id, uint32_t inst_flags,
                       WalkCtx& ctx, PerfStats& perf) {
  auto visit_leaf_tri = [&](uint32_t leaf_off, uint32_t count) {
    uint8_t hdr_buf[kVxBvhLeafHeaderBytes];
    read_scene_bytes(l, leaf_off, sizeof(hdr_buf), hdr_buf);
    const VxBvhLeafHeader* hdr =
        reinterpret_cast<const VxBvhLeafHeader*>(hdr_buf);
    uint32_t leaf_geom = hdr->geometry_index;
    uint32_t leaf_prim_base = hdr->prim_base;  // Vulkan gl_PrimitiveID base
    uint32_t tris_off = leaf_off + kVxBvhLeafHeaderBytes;
    for (uint32_t i = 0; i < count; ++i) {
      if (ctx.terminated) return;
      uint8_t tri_buf[kVxBvhTriStride];
      read_scene_bytes(l, tris_off + i * kVxBvhTriStride,
                       kVxBvhTriStride, tri_buf);
      const float* tri = reinterpret_cast<const float*>(tri_buf);
      uint32_t tri_flags = 0;
      std::memcpy(&tri_flags, tri_buf + kPhase2TriFlagsOff,
                  sizeof(uint32_t));

      float t_hit = 0.f, u = 0.f, v = 0.f;
      bool back_facing = false;
      ++perf.bvh_tri_tests;
      if (!ray_triangle(ro, rd, &tri[0], &tri[3], &tri[6],
                        ctx.tmin, ctx.tmax,
                        t_hit, u, v, back_facing)) {
        continue;
      }

      TriClassify cls = classify_tri_hit(ctx.ray_flags, tri_flags,
                                          inst_flags, back_facing);
      if (cls.action == TriAction::Ignore) continue;

      if (cls.action == TriAction::Commit) {
        if (t_hit < ctx.best_t) {
          ctx.best_t = t_hit; ctx.best_u = u; ctx.best_v = v;
          ctx.best_prim = leaf_prim_base + i;
          ctx.best_instance = instance_id;
          ctx.best_custom = custom_id;
          ctx.best_geom = leaf_geom;
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
        }
      } else {  // TriAction::Yield
        if (t_hit < ctx.best_t && t_hit < ctx.yield_t) {
          ctx.yield_pending = true;
          ctx.yield_t = t_hit; ctx.yield_u = u; ctx.yield_v = v;
          ctx.yield_prim = leaf_prim_base + i;
          ctx.yield_instance = instance_id;
          ctx.yield_custom = custom_id;
          ctx.yield_geom = leaf_geom;
          ctx.yield_sbt = cls.yield_sbt_idx;
          ctx.yield_cb_type = cls.yield_cb_type;
          vcopy3(ctx.yield_obj_o, ro);  // object-space ray for AHS/IS
          vcopy3(ctx.yield_obj_d, rd);
        }
      }
    }
  };

  // Procedural-AABB leaf. Each record is a custom
  // primitive's bounding box; a ray-AABB hit yields an IS callback so the
  // kernel's intersection shader computes the real hit. The candidate t is
  // the AABB entry parameter (a lower bound); the IS supplies the true t
  // via VX_RT_HIT_T, committed on ACCEPT (see rtu_core CB_ACTION drain).
  auto visit_leaf_proc = [&](uint32_t leaf_off, uint32_t count) {
    uint8_t hdr_buf[kVxBvhLeafHeaderBytes];
    read_scene_bytes(l, leaf_off, sizeof(hdr_buf), hdr_buf);
    const VxBvhLeafHeader* hdr =
        reinterpret_cast<const VxBvhLeafHeader*>(hdr_buf);
    uint32_t leaf_sbt =
        (hdr->flags >> kVxBvhLeafSbtIdxShift) & kVxBvhLeafSbtIdxMask;
    uint32_t aabbs_off = leaf_off + kVxBvhLeafHeaderBytes;
    for (uint32_t i = 0; i < count; ++i) {
      if (ctx.terminated) return;
      uint8_t rec_buf[sizeof(VxBvhProcAabb)];
      read_scene_bytes(l, aabbs_off + i * uint32_t(sizeof(VxBvhProcAabb)),
                       sizeof(rec_buf), rec_buf);
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
      if (t_near < ctx.best_t && t_near < ctx.yield_t) {
        ctx.yield_pending = true;
        ctx.yield_t = t_near; ctx.yield_u = 0.f; ctx.yield_v = 0.f;
        ctx.yield_prim = i;
        ctx.yield_instance = instance_id;
        ctx.yield_custom = custom_id;
        ctx.yield_geom = hdr->geometry_index;
        ctx.yield_sbt = leaf_sbt;
        ctx.yield_cb_type = VX_RT_CB_TYPE_PROC;
        vcopy3(ctx.yield_obj_o, ro);
        vcopy3(ctx.yield_obj_d, rd);
      }
    }
  };

  auto visit_leaf_inst = [&](uint32_t leaf_off, uint32_t count) {
    uint32_t insts_off = leaf_off + kVxBvhLeafHeaderBytes;
    for (uint32_t i = 0; i < count; ++i) {
      uint8_t inst_buf[kVxBvhInstanceStride];
      read_scene_bytes(l, insts_off + i * kVxBvhInstanceStride,
                       kVxBvhInstanceStride, inst_buf);
      const VxBvhInstance* inst =
          reinterpret_cast<const VxBvhInstance*>(inst_buf);
      // Vulkan instanceCullMask: skip the instance entirely if
      // its mask byte and the ray's cull_mask have no bits in
      // common. Both default to 0xff in the no-culling path
      // (lavapipe / lvp_nir lowers a missing cullMask to 0xff and
      // scene generators set the instance byte the same way), so
      // existing tests pass unchanged.
      if ((inst->cull_mask & ctx.ray_cull_mask & 0xffu) == 0) continue;
      // VkGeometryInstanceFlagBits packed into cull_mask bits 15..8.
      uint32_t inst_flags =
          (inst->cull_mask >> kRtuInstanceFlagsShift) & kRtuInstanceFlagsMask;
      float obj_ro[3], obj_rd[3];
      affine_inverse_transform_ray(inst->xform, ro, rd, obj_ro, obj_rd);
      ++perf.bvh_instance_descents;
      walk_bvh4_subtree(l, obj_ro, obj_rd,
                        inst->blas_root_byte_offset,
                        inst->instance_id,
                        inst->custom_id, inst_flags,
                        ctx, perf);
    }
  };

  // SimX is the correctness oracle: keep an UNBOUNDED traversal stack so deep
  // sub-trees are never dropped. The HW short-stack is only
  // VX_CFG_RTU_STACK_DEPTH deep and trail-restarts to re-find subtrees it had to
  // evict — it visits the same leaves, just at extra cost.
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
    read_scene_bytes(l, current, sizeof(kind_buf), kind_buf);
    uint32_t kind_word = 0;
    std::memcpy(&kind_word, kind_buf, sizeof(uint32_t));
    uint32_t kind  = kind_word & kVxBvhKindMask;
    uint32_t count = (kind_word >> kVxBvhCountShift) & kVxBvhCountMask;

    if (kind == kVxBvhKindLeafTri) {
      ++perf.bvh_leaves_fetched;
      if (!(ctx.ray_flags & VX_RT_FLAG_SKIP_TRIANGLES)) {
        visit_leaf_tri(current, count);
      }
    } else if (kind == kVxBvhKindLeafInst) {
      ++perf.bvh_leaves_fetched;
      visit_leaf_inst(current, count);
    } else if (kind == kVxBvhKindLeafProc) {
      ++perf.bvh_leaves_fetched;
      // SKIP_AABBS: symmetric gate with SKIP_TRIANGLES. Otherwise
      // ray-test each procedural AABB and yield IS for the closest hit.
      if (!(ctx.ray_flags & VX_RT_FLAG_SKIP_AABBS)) {
        visit_leaf_proc(current, count);
      }
    } else if (kind == kVxBvhKindInternal) {
      ++perf.bvh_nodes_fetched;
      // Width-generic decode: CW-BVH4 (64 B) or CW-BVH6 (96 B) selected by
      // scene_kind. Both decode into VxBvhNodeView so the box-test loop and
      // the box-PE cycle model are fan-out independent.
      VxBvhNodeView nv;
#if VX_CFG_RTU_BVH_WIDTH == 6
      uint8_t node_buf[sizeof(VxBvh6InternalNode)];
      read_scene_bytes(l, current, sizeof(node_buf), node_buf);
      decode_bvh6_node(
          reinterpret_cast<const VxBvh6InternalNode*>(node_buf), count, nv);
#else
      uint8_t node_buf[sizeof(VxBvhInternalNode)];
      read_scene_bytes(l, current, sizeof(node_buf), node_buf);
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
                                ctx.tmin, ctx.best_t, t_near)) {
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

// End-of-lane finalise: translates the accumulated walk state into
// LaneState writes. Returns true iff a CB_YIELD should be queued
// for this lane (the actual queue push is deferred until the slot
// finishes draining PE cycles — see the orchestrator). All
// the data needed to reconstruct the QueueEntry lives in LaneState
// (cb_pending / cb_type / sbt_idx / cand_t / cand_u / cand_v /
// cand_prim), so the orchestrator can scan lanes and push without
// the walker carrying intermediate state.
bool emit_lane_result(Slot& s, LaneState& l, uint32_t t, uint32_t /*slot_idx*/,
                      bool     any_hit,
                      float    best_t, float    best_u,
                      float    best_v, uint32_t best_prim,
                      uint32_t best_instance, uint32_t best_custom,
                      uint32_t best_geom,
                      bool     yield_pending,
                      float    yield_t, float    yield_u,
                      float    yield_v, uint32_t yield_prim,
                      uint32_t yield_sbt, uint32_t yield_cb_type,
                      uint32_t yield_instance, uint32_t yield_custom,
                      uint32_t yield_geom,
                      const float best_obj_o[3], const float best_obj_d[3],
                      const float yield_obj_o[3], const float yield_obj_d[3]) {
  l.hit       = any_hit;
  l.hit_t     = best_t;
  l.hit_u     = best_u;
  l.hit_v     = best_v;
  l.hit_prim  = best_prim;
  l.hit_instance_id = any_hit ? best_instance
                              : (yield_pending ? yield_instance : 0u);
  l.hit_instance_custom = any_hit ? best_custom
                                  : (yield_pending ? yield_custom : 0u);
  l.hit_geometry  = best_geom;
  l.cand_geometry = yield_geom;
  // Stash the committed + candidate object-space rays for the regfile
  // writeback in rtu_core/rtu_unit.
  vcopy3(l.hit_obj_o,  best_obj_o);
  vcopy3(l.hit_obj_d,  best_obj_d);
  vcopy3(l.cand_obj_o, yield_obj_o);
  vcopy3(l.cand_obj_d, yield_obj_d);

  LaneAction action = finalise_lane(s.req.flags[t], any_hit,
                                     yield_pending, yield_cb_type);
  switch (action) {
  case LaneAction::TerminalHit:
  case LaneAction::TerminalMiss:
    return false;
  case LaneAction::YieldAhs:
  case LaneAction::YieldIs:
    l.cb_pending = true;
    l.cb_type    = yield_cb_type;
    l.sbt_idx    = yield_sbt;
    l.cand_t     = yield_t;
    l.cand_u     = yield_u;
    l.cand_v     = yield_v;
    l.cand_prim  = yield_prim;
    l.cand_instance = yield_instance;
    l.cand_custom   = yield_custom;
    return true;
  case LaneAction::YieldChs:
    l.cb_pending = true;
    l.cb_type    = VX_RT_CB_TYPE_CHS;
    l.sbt_idx    = 0;
    l.cand_t     = best_t;
    l.cand_u     = best_u;
    l.cand_v     = best_v;
    l.cand_prim  = best_prim;
    l.cand_instance = best_instance;
    l.cand_custom   = best_custom;
    // A CHS candidate IS the committed hit, so gl_GeometryIndexEXT
    // reads the committed leaf's geometry (not the stale yield_geom).
    l.cand_geometry = best_geom;
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

}  // namespace

// ════════════════════════════════════════════════════════════════════
// FlatWalker
// ════════════════════════════════════════════════════════════════════

bool FlatWalker::walk_lane(Slot& s, LaneState& l, uint32_t t,
                            uint32_t slot_idx) {
  // TLAS scenes walk one or more instances; each instance
  // points at a BLAS (a triangle list) and (optionally) applies an
  // object→world affine transform.
  uint32_t num_instances = 1;
#ifdef VX_CFG_RTU_TLAS_ENABLE
  if (l.instance_count == 0) {
    l.hit = false;
    return false;
  }
  num_instances = l.instance_count;
#else
  if (l.triangle_count == 0) {
    l.hit = false;
    return false;
  }
#endif

  float best_t = s.req.tmax[t];
  float best_u = 0.f;
  float best_v = 0.f;
  uint32_t best_prim = 0;
  uint32_t best_instance = 0;
  uint32_t best_custom = 0;
  bool any_hit = false;
  bool yield_pending = false;
  // Init yield_t to tmax so the first non-opaque candidate always
  // wins the "closer than current pending candidate" check
  // (single-closest-yield).
  float yield_t = s.req.tmax[t];
  float yield_u = 0.f, yield_v = 0.f;
  uint32_t yield_prim = 0;
  uint32_t yield_sbt  = 0;
  uint32_t yield_cb_type = VX_RT_CB_TYPE_ANYHIT;
  uint32_t yield_instance = 0;
  uint32_t yield_custom = 0;
  float ro[3] = { s.req.origin_x[t], s.req.origin_y[t], s.req.origin_z[t] };
  float rd[3] = { s.req.dir_x[t],    s.req.dir_y[t],    s.req.dir_z[t]   };
  // Object-space ray of the committed / candidate hit. Default = world
  // ray (TriList); set to the transformed ray below when the hit is
  // under a TLAS instance.
  float best_obj_o[3]  = { ro[0], ro[1], ro[2] };
  float best_obj_d[3]  = { rd[0], rd[1], rd[2] };
  float yield_obj_o[3] = { ro[0], ro[1], ro[2] };
  float yield_obj_d[3] = { rd[0], rd[1], rd[2] };
  uint8_t tri_buf[kPhase2TriStride];

  // Scan ALL instances (not stopping at the first with a yield candidate): a
  // later instance may hold a NEARER opaque hit (which occludes an alpha-tested
  // candidate) or a nearer non-opaque candidate. The inner logic already keeps
  // the single closest opaque + closest non-opaque via the best_t / yield_t
  // monotone compares, so scanning every instance is order-independent and
  // matches the Bvh4Walker whole-tree walk. Exception: a terminate-on-first-hit
  // commit halts the entire walk (matching Bvh4Walker's ctx.terminated), so no
  // later instance can substitute a different reported hit.
  bool terminated = false;
  for (uint32_t inst_idx = 0; inst_idx < num_instances && !terminated; ++inst_idx) {
    uint32_t blas_tri_off   = kRtuSceneHeaderBytes;
    uint32_t blas_tri_count = l.triangle_count;
    uint32_t cur_custom     = 0;
    uint32_t cur_inst_flags = 0;
    float ray_o[3] = { ro[0], ro[1], ro[2] };
    float ray_d[3] = { rd[0], rd[1], rd[2] };
#ifdef VX_CFG_RTU_TLAS_ENABLE
    {
      uint32_t inst_off = kRtuSceneHeaderBytes
                        + inst_idx * kRtuInstanceStride;
      uint8_t inst_buf[kRtuInstanceStride];
      read_scene_bytes(l, inst_off, sizeof(inst_buf), inst_buf);
      // Vulkan instanceCullMask gate — skip the entire
      // instance (transform + BLAS scan) before doing the
      // affine ray transform when masks don't overlap. Same
      // semantics as the BVH4 LeafInst gate in visit_leaf_inst.
      uint32_t inst_cull_mask = 0;
      std::memcpy(&inst_cull_mask,
                  inst_buf + kRtuInstanceCullMaskOff,
                  sizeof(uint32_t));
      if ((inst_cull_mask & s.req.cull_mask[t] & 0xffu) == 0) continue;
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
      // World→object ray transform. For pure rotation +
      // translation the t parameter is preserved, so the
      // BLAS-reported hit_t is also the world hit_t.
      affine_inverse_transform_ray(xform, ro, rd, ray_o, ray_d);
      uint8_t blas_hdr[4];
      read_scene_bytes(l, blas_byte_off, sizeof(blas_hdr), blas_hdr);
      uint32_t bcount = 0;
      std::memcpy(&bcount, blas_hdr, sizeof(uint32_t));
      if (bcount > kRtuMaxTrisPerScene) bcount = kRtuMaxTrisPerScene;
      blas_tri_count = bcount;
      blas_tri_off   = blas_byte_off + kRtuSceneHeaderBytes;
    }
#endif
    uint32_t n_tris = std::min(blas_tri_count, kRtuMaxTrisPerScene);

    // Walk the *full* triangle list. Track best opaque hit and the
    // closest non-opaque candidate separately. If a non-opaque
    // candidate ends up closer than the best opaque, yield it;
    // otherwise the opaque commits and no AHS fires (alpha-test fast
    // path).
    //
    // SKIP_TRIANGLES bails the whole leaf-tri scan. (Flat-list
    // scenes only have tri leaves, so SKIP_AABBS is a no-op here.)
    const uint32_t ray_flags = s.req.flags[t];
    if (ray_flags & VX_RT_FLAG_SKIP_TRIANGLES) {
      // Skip all triangles; remaining hit/yield logic falls through
      // with no_hit and no pending candidate.
    } else
    for (uint32_t i = 0; i < n_tris; ++i) {
      uint32_t tri_off = blas_tri_off + i * kPhase2TriStride;
      read_scene_bytes(l, tri_off, kPhase2TriStride, tri_buf);
      const float* tri = reinterpret_cast<const float*>(tri_buf);
      uint32_t tri_flags = 0;
      std::memcpy(&tri_flags, tri_buf + kPhase2TriFlagsOff,
                  sizeof(uint32_t));

      float t_hit = 0.f, u = 0.f, v = 0.f;
      bool back_facing = false;
      ++perf_.bvh_tri_tests;
      // Test against ray.tmax (not best_t) so an opaque hit committed
      // earlier in this walk doesn't pre-cull a non-opaque candidate
      // that might survive an ACCEPT.
      if (!ray_triangle(ray_o, ray_d, &tri[0], &tri[3], &tri[6],
                        s.req.tmin[t], s.req.tmax[t],
                        t_hit, u, v, back_facing)) {
        continue;
      }

      TriClassify cls = classify_tri_hit(ray_flags, tri_flags,
                                          cur_inst_flags, back_facing);
      if (cls.action == TriAction::Ignore) continue;

      if (cls.action == TriAction::Commit) {
        if (t_hit < best_t) {
          best_t = t_hit; best_u = u; best_v = v; best_prim = i;
          best_instance = inst_idx;
          best_custom = cur_custom;
          any_hit = true;
          vcopy3(best_obj_o, ray_o);   // this instance's object ray
          vcopy3(best_obj_d, ray_d);
          if (yield_pending && yield_t >= best_t) {
            yield_pending = false;
            yield_t = s.req.tmax[t];
          }
          if (cls.terminate_on_first_hit) {
            // Halt the whole walk: this hit is committed as the result and no
            // later triangle or instance may replace it.
            s.req.tmax[t] = best_t;
            terminated = true;
            break;
          }
        }
      } else {  // TriAction::Yield
        if (t_hit < best_t && t_hit < yield_t) {
          yield_pending = true;
          yield_t = t_hit; yield_u = u; yield_v = v; yield_prim = i;
          yield_sbt = cls.yield_sbt_idx;
          yield_cb_type = cls.yield_cb_type;
          yield_instance = inst_idx;
          yield_custom = cur_custom;
          vcopy3(yield_obj_o, ray_o);  // object ray for AHS/IS
          vcopy3(yield_obj_d, ray_d);
        }
      }
    }
  }

  // Flat-list scenes carry no per-geometry split; report geometry 0.
  return emit_lane_result(s, l, t, slot_idx,
                          any_hit, best_t, best_u, best_v, best_prim,
                          best_instance, best_custom, 0u,
                          yield_pending, yield_t, yield_u, yield_v,
                          yield_prim, yield_sbt, yield_cb_type,
                          yield_instance, yield_custom, 0u,
                          best_obj_o, best_obj_d, yield_obj_o, yield_obj_d);
}

// ════════════════════════════════════════════════════════════════════
// Bvh4Walker
// ════════════════════════════════════════════════════════════════════

bool Bvh4Walker::walk_lane(Slot& s, LaneState& l, uint32_t t,
                            uint32_t slot_idx) {
  const float ro[3] = { s.req.origin_x[t], s.req.origin_y[t], s.req.origin_z[t] };
  const float rd[3] = { s.req.dir_x[t],    s.req.dir_y[t],    s.req.dir_z[t]   };

  WalkCtx ctx;
  ctx.tmin = s.req.tmin[t];
  ctx.tmax = s.req.tmax[t];
  ctx.ray_flags = s.req.flags[t];
  ctx.ray_cull_mask = s.req.cull_mask[t];
  ctx.terminated = false;
  ctx.best_t = ctx.tmax;
  ctx.best_u = 0.f; ctx.best_v = 0.f;
  ctx.best_prim = 0; ctx.best_instance = 0; ctx.best_custom = 0; ctx.best_geom = 0;
  ctx.any_hit = false;
  ctx.yield_pending = false;
  ctx.yield_t = ctx.tmax; ctx.yield_u = 0.f; ctx.yield_v = 0.f;
  ctx.yield_prim = 0; ctx.yield_sbt = 0;
  ctx.yield_cb_type = VX_RT_CB_TYPE_ANYHIT;
  ctx.yield_instance = 0; ctx.yield_custom = 0; ctx.yield_geom = 0;
  // Default object ray = world ray (overwritten at a BLAS leaf if the hit
  // is under an instance).
  vcopy3(ctx.best_obj_o, ro);  vcopy3(ctx.best_obj_d, rd);
  vcopy3(ctx.yield_obj_o, ro); vcopy3(ctx.yield_obj_d, rd);

  // Top-level (non-instanced) triangles carry no instance flags.
  walk_bvh4_subtree(l, ro, rd, l.bvh_root_offset, 0, 0, 0, ctx, perf_);

  return emit_lane_result(s, l, t, slot_idx,
                          ctx.any_hit, ctx.best_t, ctx.best_u, ctx.best_v,
                          ctx.best_prim, ctx.best_instance, ctx.best_custom,
                          ctx.best_geom,
                          ctx.yield_pending, ctx.yield_t, ctx.yield_u,
                          ctx.yield_v, ctx.yield_prim, ctx.yield_sbt,
                          ctx.yield_cb_type, ctx.yield_instance, ctx.yield_custom,
                          ctx.yield_geom,
                          ctx.best_obj_o, ctx.best_obj_d,
                          ctx.yield_obj_o, ctx.yield_obj_d);
}

}}  // namespace vortex::rtu
