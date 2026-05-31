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
//
// PRISM RtuCore — Phase 1 implementation.
//
// Phase 1 scene format (host-defined, see tests/regression/rtu_smoke/):
//   struct simple_scene_t {
//       uint32_t triangle_count;     // [byte 0]
//       uint32_t reserved[3];        // [byte 4..15] — align to 16
//       float    triangles[N][9];    // v0xyz, v1xyz, v2xyz
//   };
// For Phase 1 smoke we fix N=1 so the entire scene fits in one 64 B cache
// line (16 + 36 = 52 bytes). Phase 1.5+ extends to multi-triangle scenes
// and Phase 2 swaps to vk_bvh.h CW-BVH4 layout.

#include "rtu_core.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <unordered_map>
#include "rtu_types.h"
#include "rtu_bvh.h"
#include "rtu_isect.h"   // §step-3: ray_triangle / ray_aabb_intersect /
                         //          affine_inverse_transform_ray
#include "cluster.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;
using namespace vortex::rtu;

namespace {

// §step-2 refactor: scene-format constants, math primitives (Vec3,
// dot, cross), and the inline geometry helpers (tri_list_bytes,
// tlas_bytes, lines_for_bytes, lines_for_scene) now live in
// rtu_types.h (vortex::rtu namespace, pulled in via `using namespace
// vortex::rtu;` above). All names used here resolve to those.

// §step-3: ray_triangle, ray_aabb_intersect, affine_inverse_transform_ray
// moved to rtu_isect.{h,cpp}. The file-local float3/dot_/cross_ aliases
// added in step 2 are no longer needed (their only callers were those
// three functions). reconstruct_child_aabb stays here for now — it's
// still file-local; a follow-up could promote it to rtu_bvh.h.

// Phase 4 — reconstruct a CW-BVH4 child AABB from the quantized
// representation: real = origin + qaabb * 2^exp (per axis).
inline void reconstruct_child_aabb(const float origin[3], const int8_t exp[3],
                                   const uint8_t qmin[3], const uint8_t qmax[3],
                                   float out_mn[3], float out_mx[3]) {
  for (int i = 0; i < 3; ++i) {
    // ldexp(1, exp) = 2^exp; allows negative exp for sub-unit scales.
    float scale = std::ldexp(1.0f, exp[i]);
    out_mn[i] = origin[i] + static_cast<float>(qmin[i]) * scale;
    out_mx[i] = origin[i] + static_cast<float>(qmax[i]) * scale;
  }
}

} // namespace

// ════════════════════════════════════════════════════════════════════
// RtuCore::Impl
// ════════════════════════════════════════════════════════════════════
//
// State machine per slot:
//   ISSUE → AWAIT(per-lane cache-line fetches) → COMPUTE → RESP
//
// Phase 1 simplification: lanes share the same scene_root in the smoke
// test (single AS per dispatch), so we issue at most NUM_THREADS distinct
// cache-line addresses per request (coalesced when identical).

class RtuCore::Impl {
public:
  // §step-2 refactor: Slot, LaneState, SlotState (formerly State enum)
  // and QueueEntry now live in rtu_types.h (vortex::rtu namespace).
  // The `using namespace vortex::rtu;` at file scope brings them in
  // unqualified; legacy `State::ISSUE` references below are migrated
  // to `SlotState::ISSUE`.
  using State = SlotState;  // local alias to avoid touching every
                            // State::ISSUE etc. in this file

  explicit Impl(RtuCore* simobject)
    : simobject_(simobject)
    , slots_(VX_CFG_RTU_CONTEXT_POOL)
  {}

  // §8.9 stats dump. Opt-in via VX_RTU_STATS env var (any non-empty
  // value). Prints to stderr at destruction so a smoke test that
  // wants observability can set the var on its `env` line. Silent
  // by default — most regression tests don't want the noise.
  ~Impl() {
    const char* env = std::getenv("VX_RTU_STATS");
    if (env == nullptr || env[0] == '\0') return;
    const auto& p = perf_stats_;
    std::fprintf(stderr, "[rtu-stats] rays_issued=%llu rays_hit=%llu rays_miss=%llu "
                         "mem_reads=%llu bvh_nodes=%llu bvh_leaves=%llu "
                         "instance_descents=%llu box_tests=%llu tri_tests=%llu "
                         "cb_ahs=%llu cb_chs=%llu cb_miss=%llu cb_is=%llu "
                         "reformation_yields=%llu coh_hits=%llu coh_misses=%llu\n",
                 (unsigned long long)p.rays_issued,
                 (unsigned long long)p.rays_hit,
                 (unsigned long long)p.rays_miss,
                 (unsigned long long)p.mem_reads,
                 (unsigned long long)p.bvh_nodes_fetched,
                 (unsigned long long)p.bvh_leaves_fetched,
                 (unsigned long long)p.bvh_instance_descents,
                 (unsigned long long)p.bvh_box_tests,
                 (unsigned long long)p.bvh_tri_tests,
                 (unsigned long long)p.ahs_callbacks,
                 (unsigned long long)p.chs_callbacks,
                 (unsigned long long)p.miss_callbacks,
                 (unsigned long long)p.is_callbacks,
                 (unsigned long long)p.reformation_yields,
                 (unsigned long long)p.coherency_hits,
                 (unsigned long long)p.coherency_misses);
  }

  void reset() {
    for (auto& s : slots_) {
      s.in_use = false;
      s.state = State::ISSUE;
      s.pending_mem = 0;
      for (auto& l : s.lanes) {
        l.active = false;
        l.line_filled.fill(false);
        l.line_issued.fill(false);
        l.lines_needed  = 1;
        l.lines_filled  = 0;
        l.lines_issued  = 0;
        l.header_parsed = false;
        l.triangle_count = 0;
        l.scene_kind    = kRtuSceneKindTriList;
        l.instance_count = 0;
        l.hit_instance_id = 0;
      }
    }
    pending_mem_.clear();
    ahs_queue_.clear();
    warp_cb_inflight_.fill(false);
    perf_stats_ = RtuCore::PerfStats();
    next_tag_ = 0;
    last_compute_signature_ = 0;
  }

  void tick() {
    drain_mem_rsp();
    drain_requests();
    issue_memory();
    compute_intersections();
    reformation_dispatch();
    emit_completions();
  }

  void drain_requests() {
    for (auto& ch : simobject_->rtu_req_in) {
      while (!ch.empty()) {
        const RtuReq& req = ch.peek();
        if (req.kind == RtuReqKind::CB_ACTION) {
          // Phase 3-A2: per-lane CB_RET action. Each active lane in the
          // packet carries its own slot handle (cb_handle, written by
          // SfuUnit at process_cb_ret time from VX_RT_CB_HANDLE). The
          // gathered batch may have routed lanes from MULTIPLE slots into
          // one virtual warp, so we look each lane up by handle and apply
          // ACCEPT/IGNORE/TERMINATE on its own slot.
          for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
            if (((req.tmask_bits >> t) & 1u) == 0) continue;
            uint32_t handle = req.cb_handle[t];
            if (handle >= slots_.size()) continue;
            auto& s = slots_[handle];
            if (!s.in_use || s.state != State::IN_QUEUE) continue;
            LaneState& l = s.lanes[t];
            if (!l.cb_pending) continue;
            uint32_t action = req.cb_action[t];
            if (action == VX_RT_CB_ACCEPT || action == VX_RT_CB_TERMINATE) {
              l.hit      = true;
              l.hit_t    = l.cand_t;
              l.hit_u    = l.cand_u;
              l.hit_v    = l.cand_v;
              l.hit_prim = l.cand_prim;
            }
            // VX_RT_CB_IGNORE: leave best_hit unchanged. Phase 3-A2
            // minimum has single-yield-per-lane traversal, so the slot
            // transitions straight to RESP (a richer multi-yield
            // traversal would loop back to COMPUTE for the lane's
            // remaining candidates).
            //
            // Phase 5 VX_RT_CB_DONE: the CHS dispatcher has finished
            // shading the already-committed hit; no hit-state mutation,
            // just drain so the slot can transition to RESP.
            l.cb_pending = false;
            // If this was the last cb_pending lane in the slot, the slot
            // is fully resolved → RESP. Otherwise stay IN_QUEUE for the
            // next batched dispatch.
            bool any_pending = false;
            for (auto const& ll : s.lanes) {
              if (ll.cb_pending) { any_pending = true; break; }
            }
            if (!any_pending) s.state = State::RESP;
          }
          // Clear this warp's "callback in flight" gate so the next
          // queued CB_YIELD for the same warp (e.g. the second SBT
          // group in the divergent-SBT smoke) can be emitted.
          warp_cb_inflight_[req.warp_id] = false;
          DT(3, "rtu-core cb_action applied (queue): tmask=0x"
                << std::hex << req.tmask_bits << std::dec);
          ch.pop();
          continue;
        }
        // TRACE_NEW path (Phase 1).
        int free_idx = -1;
        for (size_t i = 0; i < slots_.size(); ++i) {
          if (!slots_[i].in_use) { free_idx = (int)i; break; }
        }
        if (free_idx < 0) break;
        auto& s = slots_[free_idx];
        s.in_use = true;
        s.state  = State::ISSUE;
        s.req    = req;
        s.pending_mem = 0;
        for (auto& l : s.lanes) {
          l.active = false;
          l.line_filled.fill(false);
          l.line_issued.fill(false);
          l.lines_needed   = 1;
          l.lines_filled   = 0;
          l.lines_issued   = 0;
          l.header_parsed  = false;
          l.triangle_count = 0;
          l.scene_kind     = kRtuSceneKindTriList;
          l.instance_count = 0;
          l.hit_instance_id = 0;
          l.cb_pending = false;
          l.hit = false;
        }
        uint32_t first_active = uint32_t(-1);
        for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
          if (s.req.tmask_bits & (1u << t)) {
            s.lanes[t].active = true;
            if (first_active == uint32_t(-1)) first_active = t;
          }
        }
        // §8.9 octant signature from first active lane's ray direction.
        if (first_active != uint32_t(-1)) {
          uint8_t sig = 0;
          if (s.req.dir_x[first_active] < 0.f) sig |= 0x1;
          if (s.req.dir_y[first_active] < 0.f) sig |= 0x2;
          if (s.req.dir_z[first_active] < 0.f) sig |= 0x4;
          s.coh_signature = sig;
        }
        ch.pop();
        ++perf_stats_.rays_issued;
        DT(3, "rtu-core accept: tag=" << s.req.tag);
      }
    }
  }

  void issue_memory() {
    if (simobject_->dcache_req_out.empty()) return;
    auto& port = simobject_->dcache_req_out.at(0);
    for (auto& s : slots_) {
      if (!s.in_use) continue;
      if (s.state != State::ISSUE && s.state != State::AWAIT) continue;
      // Phase 4 multi-line fetch. Each active lane issues line 0 first;
      // once the header drains (drain_mem_rsp parses triangle_count and
      // sets lines_needed), body lines 1..lines_needed-1 are issued in
      // subsequent ticks. Stay in ISSUE while any active lane still has
      // work to schedule; otherwise drop to AWAIT until rsps drain.
      bool all_issued = true;
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        auto& l = s.lanes[t];
        if (!l.active) continue;
        if (l.lines_issued >= l.lines_needed) continue;
        all_issued = false;
        if (port.full()) break;
        uint32_t line_idx = l.lines_issued;
        if (line_idx == 0) {
          // Cache-line-aligned header for the smoke-test scene layout.
          uint64_t addr = uint64_t(s.req.scene_root[t]);
          uint64_t line = addr & kRtuLineMask;
          l.line_byte_off = uint32_t(addr - line);
        }
        // Subsequent lines walk sequentially from line 0.
        uint64_t base_addr  = uint64_t(s.req.scene_root[t]) & kRtuLineMask;
        uint64_t line_addr  = base_addr + uint64_t(line_idx) * VX_CFG_MEM_BLOCK_SIZE;
        uint32_t tag = next_tag_++;
        MemReq m;
        m.addr    = line_addr;
        m.op      = MemOp::LD;
        m.tag     = tag;
        m.hart_id = 0;
        m.uuid    = s.req.uuid;
        port.send(m);
        pending_mem_[tag] = PendingFill{ uint32_t(&s - &slots_[0]),
                                         uint8_t(t),
                                         uint8_t(line_idx) };
        l.line_issued[line_idx] = true;
        ++l.lines_issued;
        ++s.pending_mem;
        ++perf_stats_.mem_reads;
        // Recompute all_issued on remaining lanes for next loop entry.
        all_issued = true;
        for (uint32_t u = 0; u < VX_CFG_NUM_THREADS; ++u) {
          const auto& ll = s.lanes[u];
          if (ll.active && ll.lines_issued < ll.lines_needed) {
            all_issued = false; break;
          }
        }
      }
      if (all_issued) {
        s.state = (s.pending_mem == 0) ? State::COMPUTE : State::AWAIT;
      } else if (s.state == State::AWAIT) {
        // We're back in ISSUE because lines_needed grew after a header
        // drain — issue more next tick.
        s.state = State::ISSUE;
      }
    }
  }

  void drain_mem_rsp() {
    for (auto& ch : simobject_->dcache_rsp_in) {
      while (!ch.empty()) {
        auto& rsp = ch.peek();
        auto it = pending_mem_.find(uint32_t(rsp.tag));
        if (it == pending_mem_.end()) {
          ch.pop();
          continue;
        }
        PendingFill pf = it->second;
        pending_mem_.erase(it);
        Slot& s = slots_[pf.slot_idx];
        LaneState& l = s.lanes[pf.lane];
        if (rsp.data) {
          std::memcpy(l.line_data[pf.line_idx].data(), rsp.data->data(),
                      VX_CFG_MEM_BLOCK_SIZE);
        }
        l.line_filled[pf.line_idx] = true;
        ++l.lines_filled;
        if (s.pending_mem > 0) --s.pending_mem;

        // Phase 4 / 8: on the header line (line 0), parse the scene
        // header. The header layout is:
        //   uint32 primary_count;  // tris (TRI_LIST) or instances (TLAS)
        //   uint32 scene_kind;     // 0 = TRI_LIST, 1 = TLAS
        //   uint32 reserved[2];
        // For TRI_LIST: grow lines_needed to cover the tri list.
        // For TLAS (Phase 8): grow lines_needed to cover the worst-case
        // TLAS + inline-BLAS layout (one fetch budget; the smoke uses
        // 1 instance + 1 BLAS tri, well under the cap).
        if (pf.line_idx == 0 && !l.header_parsed) {
          uint32_t primary_count = 0;
          uint32_t scene_kind    = 0;
          const uint8_t* hdr     = l.line_data[0].data() + l.line_byte_off;
          std::memcpy(&primary_count, hdr + 0, sizeof(uint32_t));
          std::memcpy(&scene_kind,    hdr + 4, sizeof(uint32_t));
          l.scene_kind    = scene_kind;
          l.header_parsed = true;
          uint32_t needed = 1;
          if (scene_kind == kRtuSceneKindTlas) {
            if (primary_count > kRtuMaxInstancesPerTlas) {
              primary_count = kRtuMaxInstancesPerTlas;
            }
            l.instance_count = primary_count;
            needed = lines_for_bytes(l.line_byte_off, tlas_bytes(primary_count));
          } else if (scene_kind == kRtuSceneKindBvh4) {
            // Phase 4: VxBvhSceneHeader layout (see rtu_bvh.h):
            //   uint32 root_node_offset  (== primary_count slot here)
            //   uint32 scene_kind
            //   uint32 node_count + leaf_count (diagnostic, ignored)
            // Pre-fetch the entire BVH up to the per-lane line budget;
            // the walker reads from line_data synchronously via
            // read_scene_bytes. Chunk-3+ work may convert this to
            // demand-fetch as scenes grow past the line budget.
            l.bvh_root_offset = primary_count;
            l.triangle_count  = 0;
            l.instance_count  = 0;
            needed = kRtuMaxLinesPerLane;
          } else {
            if (primary_count > kRtuMaxTrisPerScene) primary_count = kRtuMaxTrisPerScene;
            l.triangle_count = primary_count;
            needed = lines_for_scene(l.line_byte_off, primary_count);
          }
          if (needed > kRtuMaxLinesPerLane) needed = kRtuMaxLinesPerLane;
          if (needed > l.lines_needed) {
            l.lines_needed = needed;
            // Drop slot back to ISSUE so the body lines get scheduled.
            s.state = State::ISSUE;
          }
        }

        // Transition to compute only when every active lane has all its
        // lines filled. Cross-lane lines_needed can differ if scenes are
        // per-lane (Phase 3-A2 SBT smoke).
        if (s.pending_mem == 0 && s.state == State::AWAIT) {
          bool all_done = true;
          for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
            const auto& ll = s.lanes[t];
            if (ll.active && ll.lines_filled < ll.lines_needed) {
              all_done = false; break;
            }
          }
          if (all_done) s.state = State::COMPUTE;
        }
        ch.pop();
      }
    }
  }

  // Phase 4 helper: read `len` bytes from the lane's logical scene
  // buffer at offset `off` (from line 0 + byte_off), crossing line
  // boundaries as needed. The caller passes a buffer big enough for the
  // largest single read (triangle = 40 B).
  static void read_scene_bytes(const LaneState& l, uint32_t off,
                               uint32_t len, uint8_t* out) {
    uint32_t base = l.line_byte_off + off;
    for (uint32_t i = 0; i < len; ++i) {
      uint32_t pos = base + i;
      uint32_t li  = pos / VX_CFG_MEM_BLOCK_SIZE;
      uint32_t bo  = pos % VX_CFG_MEM_BLOCK_SIZE;
      out[i] = (li < kRtuMaxLinesPerLane) ? l.line_data[li][bo] : uint8_t(0);
    }
  }

  // Per-lane traversal accumulator shared across BVH sub-tree walks.
  // The recursive walker writes hit + candidate state in place so a
  // BLAS-side hit can update the same best_t that culls subsequent
  // TLAS-side AABB tests.
  struct BvhWalkCtx {
    float tmin, tmax;
    uint32_t ray_flags;      // §8.8 ray-flag fast-out
    bool     terminated;     // set when TERMINATE_ON_FIRST_HIT fires
    float best_t, best_u, best_v;
    uint32_t best_prim;
    uint32_t best_instance;
    bool any_hit;
    bool yield_pending;
    float yield_t, yield_u, yield_v;
    uint32_t yield_prim;
    uint32_t yield_sbt;
    uint32_t yield_cb_type;
    uint32_t yield_instance;
  };

  // Phase 4 — depth-first walker for one BVH4 sub-tree under the
  // supplied (object-space) ray. Recurses on LeafInst so each
  // instance's BLAS gets walked with its transformed ray. ctx
  // accumulates hits / yields across the whole call tree.
  //
  // Stack depth caps at kBvhStackCap; deeper sub-trees silently
  // truncate (trail-based RESTART is a later refinement per proposal
  // §8.5.1). instance_id is the TLAS-assigned ID the caller wants
  // recorded for any hit found in this sub-tree.
  void walk_bvh4_subtree(LaneState& l,
                         const float ro[3], const float rd[3],
                         uint32_t root_off, uint32_t instance_id,
                         BvhWalkCtx& ctx) {
    auto visit_leaf_tri = [&](uint32_t leaf_off, uint32_t count) {
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

        // §8.8 effective-opacity override. Vulkan ray flags
        // VX_RT_FLAG_OPAQUE / VX_RT_FLAG_NO_OPAQUE force all hits
        // along the ray to one opacity class regardless of per-tri
        // flags (OPAQUE wins if both are set).
        bool tri_opaque = (tri_flags & kPhase2TriFlagOpaque) != 0;
        if (ctx.ray_flags & VX_RT_FLAG_OPAQUE)        tri_opaque = true;
        else if (ctx.ray_flags & VX_RT_FLAG_NO_OPAQUE) tri_opaque = false;

        // §8.8 CULL_OPAQUE / CULL_NO_OPAQUE. Skip triangles whose
        // effective opacity matches the cull class.
        if (tri_opaque    && (ctx.ray_flags & VX_RT_FLAG_CULL_OPAQUE))    continue;
        if (!tri_opaque   && (ctx.ray_flags & VX_RT_FLAG_CULL_NO_OPAQUE)) continue;

        float t_hit = 0.f, u = 0.f, v = 0.f;
        bool back_facing = false;
        ++perf_stats_.bvh_tri_tests;
        if (!ray_triangle(ro, rd, &tri[0], &tri[3], &tri[6],
                          ctx.tmin, ctx.tmax,
                          t_hit, u, v, back_facing)) {
          continue;
        }

        // §8.8 CULL_BACK_FACING / CULL_FRONT_FACING. Test against
        // the det-sign from ray_triangle.
        if (back_facing  && (ctx.ray_flags & VX_RT_FLAG_CULL_BACK_FACING))  continue;
        if (!back_facing && (ctx.ray_flags & VX_RT_FLAG_CULL_FRONT_FACING)) continue;

        if (tri_opaque) {
          if (t_hit < ctx.best_t) {
            ctx.best_t = t_hit; ctx.best_u = u; ctx.best_v = v;
            ctx.best_prim = i;
            ctx.best_instance = instance_id;
            ctx.any_hit = true;
            if (ctx.yield_pending && ctx.yield_t >= ctx.best_t) {
              ctx.yield_pending = false;
              ctx.yield_t = ctx.tmax;
            }
            // §8.8 TERMINATE_ON_FIRST_HIT: commit and stop. Shadow-
            // ray fast path; saves the rest of the BVH walk.
            if (ctx.ray_flags & VX_RT_FLAG_TERMINATE_ON_FIRST_HIT) {
              ctx.terminated = true;
              return;
            }
          }
        } else {
          if (t_hit < ctx.best_t && t_hit < ctx.yield_t) {
            ctx.yield_pending = true;
            ctx.yield_t = t_hit; ctx.yield_u = u; ctx.yield_v = v;
            ctx.yield_prim = i;
            ctx.yield_instance = instance_id;
            ctx.yield_sbt = (tri_flags >> kPhase2TriSbtIdxShift) & kPhase2TriSbtIdxMask;
            ctx.yield_cb_type = (tri_flags & kPhase2TriFlagProc)
                                  ? VX_RT_CB_TYPE_PROC
                                  : VX_RT_CB_TYPE_ANYHIT;
          }
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
        // Transform world ray into the instance's object space.
        float obj_ro[3], obj_rd[3];
        affine_inverse_transform_ray(inst->xform, ro, rd, obj_ro, obj_rd);
        // Use the instance's HW-assigned id (preferred over the lane
        // index) so the caller can reconstruct gl_InstanceID.
        ++perf_stats_.bvh_instance_descents;
        walk_bvh4_subtree(l, obj_ro, obj_rd,
                          inst->blas_root_byte_offset,
                          inst->instance_id,
                          ctx);
      }
    };

    constexpr uint32_t kBvhStackCap = 16;
    uint32_t stack[kBvhStackCap];
    uint32_t stack_top = 0;
    uint32_t current = root_off;
    bool have_current = true;

    while (have_current) {
      if (ctx.terminated) break;  // §8.8 TERMINATE_ON_FIRST_HIT
      uint8_t kind_buf[4];
      read_scene_bytes(l, current, sizeof(kind_buf), kind_buf);
      uint32_t kind_word = 0;
      std::memcpy(&kind_word, kind_buf, sizeof(uint32_t));
      uint32_t kind  = kind_word & kVxBvhKindMask;
      uint32_t count = (kind_word >> kVxBvhCountShift) & kVxBvhCountMask;

      if (kind == kVxBvhKindLeafTri) {
        ++perf_stats_.bvh_leaves_fetched;
        // §8.8 SKIP_TRIANGLES skips the entire tri-leaf class.
        if (!(ctx.ray_flags & VX_RT_FLAG_SKIP_TRIANGLES)) {
          visit_leaf_tri(current, count);
        }
      } else if (kind == kVxBvhKindLeafInst) {
        ++perf_stats_.bvh_leaves_fetched;
        visit_leaf_inst(current, count);
      } else if (kind == kVxBvhKindLeafProc) {
        ++perf_stats_.bvh_leaves_fetched;
        // §8.8 SKIP_AABBS is a no-op-but-explicit gate for the
        // procedural-leaf path; the walker still records no hit
        // from LeafProc (Phase 6 work — wires the IS callback).
        // The check is here so the gating logic is symmetric with
        // SKIP_TRIANGLES once IS lands.
        (void)0;
      } else if (kind == kVxBvhKindInternal) {
        ++perf_stats_.bvh_nodes_fetched;
        uint8_t node_buf[sizeof(VxBvhInternalNode)];
        read_scene_bytes(l, current, sizeof(node_buf), node_buf);
        const VxBvhInternalNode* node =
            reinterpret_cast<const VxBvhInternalNode*>(node_buf);
        uint32_t nch = count;
        if (nch > kVxBvhWidth) nch = kVxBvhWidth;

        struct ChildHit { uint32_t offset; float t_near; };
        ChildHit hits[kVxBvhWidth];
        uint32_t hit_count = 0;
        for (uint32_t i = 0; i < nch; ++i) {
          uint32_t off_word = node->child_offsets[i];
          uint32_t child_off = off_word & kVxBvhChildOffsetMask;
          if (off_word == kVxBvhChildEmpty) continue;
          float mn[3], mx[3];
          reconstruct_child_aabb(node->origin, node->exp,
                                  node->qaabb_min[i], node->qaabb_max[i],
                                  mn, mx);
          float t_near = 0.f;
          ++perf_stats_.bvh_box_tests;
          if (!ray_aabb_intersect(ro, rd, mn, mx,
                                  ctx.tmin, ctx.best_t, t_near)) {
            continue;
          }
          hits[hit_count++] = { child_off, t_near };
        }
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
            if (stack_top < kBvhStackCap) {
              stack[stack_top++] = hits[i].offset;
            }
          }
          current = hits[0].offset;
          have_current = true;
          continue;
        }
      }
      // kVxBvhKindLeafProc — chunk 4-late. Procedural leaves require
      // the IS shader path; the walker currently records no hit from
      // them, matching the flat-list path's behaviour for the
      // pre-Phase-6 era.

      if (stack_top == 0) {
        have_current = false;
      } else {
        current = stack[--stack_top];
      }
    }
  }

  // Phase 4: walk a BVH4 scene for one lane. Top-level entry — sets
  // up the BvhWalkCtx, invokes walk_bvh4_subtree at the scene root,
  // then translates accumulated state to LaneState / ahs_queue_.
  //
  // Returns true iff this lane queued a CB_YIELD entry.
  bool compute_intersections_bvh4_lane(Slot& s, LaneState& l, uint32_t t,
                                       uint32_t slot_idx) {
    const float ro[3] = { s.req.origin_x[t], s.req.origin_y[t], s.req.origin_z[t] };
    const float rd[3] = { s.req.dir_x[t],    s.req.dir_y[t],    s.req.dir_z[t]   };

    BvhWalkCtx ctx;
    ctx.tmin = s.req.tmin[t];
    ctx.tmax = s.req.tmax[t];
    ctx.ray_flags = s.req.flags[t];
    ctx.terminated = false;
    ctx.best_t = ctx.tmax;
    ctx.best_u = 0.f; ctx.best_v = 0.f;
    ctx.best_prim = 0; ctx.best_instance = 0;
    ctx.any_hit = false;
    ctx.yield_pending = false;
    ctx.yield_t = ctx.tmax; ctx.yield_u = 0.f; ctx.yield_v = 0.f;
    ctx.yield_prim = 0; ctx.yield_sbt = 0;
    ctx.yield_cb_type = VX_RT_CB_TYPE_ANYHIT;
    ctx.yield_instance = 0;

    walk_bvh4_subtree(l, ro, rd, l.bvh_root_offset, 0, ctx);

    bool any_hit = ctx.any_hit;
    float best_t = ctx.best_t;
    float best_u = ctx.best_u;
    float best_v = ctx.best_v;
    uint32_t best_prim = ctx.best_prim;
    bool yield_pending = ctx.yield_pending;
    float yield_t = ctx.yield_t;
    float yield_u = ctx.yield_u;
    float yield_v = ctx.yield_v;
    uint32_t yield_prim = ctx.yield_prim;
    uint32_t yield_sbt = ctx.yield_sbt;
    uint32_t yield_cb_type = ctx.yield_cb_type;

    l.hit             = any_hit;
    l.hit_t           = best_t;
    l.hit_u           = best_u;
    l.hit_v           = best_v;
    l.hit_prim        = best_prim;
    // Phase 4 chunk 4: best_instance is the HW-assigned ID from the
    // TLAS instance record; for hits found while walking a BLAS via
    // LeafInst recursion, the walker stamps the originating instance.
    // Single-level (TRI-only) BVHs leave it at 0 — matches Phase 8.
    l.hit_instance_id = any_hit ? ctx.best_instance
                                : (yield_pending ? ctx.yield_instance : 0u);

    if (yield_pending) {
      l.cb_pending = true;
      l.cb_type    = yield_cb_type;
      l.sbt_idx    = yield_sbt;
      l.cand_t     = yield_t;
      l.cand_u     = yield_u;
      l.cand_v     = yield_v;
      l.cand_prim  = yield_prim;
      QueueEntry e;
      e.slot_idx  = slot_idx;
      e.warp_id   = s.req.warp_id;
      e.lane      = uint8_t(t);
      e.sbt_idx   = yield_sbt;
      e.cb_type   = yield_cb_type;
      e.cand_t    = yield_t;
      e.cand_u    = yield_u;
      e.cand_v    = yield_v;
      e.cand_prim = yield_prim;
      ahs_queue_.push_back(e);
      return true;
    }
    // §8.8 SKIP_CLOSEST_HIT suppresses the CHS dispatch even when
    // ENABLE_CHS is set. Common shadow-ray idiom: want hit/miss but
    // not the full shading callback.
    if (any_hit && (s.req.flags[t] & VX_RT_FLAG_ENABLE_CHS)
                && !(s.req.flags[t] & VX_RT_FLAG_SKIP_CLOSEST_HIT)) {
      l.cb_pending = true;
      l.cb_type    = VX_RT_CB_TYPE_CHS;
      l.sbt_idx    = 0;
      l.cand_t     = best_t;
      l.cand_u     = best_u;
      l.cand_v     = best_v;
      l.cand_prim  = best_prim;
      QueueEntry e;
      e.slot_idx  = slot_idx;
      e.warp_id   = s.req.warp_id;
      e.lane      = uint8_t(t);
      e.sbt_idx   = 0;
      e.cb_type   = VX_RT_CB_TYPE_CHS;
      e.cand_t    = best_t;
      e.cand_u    = best_u;
      e.cand_v    = best_v;
      e.cand_prim = best_prim;
      ahs_queue_.push_back(e);
      return true;
    }
    if (!any_hit && (s.req.flags[t] & VX_RT_FLAG_ENABLE_MISS)) {
      l.cb_pending = true;
      l.cb_type    = VX_RT_CB_TYPE_MISS;
      l.sbt_idx    = 0;
      l.cand_t     = 0.f;
      l.cand_u     = 0.f;
      l.cand_v     = 0.f;
      l.cand_prim  = 0;
      QueueEntry e;
      e.slot_idx  = slot_idx;
      e.warp_id   = s.req.warp_id;
      e.lane      = uint8_t(t);
      e.sbt_idx   = 0;
      e.cb_type   = VX_RT_CB_TYPE_MISS;
      e.cand_t    = 0.f;
      e.cand_u    = 0.f;
      e.cand_v    = 0.f;
      e.cand_prim = 0;
      ahs_queue_.push_back(e);
      return true;
    }
    return false;
  }

  void compute_intersections() {
    // §8.9 coherency gather: visit slots whose octant signature
    // matches the last-processed signature first (warmer cache),
    // then non-matching. Per-tick perf counters: every COMPUTE-ready
    // slot picked increments either coherency_hits or _misses.
    for (uint32_t pass = 0; pass < 2; ++pass) {
      for (auto& s : slots_) {
        if (!s.in_use || s.state != State::COMPUTE) continue;
        bool sig_matches = (s.coh_signature == last_compute_signature_);
        if (pass == 0 && !sig_matches) continue;  // matching pass
        if (pass == 1 && sig_matches)  continue;  // non-matching pass
        if (sig_matches) ++perf_stats_.coherency_hits;
        else             ++perf_stats_.coherency_misses;
        last_compute_signature_ = s.coh_signature;

        bool any_cb_pending = false;
        uint32_t slot_idx = uint32_t(&s - &slots_[0]);
        for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        LaneState& l = s.lanes[t];
        if (!l.active) continue;
        // Phase 4: BVH4 walker dispatched here. Sets per-lane state
        // and queues any CB_YIELD itself; we skip the flat-list path
        // below for this lane.
        if (l.scene_kind == kRtuSceneKindBvh4) {
          if (compute_intersections_bvh4_lane(s, l, t, slot_idx)) {
            any_cb_pending = true;
          }
          continue;
        }
        // Phase 8: TLAS scenes walk one or more instances; each
        // instance points at a BLAS (a triangle list) and (optionally)
        // applies an object→world affine transform. For Phase 8
        // minimum the transform is treated as identity, so world ray
        // ≡ object ray and the inner walker is unchanged.
        uint32_t num_instances = 1;
        if (l.scene_kind == kRtuSceneKindTlas) {
          if (l.instance_count == 0) {
            l.hit = false;
            continue;
          }
          num_instances = l.instance_count;
        } else {
          if (l.triangle_count == 0) {
            l.hit = false;
            continue;
          }
        }
        float best_t = s.req.tmax[t];
        float best_u = 0.f;
        float best_v = 0.f;
        uint32_t best_prim = 0;
        uint32_t best_instance = 0;
        bool any_hit = false;
        bool yield_pending = false;
        // Init yield_t to tmax so the first non-opaque candidate
        // always wins the "closer than current pending candidate"
        // check (Phase 11 single-closest-yield).
        float yield_t = s.req.tmax[t];
        float yield_u = 0.f, yield_v = 0.f;
        uint32_t yield_prim = 0;
        uint32_t yield_sbt  = 0;
        uint32_t yield_cb_type = VX_RT_CB_TYPE_ANYHIT;
        uint32_t yield_instance = 0;
        float ro[3] = { s.req.origin_x[t], s.req.origin_y[t], s.req.origin_z[t] };
        float rd[3] = { s.req.dir_x[t],    s.req.dir_y[t],    s.req.dir_z[t]   };
        uint8_t tri_buf[kPhase2TriStride];
        for (uint32_t inst_idx = 0; inst_idx < num_instances && !yield_pending;
             ++inst_idx) {
          // Resolve BLAS base + tri count + object-space ray for this
          // instance. For TRI_LIST we just walk the only "instance"
          // with the world ray and BLAS at offset 16; for TLAS we
          // pull the instance record, invert its 3x4 affine to map
          // the world ray to object space, then walk the BLAS.
          uint32_t blas_tri_off   = kRtuSceneHeaderBytes;
          uint32_t blas_tri_count = l.triangle_count;
          float ray_o[3] = { ro[0], ro[1], ro[2] };
          float ray_d[3] = { rd[0], rd[1], rd[2] };
          if (l.scene_kind == kRtuSceneKindTlas) {
            uint32_t inst_off = kRtuSceneHeaderBytes
                              + inst_idx * kRtuInstanceStride;
            // Read the 3x4 affine transform (48 B) followed by the
            // blas_byte_offset + custom_id tail (16 B) — together the
            // full 64 B instance record.
            uint8_t inst_buf[kRtuInstanceStride];
            read_scene_bytes(l, inst_off, sizeof(inst_buf), inst_buf);
            const float* xform = reinterpret_cast<const float*>(inst_buf);
            uint32_t blas_byte_off = 0;
            std::memcpy(&blas_byte_off,
                        inst_buf + kRtuInstanceBlasOffOff,
                        sizeof(uint32_t));
            // World→object ray transform (Phase 9). For pure rotation
            // + translation the t parameter is preserved, so the
            // BLAS-reported hit_t is also the world hit_t.
            affine_inverse_transform_ray(xform, ro, rd, ray_o, ray_d);
            // Parse the BLAS header to learn its triangle count.
            uint8_t blas_hdr[4];
            read_scene_bytes(l, blas_byte_off, sizeof(blas_hdr), blas_hdr);
            uint32_t bcount = 0;
            std::memcpy(&bcount, blas_hdr, sizeof(uint32_t));
            if (bcount > kRtuMaxTrisPerScene) bcount = kRtuMaxTrisPerScene;
            blas_tri_count = bcount;
            blas_tri_off   = blas_byte_off + kRtuSceneHeaderBytes;
          }
          uint32_t n_tris = std::min(blas_tri_count, kRtuMaxTrisPerScene);
          // Phase 11: walk the *full* triangle list. Track the best
          // committed opaque hit AND the closest non-opaque candidate
          // separately. If a non-opaque candidate ends up closer than
          // the best opaque, yield it; otherwise the opaque commits
          // and no AHS fires (alpha-test fast path). This fixes the
          // prior "break on first non-opaque" path, where an AHS
          // IGNORE could drop a closer opaque hit found later in the
          // scene (was MISS instead of HIT).
          //
          // Per-tri flags layout (Phase 2/3-A2/6): bit 0 = OPAQUE,
          // bit 1 = PROCEDURAL, bits 8..15 = SBT_IDX.
          // §8.8 ray-flag handling: SKIP_TRIANGLES bails the
          // whole leaf-tri scan. (Flat-list scenes only have tri
          // leaves, so SKIP_AABBS is a no-op here.)
          const uint32_t ray_flags = s.req.flags[t];
          if (ray_flags & VX_RT_FLAG_SKIP_TRIANGLES) {
            // Skip all triangles; remaining hit/yield logic falls
            // through with no_hit and no pending candidate.
          } else
          for (uint32_t i = 0; i < n_tris; ++i) {
            uint32_t tri_off = blas_tri_off + i * kPhase2TriStride;
            read_scene_bytes(l, tri_off, kPhase2TriStride, tri_buf);
            const float* tri = reinterpret_cast<const float*>(tri_buf);
            uint32_t tri_flags = 0;
            std::memcpy(&tri_flags, tri_buf + kPhase2TriFlagsOff,
                        sizeof(uint32_t));

            // §8.8 effective opacity (Vulkan OPAQUE / NO_OPAQUE
            // ray flags override per-tri OPAQUE bit; OPAQUE wins).
            bool tri_opaque = (tri_flags & kPhase2TriFlagOpaque) != 0;
            if (ray_flags & VX_RT_FLAG_OPAQUE)         tri_opaque = true;
            else if (ray_flags & VX_RT_FLAG_NO_OPAQUE) tri_opaque = false;

            // §8.8 cull by opacity class.
            if (tri_opaque  && (ray_flags & VX_RT_FLAG_CULL_OPAQUE))    continue;
            if (!tri_opaque && (ray_flags & VX_RT_FLAG_CULL_NO_OPAQUE)) continue;

            float t_hit = 0.f, u = 0.f, v = 0.f;
            bool back_facing = false;
            ++perf_stats_.bvh_tri_tests;
            // Test against ray.tmax (not best_t) so an opaque hit
            // committed earlier in this walk doesn't pre-cull a
            // non-opaque candidate that might survive an ACCEPT.
            if (!ray_triangle(ray_o, ray_d, &tri[0], &tri[3], &tri[6],
                              s.req.tmin[t], s.req.tmax[t],
                              t_hit, u, v, back_facing)) {
              continue;
            }

            // §8.8 face culling.
            if (back_facing  && (ray_flags & VX_RT_FLAG_CULL_BACK_FACING))  continue;
            if (!back_facing && (ray_flags & VX_RT_FLAG_CULL_FRONT_FACING)) continue;

            if (tri_opaque) {
              if (t_hit < best_t) {
                best_t = t_hit; best_u = u; best_v = v; best_prim = i;
                best_instance = inst_idx;
                any_hit = true;
                // Any pending non-opaque candidate that's farther than
                // this new opaque commit is now culled.
                if (yield_pending && yield_t >= best_t) {
                  yield_pending = false;
                  yield_t = s.req.tmax[t];
                }
                // §8.8 TERMINATE_ON_FIRST_HIT — shadow ray fast path.
                // Bails out of the per-tri scan AND, via the outer
                // tmax-trimming logic, future instances/leaves can't
                // contribute either (we set tmax in s.req for them).
                if (ray_flags & VX_RT_FLAG_TERMINATE_ON_FIRST_HIT) {
                  // Cap remaining work by tightening tmax: any later
                  // tri test in this leaf scan would have t > best_t,
                  // so they'll all fail their tmax check.
                  s.req.tmax[t] = best_t;
                  break;
                }
              }
            } else {
              // Non-opaque: only consider as candidate if (a) closer
              // than current best opaque (else opaque wins anyway)
              // and (b) closer than the current pending candidate.
              // Phase 11 yields the single CLOSEST non-opaque candidate
              // — multi-yield over every alpha-tested hit is a future
              // enhancement.
              if (t_hit < best_t && t_hit < yield_t) {
                yield_pending = true;
                yield_t = t_hit; yield_u = u; yield_v = v; yield_prim = i;
                yield_sbt = (tri_flags >> kPhase2TriSbtIdxShift) & kPhase2TriSbtIdxMask;
                yield_cb_type = (tri_flags & kPhase2TriFlagProc)
                                  ? VX_RT_CB_TYPE_PROC
                                  : VX_RT_CB_TYPE_ANYHIT;
                yield_instance = inst_idx;
              }
            }
          }
        }
        l.hit       = any_hit;
        l.hit_t     = best_t;
        l.hit_u     = best_u;
        l.hit_v     = best_v;
        l.hit_prim  = best_prim;
        // Phase 8: instance id of the lane's best committed hit (TLAS).
        // For TRI_LIST scenes this stays 0 — no instance concept.
        l.hit_instance_id = any_hit ? best_instance
                                    : (yield_pending ? yield_instance : 0u);
        if (yield_pending) {
          l.cb_pending = true;
          l.cb_type    = yield_cb_type;
          l.sbt_idx    = yield_sbt;
          l.cand_t     = yield_t;
          l.cand_u     = yield_u;
          l.cand_v     = yield_v;
          l.cand_prim  = yield_prim;
          // Phase 3-A2: push one entry per yielded lane into the per-core
          // AHS queue. The reformation pass picks up these entries on a
          // later tick and groups by (warp_id, sbt_idx) to emit coherent
          // CB_YIELD batches.
          QueueEntry e;
          e.slot_idx  = slot_idx;
          e.warp_id   = s.req.warp_id;
          e.lane      = uint8_t(t);
          e.sbt_idx   = yield_sbt;
          e.cb_type   = yield_cb_type;
          e.cand_t    = yield_t;
          e.cand_u    = yield_u;
          e.cand_v    = yield_v;
          e.cand_prim = yield_prim;
          ahs_queue_.push_back(e);
          any_cb_pending = true;
        } else if (any_hit && (s.req.flags[t] & VX_RT_FLAG_ENABLE_CHS)
                            && !(s.req.flags[t] & VX_RT_FLAG_SKIP_CLOSEST_HIT)) {
          // Phase 5: opaque hit committed and the ray opted into CHS.
          // Queue a CHS yield carrying the committed hit attrs through
          // the same reformation/CB_YIELD/CB_ACTION pipe as AHS.
          // sbt_idx falls back to 0 here since opaque hits don't carry
          // a per-tri SBT key in the current scene format. §8.8
          // SKIP_CLOSEST_HIT suppresses the dispatch even when
          // ENABLE_CHS is set (shadow-ray idiom).
          l.cb_pending = true;
          l.cb_type    = VX_RT_CB_TYPE_CHS;
          l.sbt_idx    = 0;
          l.cand_t     = best_t;
          l.cand_u     = best_u;
          l.cand_v     = best_v;
          l.cand_prim  = best_prim;
          QueueEntry e;
          e.slot_idx  = slot_idx;
          e.warp_id   = s.req.warp_id;
          e.lane      = uint8_t(t);
          e.sbt_idx   = 0;
          e.cb_type   = VX_RT_CB_TYPE_CHS;
          e.cand_t    = best_t;
          e.cand_u    = best_u;
          e.cand_v    = best_v;
          e.cand_prim = best_prim;
          ahs_queue_.push_back(e);
          any_cb_pending = true;
        } else if (!any_hit && (s.req.flags[t] & VX_RT_FLAG_ENABLE_MISS)) {
          // Phase 5: no hit found and the ray opted into MISS shader.
          // Hit-attr fields are zero (no candidate); the dispatcher
          // typically synthesises a sky/env contribution from the ray
          // direction, which is still available via the RTU regfile
          // VX_RT_RAY_DIRECTION slots that the kernel staged before
          // vx_rt_trace.
          l.cb_pending = true;
          l.cb_type    = VX_RT_CB_TYPE_MISS;
          l.sbt_idx    = 0;
          l.cand_t     = 0.f;
          l.cand_u     = 0.f;
          l.cand_v     = 0.f;
          l.cand_prim  = 0;
          QueueEntry e;
          e.slot_idx  = slot_idx;
          e.warp_id   = s.req.warp_id;
          e.lane      = uint8_t(t);
          e.sbt_idx   = 0;
          e.cb_type   = VX_RT_CB_TYPE_MISS;
          e.cand_t    = 0.f;
          e.cand_u    = 0.f;
          e.cand_v    = 0.f;
          e.cand_prim = 0;
          ahs_queue_.push_back(e);
          any_cb_pending = true;
        }
      }
      s.state = any_cb_pending ? State::IN_QUEUE : State::RESP;
    }
    }  // end pass loop (§8.9 two-pass coherency gather)
  }

  // Phase 3-A2: drain ahs_queue_ into batched CB_YIELD packets. Group the
  // front entries by (warp_id, sbt_idx) so each emitted CB_YIELD raises a
  // single async trap whose tmask covers a SBT-coherent lane group — the
  // kernel's switch(sbt_idx) then dispatches one branch for the whole
  // virtual warp instead of N divergent branches. Same-warp only: we do
  // NOT touch the warp scheduler; reformation is invisible to it.
  void reformation_dispatch() {
    if (simobject_->rtu_rsp_out.empty()) return;
    auto& port = simobject_->rtu_rsp_out.at(0);
    while (!ahs_queue_.empty()) {
      if (port.full()) break;
      // Same-warp serialization gate: a CB_YIELD raises an M-mode trap
      // on the warp (mepc/mtvec/mscratch_tmask). Firing a second
      // CB_YIELD on the same warp before its CB_RET drains would
      // clobber the trap CSRs and lose the return path. So pick the
      // first queue entry whose warp is not already mid-callback.
      auto anchor_it = ahs_queue_.end();
      for (auto it = ahs_queue_.begin(); it != ahs_queue_.end(); ++it) {
        if (!warp_cb_inflight_[it->warp_id]) { anchor_it = it; break; }
      }
      if (anchor_it == ahs_queue_.end()) break;  // every warp busy
      uint32_t anchor_warp = anchor_it->warp_id;
      uint32_t anchor_sbt  = anchor_it->sbt_idx;
      RtuRsp rsp;
      rsp.kind     = RtuRspKind::CB_YIELD;
      rsp.warp_id  = anchor_warp;
      rsp.block_id = 0;  // CB_YIELD doesn't carry a parked trace; the
      rsp.trace    = nullptr;  // SfuUnit drain path only consumes warp_id.
      uint32_t cb_mask = 0;
      auto it = ahs_queue_.begin();
      while (it != ahs_queue_.end()) {
        if (it->warp_id != anchor_warp || it->sbt_idx != anchor_sbt) {
          ++it; continue;
        }
        uint8_t t = it->lane;
        if (cb_mask & (1u << t)) {
          // Lane already grouped for this CB_YIELD — defer the second
          // candidate (multi-yield per lane) to a future reformation
          // pass so the kernel doesn't see two conflicting writes into
          // VX_RT_CB_HANDLE in one trap.
          ++it; continue;
        }
        cb_mask |= (1u << t);
        rsp.cb_type[t]            = it->cb_type;
        rsp.cb_handle[t]          = it->slot_idx;
        rsp.cb_sbt_idx[t]         = it->sbt_idx;
        rsp.hit_t[t]              = it->cand_t;
        rsp.hit_bary_u[t]         = it->cand_u;
        rsp.hit_bary_v[t]         = it->cand_v;
        rsp.hit_primitive_id[t]   = it->cand_prim;
        it = ahs_queue_.erase(it);
      }
      rsp.cb_active_mask = cb_mask;
      port.send(rsp);
      warp_cb_inflight_[anchor_warp] = true;
      // §8.9 per-shader-type callback counters. cb_type is uniform
      // within a batched yield (grouped by sbt_idx) — sample any
      // active lane.
      ++perf_stats_.reformation_yields;
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        if (!(cb_mask & (1u << t))) continue;
        switch (rsp.cb_type[t]) {
          case VX_RT_CB_TYPE_ANYHIT: ++perf_stats_.ahs_callbacks;  break;
          case VX_RT_CB_TYPE_PROC:   ++perf_stats_.is_callbacks;   break;
          case VX_RT_CB_TYPE_CHS:    ++perf_stats_.chs_callbacks;  break;
          case VX_RT_CB_TYPE_MISS:   ++perf_stats_.miss_callbacks; break;
          default:                                                  break;
        }
        break;  // one sample per yield is the per-yield-type counter
      }
      DT(3, "rtu-core reform cb_yield: warp=" << anchor_warp
            << ", sbt=" << anchor_sbt
            << ", cb_mask=0x" << std::hex << cb_mask << std::dec);
    }
  }

  void emit_completions() {
    if (simobject_->rtu_rsp_out.empty()) return;
    auto& port = simobject_->rtu_rsp_out.at(0);
    for (auto& s : slots_) {
      if (!s.in_use) continue;
      if (s.state != State::RESP) continue;
      if (port.full()) break;
      RtuRsp rsp(s.req);
      rsp.kind = RtuRspKind::TERMINAL;
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        const LaneState& l = s.lanes[t];
        if (!l.active) {
          rsp.status[t] = VX_RT_STS_DONE_MISS;
          continue;
        }
        if (l.hit) {
          rsp.status[t]            = VX_RT_STS_DONE_HIT;
          rsp.hit_t[t]             = l.hit_t;
          rsp.hit_bary_u[t]        = l.hit_u;
          rsp.hit_bary_v[t]        = l.hit_v;
          rsp.hit_primitive_id[t]  = l.hit_prim;
          rsp.hit_instance_id[t]   = l.hit_instance_id;
          ++perf_stats_.rays_hit;
        } else {
          rsp.status[t] = VX_RT_STS_DONE_MISS;
          ++perf_stats_.rays_miss;
        }
      }
      port.send(rsp);
      DT(3, "rtu-core complete: tag=" << s.req.tag);
      s.in_use = false;
      s.state = State::ISSUE;
    }
  }

  const RtuCore::PerfStats& perf_stats() const { return perf_stats_; }

private:
  struct PendingFill { uint32_t slot_idx; uint8_t lane; uint8_t line_idx; };

  RtuCore* simobject_;
  std::vector<Slot> slots_;
  std::unordered_map<uint32_t, PendingFill> pending_mem_;
  // Phase 3-A2 same-warp reformation queue. Yielded lanes (one entry per
  // yielded (slot, lane)) are pushed by compute_intersections and drained
  // by reformation_dispatch into batched CB_YIELD rsps grouped by
  // (warp_id, sbt_idx).
  std::deque<QueueEntry> ahs_queue_;
  // Phase 3-A2: per-warp "callback in flight" gate. Set when
  // reformation_dispatch emits a CB_YIELD on a warp, cleared when its
  // matching CB_ACTION drains. Serializes per-warp traps so multiple
  // SBT groups for the same warp are dispatched sequentially.
  std::array<bool, VX_CFG_NUM_WARPS> warp_cb_inflight_{};
  uint32_t next_tag_ = 0;
  // §8.9 coherency gather: octant signature of the most recently
  // processed slot. Initialized to 0 (all-positive-axis ray).
  uint8_t  last_compute_signature_ = 0;
  RtuCore::PerfStats perf_stats_;
};

// ════════════════════════════════════════════════════════════════════

// Phase 1: a single mem port mirrors TCACHE/OCACHE/RCACHE (kTcacheMemPorts=1)
// and matches VX_CFG_L2_NUM_REQS in the smoke config. Phase 2+ can fan out
// to additional ports once an RTU-side cache is in place.
RtuCore::RtuCore(const SimContext& ctx, const char* name, Cluster* /*cluster*/)
  : SimObject<RtuCore>(ctx, name)
  , rtu_req_in(VX_CFG_NUM_RTU_CORES, this)
  , rtu_rsp_out(VX_CFG_NUM_RTU_CORES, this)
  , dcache_req_out(1, this)
  , dcache_rsp_in(1, this)
  , impl_(new Impl(this))
{}

RtuCore::~RtuCore() {
  delete impl_;
}

void RtuCore::on_reset() { impl_->reset(); }
void RtuCore::on_tick()  { impl_->tick();  }

const RtuCore::PerfStats& RtuCore::perf_stats() const {
  return impl_->perf_stats();
}
