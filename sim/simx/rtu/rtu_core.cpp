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
#include "rtu_types.h"
#include "rtu_bvh.h"
#include "rtu_isect.h"   // §step-3: ray_triangle / ray_aabb_intersect /
                         //          affine_inverse_transform_ray
#include "rtu_classifier.h"  // §step-4: classify_tri_hit / finalise_lane
#include "rtu_walker.h"      // §step-5: FlatWalker / Bvh4Walker
#include "rtu_memory.h"      // §step-6: MemoryEngine
#include "cluster.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;
using namespace vortex::rtu;

// §step-5: per-slot/per-lane walker mechanics (reconstruct_child_aabb,
// read_scene_bytes, BvhWalkCtx, walk_bvh4_subtree, the flat-list
// scanner) moved to rtu_walker.{h,cpp}. The Impl below now just
// dispatches per-lane to FlatWalker or Bvh4Walker.

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

  // Walker / mem-engine init must come AFTER ahs_queue_ / perf_stats_
  // in the member declaration list below (they bind references to
  // those members) — init order here matches declaration order to
  // satisfy -Wreorder.
  explicit Impl(RtuCore* simobject)
    : simobject_(simobject)
    , slots_(VX_CFG_RTU_CONTEXT_POOL)
    , perf_stats_()
    , flat_walker_(perf_stats_, ahs_queue_)
    , bvh4_walker_(perf_stats_, ahs_queue_)
    , mem_engine_(slots_,
                   simobject->dcache_req_out,
                   simobject->dcache_rsp_in,
                   perf_stats_)
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
    mem_engine_.reset();
    ahs_queue_.clear();
    warp_cb_inflight_.fill(false);
    perf_stats_ = RtuCore::PerfStats();
    last_compute_signature_ = 0;
  }

  void tick() {
    mem_engine_.drain_mem_rsp();
    drain_requests();
    mem_engine_.issue_memory();
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

  // §step-5: slim orchestrator. The §8.9 two-pass octant-signature
  // coherency loop stays here (it picks WHICH slot to process next);
  // the actual per-lane traversal is delegated to FlatWalker /
  // Bvh4Walker based on scene_kind.
  void compute_intersections() {
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
          bool queued = (l.scene_kind == kRtuSceneKindBvh4)
                          ? bvh4_walker_.walk_lane(s, l, t, slot_idx)
                          : flat_walker_.walk_lane(s, l, t, slot_idx);
          if (queued) any_cb_pending = true;
        }
        s.state = any_cb_pending ? State::IN_QUEUE : State::RESP;
      }
    }
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
  RtuCore* simobject_;
  std::vector<Slot> slots_;
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
  // §8.9 coherency gather: octant signature of the most recently
  // processed slot. Initialized to 0 (all-positive-axis ray).
  uint8_t  last_compute_signature_ = 0;
  RtuCore::PerfStats perf_stats_;

  // §step-5/6: sub-modules. References to perf_stats_ / ahs_queue_ /
  // slots_ bind at construction, so these MUST stay below those
  // members (init order matches declaration order to satisfy
  // -Wreorder).
  FlatWalker   flat_walker_;
  Bvh4Walker   bvh4_walker_;
  MemoryEngine mem_engine_;
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
