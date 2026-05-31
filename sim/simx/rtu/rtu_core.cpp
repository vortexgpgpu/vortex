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

namespace {

// ────────────────────────────────────────────────────────────────────
// SlotPool — the cluster-side context-slot array. Owns the slot
// vector and the allocate / reset bookkeeping. Engines and walkers
// take a ref to the underlying vector (via slots()) so per-slot
// iteration stays a tight `for (auto& s : pool.slots())` loop on
// both sides.
//
// In SystemC this becomes a register-file SC_MODULE or stays a
// nested submodule inside SC_MODULE(RtuCore).
// ────────────────────────────────────────────────────────────────────
class SlotPool {
public:
  explicit SlotPool(uint32_t size) : slots_(size) {}

  void reset() {
    for (auto& s : slots_) reset_slot(s);
  }

  // First-fit allocate. Returns the slot index, or -1 if all in use.
  // The caller is responsible for filling s.req and per-lane flags.
  // §8.6: slot is left in RESERVED state — issue_memory / walkers /
  // emit_completions all gate on ISSUE/COMPUTE/RESP and skip
  // RESERVED, so the slot is invisible to the FSM until
  // drain_requests promotes it on TRACE_NEW arrival. Without this,
  // a fresh slot with all-inactive lanes would loop straight to
  // RESP and emit a spurious zero-hit TERMINAL.
  int32_t allocate() {
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (!slots_[i].in_use) {
        Slot& s = slots_[i];
        s.in_use = true;
        s.state  = SlotState::RESERVED;
        s.pending_mem = 0;
        // §8.7 cycle-drain reset.
        s.compute_cycles_remaining = 0;
        s.walk_done = false;
        s.next_state_after_compute = SlotState::RESP;
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
        return int32_t(i);
      }
    }
    return -1;
  }

  Slot&       at(uint32_t idx)       { return slots_[idx]; }
  const Slot& at(uint32_t idx) const { return slots_[idx]; }
  std::vector<Slot>&       slots()       { return slots_; }
  const std::vector<Slot>& slots() const { return slots_; }
  size_t size() const { return slots_.size(); }

private:
  static void reset_slot(Slot& s) {
    s.in_use = false;
    s.state = SlotState::ISSUE;
    s.pending_mem = 0;
    s.compute_cycles_remaining = 0;
    s.walk_done = false;
    s.next_state_after_compute = SlotState::RESP;
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

  std::vector<Slot> slots_;
};

// ────────────────────────────────────────────────────────────────────
// ReformationEngine — Phase 3-A2 same-warp callback reformation.
// Walkers push per-lane yield candidates into the queue; tick()
// drains them into batched CB_YIELD rsps grouped by
// (warp_id, sbt_idx), respecting the per-warp "callback in flight"
// gate so two CB_YIELDs for the same warp never overlap.
//
// In SystemC this becomes SC_MODULE(ReformationEngine) — the
// per-warp gate is a small register file, the queue is a BRAM.
// ────────────────────────────────────────────────────────────────────
class ReformationEngine {
public:
  ReformationEngine(std::vector<SimChannel<RtuRsp>>& rsp_out,
                    PerfStats& perf)
    : rsp_out_(rsp_out), perf_(perf) {}

  void reset() {
    queue_.clear();
    warp_cb_inflight_.fill(false);
  }

  // Walkers push directly into this queue (one entry per yielded
  // (slot, lane)). Exposed by reference so walker call sites stay
  // unchanged.
  std::deque<QueueEntry>& queue() { return queue_; }

  // SfuUnit signals callback completion via CB_ACTION on the
  // request port; drain_requests forwards that to clear the
  // per-warp gate so the next CB_YIELD on the same warp can fire.
  void warp_cb_clear(uint32_t warp_id) {
    warp_cb_inflight_[warp_id] = false;
  }

  void tick() {
    if (rsp_out_.empty()) return;
    auto& port = rsp_out_.at(0);
    while (!queue_.empty()) {
      if (port.full()) break;
      // Same-warp serialization gate: a CB_YIELD raises an M-mode
      // trap on the warp (mepc/mtvec/mscratch_tmask). Firing a
      // second CB_YIELD on the same warp before its CB_RET drains
      // would clobber the trap CSRs and lose the return path. So
      // pick the first queue entry whose warp is not already
      // mid-callback.
      auto anchor_it = queue_.end();
      for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (!warp_cb_inflight_[it->warp_id]) { anchor_it = it; break; }
      }
      if (anchor_it == queue_.end()) break;  // every warp busy
      uint32_t anchor_warp = anchor_it->warp_id;
      uint32_t anchor_sbt  = anchor_it->sbt_idx;
      RtuRsp rsp;
      rsp.kind     = RtuRspKind::CB_YIELD;
      rsp.warp_id  = anchor_warp;
      rsp.block_id = 0;        // CB_YIELD doesn't carry a parked trace;
      rsp.trace    = nullptr;  // SfuUnit drain only reads warp_id.
      uint32_t cb_mask = 0;
      auto it = queue_.begin();
      while (it != queue_.end()) {
        if (it->warp_id != anchor_warp || it->sbt_idx != anchor_sbt) {
          ++it; continue;
        }
        uint8_t t = it->lane;
        if (cb_mask & (1u << t)) {
          // Lane already grouped for this CB_YIELD — defer the second
          // candidate (multi-yield per lane) to a future reformation
          // pass so the kernel doesn't see two conflicting writes
          // into VX_RT_CB_HANDLE in one trap.
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
        it = queue_.erase(it);
      }
      rsp.cb_active_mask = cb_mask;
      port.send(rsp);
      warp_cb_inflight_[anchor_warp] = true;
      // §8.9 per-shader-type callback counters. cb_type is uniform
      // within a batched yield (grouped by sbt_idx) — sample any
      // active lane.
      ++perf_.reformation_yields;
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        if (!(cb_mask & (1u << t))) continue;
        switch (rsp.cb_type[t]) {
          case VX_RT_CB_TYPE_ANYHIT: ++perf_.ahs_callbacks;  break;
          case VX_RT_CB_TYPE_PROC:   ++perf_.is_callbacks;   break;
          case VX_RT_CB_TYPE_CHS:    ++perf_.chs_callbacks;  break;
          case VX_RT_CB_TYPE_MISS:   ++perf_.miss_callbacks; break;
          default:                                            break;
        }
        break;  // one sample per yield is the per-yield-type counter
      }
      DT(3, "rtu-core reform cb_yield: warp=" << anchor_warp
            << ", sbt=" << anchor_sbt
            << ", cb_mask=0x" << std::hex << cb_mask << std::dec);
    }
  }

private:
  std::vector<SimChannel<RtuRsp>>& rsp_out_;
  PerfStats&                        perf_;
  std::deque<QueueEntry>            queue_;
  std::array<bool, VX_CFG_NUM_WARPS> warp_cb_inflight_{};
};

}  // namespace

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

  // Sub-modules bind references to perf_stats_ / pool_ / reform_;
  // declaration order in the private section below matches this init
  // list so -Wreorder stays clean.
  explicit Impl(RtuCore* simobject)
    : simobject_(simobject)
    , pool_(VX_CFG_RTU_CONTEXT_POOL)
    , perf_stats_()
    , reform_(simobject->rtu_rsp_out, perf_stats_)
    , flat_walker_(perf_stats_, reform_.queue())
    , bvh4_walker_(perf_stats_, reform_.queue())
    , mem_engine_(pool_.slots(),
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
                         "reformation_yields=%llu coh_hits=%llu coh_misses=%llu "
                         "walker_cycles=%llu walker_busy_ticks=%llu\n",
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
                 (unsigned long long)p.coherency_misses,
                 (unsigned long long)p.walker_cycles_total,
                 (unsigned long long)p.walker_busy_ticks);
  }

  void reset() {
    pool_.reset();
    mem_engine_.reset();
    reform_.reset();
    perf_stats_ = RtuCore::PerfStats();
    last_compute_signature_ = 0;
  }

  void tick() {
    mem_engine_.drain_mem_rsp();
    drain_requests();
    mem_engine_.issue_memory();
    compute_intersections();
    reform_.tick();
    emit_completions();
  }

  void drain_requests() {
    for (auto& ch : simobject_->rtu_req_in) {
      while (!ch.empty()) {
        const RtuReq& req = ch.peek();
        if (req.kind == RtuReqKind::CB_ACTION) {
          // Phase 3-A2: per-lane CB_RET action. Each active lane in
          // the packet carries its own slot handle (cb_handle, written
          // by SfuUnit at process_cb_ret time from VX_RT_CB_HANDLE).
          // The gathered batch may have routed lanes from MULTIPLE
          // slots into one virtual warp, so we look each lane up by
          // handle and apply ACCEPT/IGNORE/TERMINATE on its own slot.
          for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
            if (((req.tmask_bits >> t) & 1u) == 0) continue;
            uint32_t handle = req.cb_handle[t];
            if (handle >= pool_.size()) continue;
            Slot& s = pool_.at(handle);
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
            // minimum has single-yield-per-lane traversal, so the
            // slot transitions straight to RESP (a richer multi-yield
            // traversal would loop back to COMPUTE for the lane's
            // remaining candidates).
            //
            // Phase 5 VX_RT_CB_DONE: the CHS dispatcher has finished
            // shading the already-committed hit; no hit-state
            // mutation, just drain so the slot can transition to RESP.
            l.cb_pending = false;
            // If this was the last cb_pending lane in the slot, the
            // slot is fully resolved → RESP. Otherwise stay IN_QUEUE
            // for the next batched dispatch.
            bool any_pending = false;
            for (auto const& ll : s.lanes) {
              if (ll.cb_pending) { any_pending = true; break; }
            }
            if (!any_pending) s.state = State::RESP;
          }
          // Clear this warp's "callback in flight" gate so the next
          // queued CB_YIELD for the same warp (e.g. the second SBT
          // group in the divergent-SBT smoke) can be emitted.
          reform_.warp_cb_clear(req.warp_id);
          DT(3, "rtu-core cb_action applied (queue): tmask=0x"
                << std::hex << req.tmask_bits << std::dec);
          ch.pop();
          continue;
        }
        // §8.6: TRACE_NEW now arrives with its slot pre-allocated by
        // the per-core RtuUnit (req.slot_idx). drain_requests just
        // populates the slot's req/lane state and lets the rest of
        // the FSM (issue_memory → compute → emit_completions) drive
        // it to terminal. The pool-full case is handled at TRACE
        // dispatch time (backpressure stalls the SFU input head)
        // instead of here.
        uint32_t idx = req.slot_idx;
        if (idx >= pool_.size()) {
          // Defensive: malformed packet. Drop and continue rather
          // than crash; the kernel will see no progress on its handle.
          ch.pop();
          continue;
        }
        Slot& s = pool_.at(idx);
        // §8.6: promote RESERVED → ISSUE now that the populated req
        // packet has arrived. From here on the standard FSM
        // (issue_memory → compute → emit_completions) drives the
        // slot to RESP.
        s.req   = req;
        s.state = State::ISSUE;
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

  // §step-5/§8.7 slim orchestrator. The §8.9 two-pass octant-signature
  // coherency loop picks WHICH slot to process next; the actual
  // per-lane traversal is delegated to FlatWalker / Bvh4Walker by
  // scene_kind. §8.7 adds a one-shot walk + multi-tick drain on top:
  // the slot's first tick in COMPUTE runs the walker AND charges the
  // BoxPe + TriPe pipeline cycle cost for every box / tri test it
  // issued; subsequent ticks just decrement compute_cycles_remaining
  // until the pipe drains, then advance the slot to its
  // next_state_after_compute (RESP if no CB_YIELD queued, else
  // IN_QUEUE).
  void compute_intersections() {
    auto& slots = pool_.slots();
    bool any_drain_this_tick = false;
    for (uint32_t pass = 0; pass < 2; ++pass) {
      for (auto& s : slots) {
        if (!s.in_use || s.state != State::COMPUTE) continue;
        bool sig_matches = (s.coh_signature == last_compute_signature_);
        if (pass == 0 && !sig_matches) continue;  // matching pass
        if (pass == 1 && sig_matches)  continue;  // non-matching pass

        // First entry into COMPUTE → walk + charge cycles. The
        // walker is per-lane and functional (correctness done in
        // one tick); we read the perf counters' delta to learn
        // how many box / tri tests the lane issued, then convert
        // to pipeline cycles via BoxPe::cycles_for / TriPe::cycles_for.
        if (!s.walk_done) {
          if (sig_matches) ++perf_stats_.coherency_hits;
          else             ++perf_stats_.coherency_misses;
          last_compute_signature_ = s.coh_signature;

          uint64_t box_before = perf_stats_.bvh_box_tests;
          uint64_t tri_before = perf_stats_.bvh_tri_tests;

          bool any_cb_pending = false;
          uint32_t slot_idx = uint32_t(&s - &slots[0]);
          for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
            LaneState& l = s.lanes[t];
            if (!l.active) continue;
            bool queued = (l.scene_kind == kRtuSceneKindBvh4)
                            ? bvh4_walker_.walk_lane(s, l, t, slot_idx)
                            : flat_walker_.walk_lane(s, l, t, slot_idx);
            if (queued) any_cb_pending = true;
          }

          uint32_t box_delta = uint32_t(perf_stats_.bvh_box_tests - box_before);
          uint32_t tri_delta = uint32_t(perf_stats_.bvh_tri_tests - tri_before);
          uint32_t cycles = BoxPe::cycles_for(box_delta)
                          + TriPe::cycles_for(tri_delta);
          s.compute_cycles_remaining = cycles;
          s.next_state_after_compute = any_cb_pending ? State::IN_QUEUE
                                                       : State::RESP;
          s.walk_done = true;
          perf_stats_.walker_cycles_total += cycles;
        }

        // Drain one cycle this tick. If we hit 0, the pipe is done
        // and the slot advances. walk_done is reset by SlotPool
        // when the slot is later freed (or recycled via allocate()
        // / reset_slot()) so a future allocation of the same index
        // starts clean.
        if (s.compute_cycles_remaining > 0) {
          --s.compute_cycles_remaining;
          any_drain_this_tick = true;
        }
        if (s.compute_cycles_remaining == 0) {
          // Drain complete. If the walker set up CB_YIELD entries
          // for one or more lanes (cb_pending=true), push them
          // into the reformation queue now — NOT during the walk
          // tick — so reform_.tick() can't emit CB_YIELD until the
          // slot has actually finished its PE work AND advanced to
          // IN_QUEUE. Otherwise the matching CB_ACTION from the
          // dispatcher would arrive while the slot was still in
          // COMPUTE and drain_requests would drop it (state-gate),
          // hanging the test (the original §8.7 regression on
          // rtu_smoke_miss / chs / is / ahs / sbt).
          if (s.next_state_after_compute == State::IN_QUEUE) {
            uint32_t slot_idx = uint32_t(&s - &slots[0]);
            for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
              const LaneState& l = s.lanes[t];
              if (!l.active || !l.cb_pending) continue;
              QueueEntry e{slot_idx, s.req.warp_id, uint8_t(t),
                           l.sbt_idx, l.cb_type,
                           l.cand_t, l.cand_u, l.cand_v, l.cand_prim};
              reform_.queue().push_back(e);
            }
          }
          s.state = s.next_state_after_compute;
        }
      }
    }
    if (any_drain_this_tick) ++perf_stats_.walker_busy_ticks;
  }

  void emit_completions() {
    if (simobject_->rtu_rsp_out.empty()) return;
    auto& port = simobject_->rtu_rsp_out.at(0);
    for (auto& s : pool_.slots()) {
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
      // §8.6: rsp.slot_idx tells SfuUnit which parked WAIT trace
      // (wait_parked_[wid][slot]) to deliver this to. The slot is
      // NOT freed here; SfuUnit calls RtuCore::free_slot() once the
      // kernel's WAIT actually consumes the result (which may be
      // many cycles later, or already in flight when the rsp lands).
      // Until then the slot sits in EMITTED so emit_completions
      // doesn't re-send.
      rsp.slot_idx = uint32_t(&s - &pool_.slots()[0]);
      port.send(rsp);
      DT(3, "rtu-core complete: tag=" << s.req.tag << ", slot=" << rsp.slot_idx);
      s.state = State::EMITTED;
    }
  }

  const RtuCore::PerfStats& perf_stats() const { return perf_stats_; }

  // §8.6 async ray pool. Direct, non-channel allocator API consumed
  // by the per-core RtuUnit so vx_rt_trace can pre-bind a real
  // handle (= slot index) at issue time and vx_rt_wait can free the
  // slot once its TERMINAL has drained.
  int32_t allocate_slot() {
    return pool_.allocate();
  }
  void free_slot(uint32_t slot_idx) {
    if (slot_idx >= pool_.size()) return;
    Slot& s = pool_.at(slot_idx);
    s.in_use = false;
    s.state  = State::ISSUE;
    // Reset per-lane state so the slot is ready for the next
    // allocate(); mirrors the per-slot reset done by SlotPool::reset().
    for (auto& l : s.lanes) {
      l.active = false;
      l.cb_pending = false;
      l.header_parsed = false;
      l.hit = false;
    }
  }

private:
  RtuCore*           simobject_;
  SlotPool           pool_;
  RtuCore::PerfStats perf_stats_;
  ReformationEngine  reform_;
  // §step-5/6/7: walker + memory sub-modules. They bind refs to
  // perf_stats_, reform_.queue() and pool_.slots() at construction,
  // so declaration order MUST match the init list above (-Wreorder).
  FlatWalker         flat_walker_;
  Bvh4Walker         bvh4_walker_;
  MemoryEngine       mem_engine_;
  // §8.9 coherency gather: octant signature of the most recently
  // processed slot. Initialized to 0 (all-positive-axis ray).
  uint8_t            last_compute_signature_ = 0;
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

int32_t RtuCore::allocate_slot()        { return impl_->allocate_slot(); }
void    RtuCore::free_slot(uint32_t i)  { impl_->free_slot(i);           }
