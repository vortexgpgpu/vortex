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
// PRISM RtuCore implementation.

#include "rtu_core.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include "rtu_types.h"
#include "rtu_isect.h"       // BoxPe / TriPe pipeline depths
#include "rtu_walker.h"      // FlatWalker / Bvh4Walker
#include "rtu_memory.h"      // MemoryEngine
#include "socket.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;
using namespace vortex::rtu;

namespace {

// ────────────────────────────────────────────────────────────────────
// SlotPool — the resident-trace array. A slot stages one vx_rt_wtrace: its
// rays, its control state, its result record. Slots partition statically across
// the cores this RTU serves, so one core cannot spend another's share — the
// same guarantee the RTL's per-core credit counter gives, without a cross-core
// protocol.
// ────────────────────────────────────────────────────────────────────
class SlotPool {
public:
  explicit SlotPool(uint32_t size) : slots_(size) {}

  void reset() {
    for (auto& s : slots_) s = Slot{};
    used_by_core_.clear();
    next_age_ = 0;
  }

  // First-fit within the calling core's quota. Returns the slot index, or -1 if
  // the core already holds its share. The slot is left RESERVED: the FSM skips
  // it until drain_requests populates it with the ray, so a slot with no active
  // lanes can never fall through to a spurious terminal record.
  int32_t allocate(uint32_t core_id) {
    uint32_t& used = used_by_core_[core_id];
    if (used >= kRtuSlotsPerCore) return -1;
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (slots_[i].in_use) continue;
      Slot& s = slots_[i];
      s = Slot{};
      s.in_use  = true;
      s.state   = SlotState::RESERVED;
      s.core_id = core_id;
      s.age     = next_age_++;
      ++used;
      return int32_t(i);
    }
    return -1;
  }

  void free(uint32_t idx) {
    Slot& s = slots_.at(idx);
    if (!s.in_use) return;
    uint32_t& used = used_by_core_[s.core_id];
    if (used > 0) --used;
    s = Slot{};
  }

  Slot&       at(uint32_t idx)       { return slots_.at(idx); }
  std::vector<Slot>&       slots()       { return slots_; }
  size_t size() const { return slots_.size(); }

private:
  std::vector<Slot> slots_;
  std::unordered_map<uint32_t, uint32_t> used_by_core_;
  uint64_t next_age_ = 0;
};

// ────────────────────────────────────────────────────────────────────
// ReformationEngine — same-warp callback reformation. The walk pushes per-lane
// yield candidates into the queue; tick() drains them into batched CB_YIELD
// responses grouped by (warp_id, sbt_idx), respecting the per-warp
// "callback in flight" gate so two CB_YIELDs for the same warp never overlap.
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

  std::deque<QueueEntry>& queue() { return queue_; }

  // SfuUnit signals callback completion via CB_ACTION on the request port;
  // drain_requests forwards that here to clear the per-warp gate so the next
  // CB_YIELD on the same warp can fire.
  void warp_cb_clear(uint32_t warp_id) {
    warp_cb_inflight_[warp_id] = false;
  }

  void tick() {
    if (rsp_out_.empty()) return;
    auto& port = rsp_out_.at(0);
    while (!queue_.empty()) {
      if (port.full()) break;
      // Same-warp serialization gate: a candidate is returned to the issuing
      // warp, which reads its payload before issuing CONTINUE. Firing a second
      // candidate on the same warp before its CONTINUE drains would overwrite
      // that payload underneath the warp. So pick the first queue entry whose
      // warp is not already mid-candidate.
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
          // Lane already grouped for this CB_YIELD — defer the second candidate
          // to a future reformation pass so the kernel doesn't see two
          // conflicting payloads in one trap.
          ++it; continue;
        }
        cb_mask |= (1u << t);
        rsp.cb_type[t]            = it->cb_type;
        // Per-lane yield status returned to the warp's WAIT: ANYHIT candidate
        // -> YIELD_ANYHIT, procedural -> YIELD_PROC.
        rsp.status[t]             = (it->cb_type == VX_RT_CB_TYPE_PROC)
                                       ? VX_RT_STS_YIELD_PROC
                                       : VX_RT_STS_YIELD_ANYHIT;
        rsp.cb_handle[t]          = it->slot_idx;
        rsp.cb_sbt_idx[t]         = it->sbt_idx;
        rsp.hit_t[t]              = it->cand_t;
        rsp.hit_attr[t]           = it->hit_attr;
        rsp.hit_bary_u[t]         = it->cand_u;
        rsp.hit_bary_v[t]         = it->cand_v;
        rsp.hit_primitive_id[t]   = it->cand_prim;
        rsp.hit_geometry_index[t] = it->cand_geometry;
        rsp.hit_instance_id[t]     = it->cand_instance;
        rsp.hit_instance_custom[t] = it->cand_custom;
        // Object-space ray of the candidate, staged so the AHS/IS dispatcher
        // can read gl_ObjectRay{Origin,Direction}EXT.
        rsp.obj_o_x[t]            = it->cand_obj_o[0];
        rsp.obj_o_y[t]            = it->cand_obj_o[1];
        rsp.obj_o_z[t]            = it->cand_obj_o[2];
        rsp.obj_d_x[t]            = it->cand_obj_d[0];
        rsp.obj_d_y[t]            = it->cand_obj_d[1];
        rsp.obj_d_z[t]            = it->cand_obj_d[2];
        it = queue_.erase(it);
      }
      rsp.cb_active_mask = cb_mask;
      port.send(rsp);
      warp_cb_inflight_[anchor_warp] = true;
      ++perf_.reformation_yields;
      // cb_type is uniform within a batched yield (grouped by sbt_idx), so one
      // sample names the per-yield-type counter.
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        if (!(cb_mask & (1u << t))) continue;
        switch (rsp.cb_type[t]) {
          case VX_RT_CB_TYPE_ANYHIT: ++perf_.ahs_callbacks;  break;
          case VX_RT_CB_TYPE_PROC:   ++perf_.is_callbacks;   break;
          case VX_RT_CB_TYPE_CHS:    ++perf_.chs_callbacks;  break;
          case VX_RT_CB_TYPE_MISS:   ++perf_.miss_callbacks; break;
          default:                                            break;
        }
        break;
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
// Two arrays, two jobs.
//
// The SLOT POOL holds resident traces. A slot is claimed at trace issue, filled
// with its rays, queued, promoted when the contexts it needs are free, and
// released when its record has been consumed.
//
// The CONTEXT ARRAY holds live ray walks. A context is bound to one lane of one
// slot at promote — all-or-nothing, so a trace never holds half its contexts
// and waits for the rest — and released the moment its ray retires, which is
// what keeps the slowest ray of a trace from parking the rest.
//
// A context walks by demand fetch: the walker replays the ray, stalls on the
// first node line it does not hold, and the context requests exactly that line.
// The memory front end merges the requests (rtu_memory), the box and tri PEs
// are shared and issue one test per cycle, and the whole thing runs as a barrel
// scheduler over the context array.

class RtuCore::Impl {
public:
  explicit Impl(RtuCore* simobject)
    : simobject_(simobject)
    , pool_(VX_CFG_RTU_NUM_SLOTS)
    , contexts_(VX_CFG_RTU_NUM_CTX)
    , perf_stats_()
    , reform_(simobject->rtu_rsp_out, perf_stats_)
    , mem_engine_(contexts_,
                  simobject->dcache_req_out,
                  simobject->dcache_rsp_in,
                  perf_stats_)
  {}

  // Stats dump. Opt-in via VX_RTU_STATS (any non-empty value). This is the
  // instrumentation the slot/context/merge sweep reads.
  ~Impl() {
    const char* env = std::getenv("VX_RTU_STATS");
    if (env == nullptr || env[0] == '\0') return;
    const auto& p = perf_stats_;
    std::fprintf(stderr,
        "[rtu-stats] slots=%u ctx=%u merge=%u\n"
        "[rtu-stats] rays_issued=%llu rays_hit=%llu rays_miss=%llu\n"
        "[rtu-stats] mem_reads=%llu fetches_merged=%llu mshr_full=%llu\n"
        "[rtu-stats] bvh_nodes=%llu bvh_leaves=%llu box_tests=%llu tri_tests=%llu\n"
        "[rtu-stats] busy_ticks=%llu front_end_ticks=%llu mem_issue_ticks=%llu\n"
        "[rtu-stats] slot_busy_ticks=%llu ctx_bound_ticks=%llu ctx_active_ticks=%llu\n"
        "[rtu-stats] cb_ahs=%llu cb_chs=%llu cb_miss=%llu cb_is=%llu reform=%llu\n",
        uint32_t(VX_CFG_RTU_NUM_SLOTS), uint32_t(VX_CFG_RTU_NUM_CTX),
        uint32_t(VX_CFG_RTU_MERGE_DEPTH),
        (unsigned long long)p.rays_issued,
        (unsigned long long)p.rays_hit,
        (unsigned long long)p.rays_miss,
        (unsigned long long)p.mem_reads,
        (unsigned long long)p.fetches_merged,
        (unsigned long long)p.mshr_full_stalls,
        (unsigned long long)p.bvh_nodes_fetched,
        (unsigned long long)p.bvh_leaves_fetched,
        (unsigned long long)p.bvh_box_tests,
        (unsigned long long)p.bvh_tri_tests,
        (unsigned long long)p.rtu_busy_ticks,
        (unsigned long long)p.front_end_busy_ticks,
        (unsigned long long)p.mem_issue_ticks,
        (unsigned long long)p.slot_busy_ticks,
        (unsigned long long)p.ctx_bound_ticks,
        (unsigned long long)p.ctx_active_ticks,
        (unsigned long long)p.ahs_callbacks,
        (unsigned long long)p.chs_callbacks,
        (unsigned long long)p.miss_callbacks,
        (unsigned long long)p.is_callbacks,
        (unsigned long long)p.reformation_yields);
  }

  void reset() {
    pool_.reset();
    for (auto& cx : contexts_) cx = Context{};
    mem_engine_.reset();
    reform_.reset();
    perf_stats_ = RtuCore::PerfStats();
    last_signature_ = 0;
    rr_exec_ = 0;
    step_cycles_left_   = 0;
  }

  void tick() {
    mem_engine_.drain_mem_rsp();
    drain_requests();
    promote_slots();
    run_contexts();
    mem_engine_.issue_memory();
    complete_slots();
    reform_.tick();
    emit_completions();
    sample_occupancy();
  }

  // Accept stage: drain at most one RtuReq per tick, round-robin across the
  // rtu_req_in[] channels (mirrors the OM / DXA accept stages).
  void drain_requests() {
    auto& chs = simobject_->rtu_req_in;
    for (uint32_t i = 0; i < chs.size(); ++i) {
      uint32_t cid = (rr_req_ + i) % chs.size();
      auto& ch = chs.at(cid);
      if (ch.empty()) continue;

      const RtuReq& req = ch.peek();
      if (req.kind == RtuReqKind::CB_ACTION) {
        apply_cb_action(req);
        ch.pop();
        rr_req_ = (cid + 1) % chs.size();
        return;
      }

      // TRACE_NEW arrives with its slot already claimed by the per-core RtuUnit
      // at issue (req.slot_idx) — that claim is the credit gate, and it is why
      // a full pool can never jam the SFU's in-order input head.
      uint32_t idx = req.slot_idx;
      if (idx >= pool_.size()) {
        // Defensive: malformed packet. Drop and rotate so it cannot starve the
        // other channels.
        ch.pop();
        rr_req_ = (cid + 1) % chs.size();
        return;
      }
      Slot& s = pool_.at(idx);
      s.req   = req;
      s.state = SlotState::READY;
      uint32_t first_active = uint32_t(-1);
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        if (s.req.tmask_bits & (1u << t)) {
          s.lanes[t].active = true;
          if (first_active == uint32_t(-1)) first_active = t;
        }
      }
      // Octant signature from the first active lane's ray direction.
      if (first_active != uint32_t(-1)) {
        uint8_t sig = 0;
        if (s.req.dir_x[first_active] < 0.f) sig |= 0x1;
        if (s.req.dir_y[first_active] < 0.f) sig |= 0x2;
        if (s.req.dir_z[first_active] < 0.f) sig |= 0x4;
        s.coh_signature = sig;
      }
      ch.pop();
      ++perf_stats_.rays_issued;
      DT(3, "rtu-core accept: tag=" << s.req.tag << ", slot=" << idx);
      rr_req_ = (cid + 1) % chs.size();
      return;
    }
  }

  // A CONTINUE resolves the candidates a slot yielded. The batch may have
  // routed lanes from several slots into one virtual warp, so each lane carries
  // its own slot handle and is applied to its own slot.
  void apply_cb_action(const RtuReq& req) {
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (((req.tmask_bits >> t) & 1u) == 0) continue;
      uint32_t handle = req.cb_handle[t];
      if (handle >= pool_.size()) continue;
      Slot& s = pool_.at(handle);
      if (!s.in_use || s.state != SlotState::IN_QUEUE) continue;
      LaneState& l = s.lanes[t];
      if (!l.cb_pending) continue;
      uint32_t action = req.cb_action[t];
      if (action == VX_RT_CB_ACCEPT || action == VX_RT_CB_TERMINATE) {
        l.hit = true;
        // A procedural (IS) accept commits the shader's own hit_t; a triangle
        // AHS keeps the geometric candidate t. Either way the hitAttribute the
        // shader returned belongs to the hit it just accepted — bind it here,
        // or a later verdict (which carries no attribute) would overwrite it.
        l.hit_t    = (l.cb_type == VX_RT_CB_TYPE_PROC)
                       ? req.cb_hit_t[t] : l.cand_t;
        l.hit_attr = req.cb_attr[t];
        l.hit_u    = l.cand_u;
        l.hit_v    = l.cand_v;
        l.hit_prim = l.cand_prim;
        l.hit_geometry = l.cand_geometry;
        l.hit_instance_id     = l.cand_instance;
        l.hit_instance_custom = l.cand_custom;
        // The accepted candidate becomes the committed hit, so its object-space
        // ray is the committed one.
        for (int k = 0; k < 3; ++k) {
          l.hit_obj_o[k] = l.cand_obj_o[k];
          l.hit_obj_d[k] = l.cand_obj_d[k];
        }
      }
      // IGNORE leaves the committed hit alone; DONE means the CHS dispatcher has
      // finished shading an already-committed hit. Traversal is
      // single-yield-per-lane, so either way the lane is resolved and the slot
      // drops to its terminal record once every yielding lane has answered.
      l.cb_pending = false;
      bool any_pending = false;
      for (auto const& ll : s.lanes) {
        if (ll.cb_pending) { any_pending = true; break; }
      }
      if (!any_pending) s.state = SlotState::RESP;
    }
    // Clear this warp's callback-in-flight gate so the next queued CB_YIELD for
    // the same warp (e.g. the second SBT group) can be emitted.
    reform_.warp_cb_clear(req.warp_id);
    DT(3, "rtu-core cb_action applied: tmask=0x"
          << std::hex << req.tmask_bits << std::dec);
  }

  // The scheduler. The oldest READY slot goes first and NO younger slot may
  // bypass it: a 16-ray trace waiting for 16 free contexts would otherwise be
  // starved indefinitely by a stream of 4-ray traces that keep finding four.
  // It cannot deadlock — every context currently held belongs to a walk that
  // completes without needing anything the blocked slot holds.
  void promote_slots() {
    uint32_t best = uint32_t(-1);
    uint64_t best_age = 0;
    for (uint32_t i = 0; i < pool_.size(); ++i) {
      const Slot& s = pool_.at(i);
      if (!s.in_use || s.state != SlotState::READY) continue;
      if (best == uint32_t(-1) || s.age < best_age) {
        best = i;
        best_age = s.age;
      }
    }
    if (best == uint32_t(-1)) return;

    Slot& s = pool_.at(best);
    uint32_t need = 0;
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (s.lanes[t].active) ++need;
    }
    uint32_t avail = 0;
    for (const auto& cx : contexts_) {
      if (!cx.valid) ++avail;
    }
    // All-or-nothing: a slot that took some contexts and waited for the rest
    // would reintroduce hold-and-wait at the context level.
    if (avail < need) return;

    if (s.coh_signature == last_signature_) ++perf_stats_.coherency_hits;
    else                                    ++perf_stats_.coherency_misses;
    last_signature_ = s.coh_signature;

    uint32_t next_free = 0;
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!s.lanes[t].active) continue;
      while (contexts_[next_free].valid) ++next_free;
      bind_context(next_free, best, t, s.req.scene_root[t]);
      ++next_free;
    }
    s.ctx_pending = need;
    s.state = SlotState::WALKING;
    DT(3, "rtu-core promote: slot=" << best << ", contexts=" << need);
  }

  void bind_context(uint32_t ci, uint32_t slot, uint32_t lane,
                    uint32_t scene_root) {
    Context& cx = contexts_[ci];
    cx = Context{};
    cx.valid = true;
    cx.slot  = slot;
    cx.lane  = lane;
    cx.base_line = uint64_t(scene_root) & kRtuLineMask;
    cx.byte_off  = uint32_t(uint64_t(scene_root) - cx.base_line);
    // The ray opens on the reciprocal (1/dir) setup and the scene-header fetch,
    // which cost front-end states like any other step, then drain the reciprocal
    // pipeline before the first node can be tested.
    cx.state      = CtxState::PE;
    cx.next_state = CtxState::PROBE;
    cx.fsm_states = kRtuStatesPerRay;
    cx.img_states = kRtuImageStatesPerRay;
    cx.pe_lat     = kRtuSetupLatency;
  }

  // One tick of the traversal engine: step every context whose line has landed,
  // then run the shared front end for one cycle.
  void run_contexts() {
    for (uint32_t i = 0; i < contexts_.size(); ++i) {
      Context& cx = contexts_[i];
      if (!cx.valid || cx.state != CtxState::PROBE) continue;
      probe_context(cx);
    }
    run_front_end();
  }

  // Replay this ray's walk against the lines the context holds. The walk is
  // deterministic, so the counters it reports are cumulative over the prefix it
  // could execute; the difference against what has already been charged is the
  // work the last-arrived node unlocked. Where it stops is the next fetch.
  void probe_context(Context& cx) {
    Slot& s = pool_.at(cx.slot);
    SceneView sv;
    sv.lines     = &cx.lines;
    sv.base_line = cx.base_line;
    sv.byte_off  = cx.byte_off;

    PerfStats walk;   // cumulative over the replayed prefix
    LaneState result = s.lanes[cx.lane];
#if VX_CFG_RTU_BVH_WIDTH == 0
    WalkResult r = flat_walker_.walk_lane(s.req, cx.lane, sv, result, walk);
#else
    WalkResult r = bvh4_walker_.walk_lane(s.req, cx.lane, sv, result, walk);
#endif

    uint32_t box     = uint32_t(walk.bvh_box_tests         - cx.chg_box);
    uint32_t tri     = uint32_t(walk.bvh_tri_tests         - cx.chg_tri);
    uint32_t inst    = uint32_t(walk.bvh_instance_descents - cx.chg_inst);
    uint32_t restart = uint32_t(walk.bvh_stack_restarts    - cx.chg_restart);
    uint32_t nodes   = uint32_t(walk.bvh_nodes_fetched     - cx.chg_nodes);
    uint32_t leaves  = uint32_t(walk.bvh_leaves_fetched    - cx.chg_leaves);
    perf_stats_.bvh_box_tests         += box;
    perf_stats_.bvh_tri_tests         += tri;
    perf_stats_.bvh_instance_descents += inst;
    perf_stats_.bvh_stack_restarts    += restart;
    perf_stats_.bvh_nodes_fetched     += nodes;
    perf_stats_.bvh_leaves_fetched    += leaves;
    cx.chg_box     = walk.bvh_box_tests;
    cx.chg_tri     = walk.bvh_tri_tests;
    cx.chg_inst    = walk.bvh_instance_descents;
    cx.chg_restart = walk.bvh_stack_restarts;
    cx.chg_nodes   = walk.bvh_nodes_fetched;
    cx.chg_leaves  = walk.bvh_leaves_fetched;

    // What this step costs the shared front end: one FSM state per node, leaf,
    // triangle, child box and instance descent that the newly-arrived line
    // unlocked — and the front end retires those one at a time, for the whole RTU.
    cx.fsm_states = nodes  * kRtuStatesPerNode
                  + box    * kRtuStatesPerBox
                  + leaves * kRtuStatesPerLeaf
                  + tri    * kRtuStatesPerTri
                  + inst   * kRtuStatesPerInst;
    cx.img_states = nodes  * kRtuImageStatesPerNode
                  + box    * kRtuImageStatesPerBox
                  + leaves * kRtuImageStatesPerLeaf
                  + tri    * kRtuImageStatesPerTri
                  + inst   * kRtuImageStatesPerInst;
    // Behind the last test fed, the PE pipeline drains while the context is
    // PARKED — box and tri results both carry their context id, so waiting for
    // one costs the shared front end nothing. A trail restart re-descends every
    // subtree the short stack had to evict.
    uint32_t lat = 0;
    if (box) lat = std::max(lat, BoxPe::pipe_depth());
    if (tri) lat = std::max(lat, TriPe::pipe_depth());
    lat += kRtuXformLatency * inst;
    lat += VX_CFG_RTU_STACK_DEPTH * restart;
    cx.pe_lat = lat;
    perf_stats_.walker_cycles_total += cx.fsm_states * kRtuPhasesPerStep
                                     + cx.img_states * kRtuPhasesImageAdd + lat;

    if (r.stalled) {
      cx.req_addr   = sv.miss_line;
      cx.next_state = CtxState::REQ;
    } else {
      s.lanes[cx.lane] = result;   // carries cb_pending if the ray yielded
      cx.next_state = CtxState::DONE;
    }
    cx.state = (cx.fsm_states || lat) ? CtxState::PE : cx.next_state;
    if (cx.state == CtxState::DONE) retire_context(cx);
  }

  // The traversal front end. One select-execute machine serves the whole context
  // array, so exactly one context advances one FSM state at a time — however many
  // contexts are resident. A state costs two cycles, or three when it has to stage
  // the fetched image before decoding it. A context that has run out of states
  // drains its PE pipeline, which is not front-end work and does overlap.
  void run_front_end() {
    const uint32_t n = uint32_t(contexts_.size());
    bool busy = false;

    if (step_cycles_left_ > 0) {
      --step_cycles_left_;   // the selected context is still executing its state
    } else {
      for (uint32_t i = 0; i < n; ++i) {
        uint32_t idx = (rr_exec_ + i) % n;
        Context& cx = contexts_[idx];
        if (!cx.valid || cx.state != CtxState::PE || cx.fsm_states == 0) continue;
        uint32_t cost = kRtuPhasesPerStep;
        if (cx.img_states != 0) {
          --cx.img_states;
          cost += kRtuPhasesImageAdd;
        }
        --cx.fsm_states;
        step_cycles_left_ = cost - 1;   // this cycle is the first of the state's cost
        rr_exec_ = (idx + 1) % n;
        ++perf_stats_.front_end_busy_ticks;
        break;
      }
    }

    for (uint32_t i = 0; i < n; ++i) {
      Context& cx = contexts_[i];
      if (!cx.valid || cx.state != CtxState::PE) continue;
      busy = true;
      if (cx.fsm_states != 0) continue;   // still owes the front end
      if (cx.pe_lat > 0) --cx.pe_lat;
      if (cx.pe_lat == 0) {
        cx.state = cx.next_state;
        if (cx.state == CtxState::DONE) retire_context(cx);
      }
    }
    if (busy) ++perf_stats_.walker_busy_ticks;
  }

  // A ray that has finished releases its context immediately — it does not sit
  // bound while its trace's slowest sibling walks on. That tail is exactly what
  // a per-slot static context partition would pay.
  void retire_context(Context& cx) {
    Slot& s = pool_.at(cx.slot);
    assert(s.ctx_pending > 0 && "RTU: context retiring against an idle slot");
    --s.ctx_pending;
    cx = Context{};
  }

  // A slot whose every ray has retired writes its record — or queues the
  // callbacks its rays raised.
  void complete_slots() {
    for (uint32_t i = 0; i < pool_.size(); ++i) {
      Slot& s = pool_.at(i);
      if (!s.in_use || s.state != SlotState::WALKING) continue;
      if (s.ctx_pending != 0) continue;

      bool any_cb = false;
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        const LaneState& l = s.lanes[t];
        if (!l.active || !l.cb_pending) continue;
        any_cb = true;
        QueueEntry e{i, s.req.warp_id, uint8_t(t),
                     l.sbt_idx, l.cb_type,
                     l.cand_t, l.cand_u, l.cand_v, l.cand_prim,
                     l.cand_geometry,
                     l.cand_instance, l.cand_custom,
                     l.hit_attr,
                     {l.cand_obj_o[0], l.cand_obj_o[1], l.cand_obj_o[2]},
                     {l.cand_obj_d[0], l.cand_obj_d[1], l.cand_obj_d[2]}};
        reform_.queue().push_back(e);
      }
      s.state = any_cb ? SlotState::IN_QUEUE : SlotState::RESP;
    }
  }

  void emit_completions() {
    if (simobject_->rtu_rsp_out.empty()) return;
    auto& port = simobject_->rtu_rsp_out.at(0);
    for (uint32_t i = 0; i < pool_.size(); ++i) {
      Slot& s = pool_.at(i);
      if (!s.in_use || s.state != SlotState::RESP) continue;
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
          rsp.hit_attr[t]          = l.hit_attr;
          rsp.hit_bary_u[t]        = l.hit_u;
          rsp.hit_bary_v[t]        = l.hit_v;
          rsp.hit_primitive_id[t]  = l.hit_prim;
          rsp.hit_instance_id[t]   = l.hit_instance_id;
          rsp.hit_instance_custom[t] = l.hit_instance_custom;
          rsp.hit_geometry_index[t] = l.hit_geometry;
          rsp.obj_o_x[t]           = l.hit_obj_o[0];
          rsp.obj_o_y[t]           = l.hit_obj_o[1];
          rsp.obj_o_z[t]           = l.hit_obj_o[2];
          rsp.obj_d_x[t]           = l.hit_obj_d[0];
          rsp.obj_d_y[t]           = l.hit_obj_d[1];
          rsp.obj_d_z[t]           = l.hit_obj_d[2];
          ++perf_stats_.rays_hit;
        } else {
          rsp.status[t] = VX_RT_STS_DONE_MISS;
          ++perf_stats_.rays_miss;
        }
      }
      // The slot is NOT freed here: it still names the handle the kernel holds,
      // so it stays live until the WAIT that consumes the record calls
      // free_slot(). Until then it sits in EMITTED and is not re-sent.
      rsp.slot_idx = i;
      port.send(rsp);
      DT(3, "rtu-core complete: tag=" << s.req.tag << ", slot=" << i);
      s.state = SlotState::EMITTED;
    }
  }

  void sample_occupancy() {
    uint32_t slots_used = 0;
    for (uint32_t i = 0; i < pool_.size(); ++i) {
      if (pool_.at(i).in_use) ++slots_used;
    }
    uint32_t bound = 0, active = 0;
    for (const auto& cx : contexts_) {
      if (!cx.valid) continue;
      ++bound;
      // A context in WAIT is bound but idle — it is holding 1.9 kb of state to
      // wait for a cache line. bound minus active is what an MSHR-side fix buys
      // and a wider context array does not.
      if (cx.state != CtxState::WAIT) ++active;
    }
    perf_stats_.slot_busy_ticks  += slots_used;
    perf_stats_.ctx_bound_ticks  += bound;
    perf_stats_.ctx_active_ticks += active;
    if (slots_used != 0) ++perf_stats_.rtu_busy_ticks;
  }

  const RtuCore::PerfStats& perf_stats() const { return perf_stats_; }

  int32_t allocate_slot(uint32_t core_id) {
    return pool_.allocate(core_id);
  }
  void free_slot(uint32_t slot_idx) {
    if (slot_idx >= pool_.size()) return;
    pool_.free(slot_idx);
  }

private:
  RtuCore*             simobject_;
  SlotPool             pool_;
  std::vector<Context> contexts_;
  RtuCore::PerfStats   perf_stats_;
  ReformationEngine    reform_;
#if VX_CFG_RTU_BVH_WIDTH == 0
  FlatWalker           flat_walker_;
#else
  Bvh4Walker           bvh4_walker_;
#endif
  MemoryEngine         mem_engine_;
  // Coherency gather: octant signature of the most recently promoted slot.
  uint8_t              last_signature_ = 0;
  uint32_t             rr_req_ = 0;   // request-channel rotation
  uint32_t             rr_exec_ = 0;  // front-end select rotation
  uint32_t             step_cycles_left_   = 0;  // cycles owed by the state in flight
};

// ════════════════════════════════════════════════════════════════════

RtuCore::RtuCore(const SimContext& ctx, const char* name, Socket* /*socket*/)
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

int32_t RtuCore::allocate_slot(uint32_t core_id) {
  return impl_->allocate_slot(core_id);
}
void RtuCore::free_slot(uint32_t i) {
  impl_->free_slot(i);
}
