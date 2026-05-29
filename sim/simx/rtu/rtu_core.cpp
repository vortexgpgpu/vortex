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
#include <cstring>
#include <unordered_map>
#include "cluster.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;

namespace {

constexpr uint64_t kRtuLineMask = ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);

// Phase 1: single-triangle-per-scene cap so the scene fits in one line.
constexpr uint32_t kPhase1MaxTris = 1;

// Phase 2 scene format: per-triangle stride extended from 36 B (9 floats:
// v0/v1/v2 xyz) to 40 B with a trailing uint32 `flags`. Bit 0 = OPAQUE; a
// 0 here triggers an AHS callback yield. Single-triangle scenes still fit
// in one 64 B cache line (16 B header + 40 B = 56 B).
constexpr uint32_t kPhase2TriStride  = 40;
constexpr uint32_t kPhase2TriFlagsOff = 36;
constexpr uint32_t kPhase2TriFlagOpaque = 0x1u;

// Float3 helper for intersection math.
struct float3 {
  float x, y, z;
  float3 operator-(const float3& o) const { return {x-o.x, y-o.y, z-o.z}; }
};
inline float3 cross_(const float3& a, const float3& b) {
  return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}
inline float dot_(const float3& a, const float3& b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Möller-Trumbore ray-triangle intersection.
bool ray_triangle(const float ro[3], const float rd[3],
                  const float v0[3], const float v1[3], const float v2[3],
                  float tmin, float tmax,
                  float& out_t, float& out_u, float& out_v) {
  float3 O = { ro[0], ro[1], ro[2] };
  float3 D = { rd[0], rd[1], rd[2] };
  float3 V0 = { v0[0], v0[1], v0[2] };
  float3 V1 = { v1[0], v1[1], v1[2] };
  float3 V2 = { v2[0], v2[1], v2[2] };

  float3 e1 = V1 - V0;
  float3 e2 = V2 - V0;
  float3 P  = cross_(D, e2);
  float det = dot_(e1, P);
  constexpr float EPS = 1e-6f;
  if (det > -EPS && det < EPS) return false;
  float invDet = 1.0f / det;
  float3 T = O - V0;
  float u = dot_(T, P) * invDet;
  if (u < 0.f || u > 1.f) return false;
  float3 Q = cross_(T, e1);
  float v = dot_(D, Q) * invDet;
  if (v < 0.f || u + v > 1.f) return false;
  float t = dot_(e2, Q) * invDet;
  if (t < tmin || t > tmax) return false;
  out_t = t;
  out_u = u;
  out_v = v;
  return true;
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
  enum class State : uint8_t {
    ISSUE,            // need to issue mem reads for active lanes
    AWAIT,            // mem reads outstanding
    COMPUTE,          // ready to run ray-triangle intersection
    AWAIT_CALLBACK,   // Phase 2: yielded to AHS/IS; waiting for CB_ACTION
    RESP              // terminal status ready to emit
  };

  struct LaneState {
    bool   active = false;
    bool   hit    = false;            // a *committed* hit (best so far)
    float  hit_t  = 0.f;
    float  hit_u  = 0.f;
    float  hit_v  = 0.f;
    uint32_t hit_prim = 0;
    // Phase 2: candidate hit + yield state. When a non-opaque triangle
    // intersects, we stash its attrs here and wait for the dispatcher's
    // CB_RET action to decide whether to commit, discard, or terminate.
    bool   cb_pending      = false;
    uint32_t cb_type       = 0;  // VX_RT_CB_TYPE_ANYHIT | _PROC
    float  cand_t          = 0.f;
    float  cand_u          = 0.f;
    float  cand_v          = 0.f;
    uint32_t cand_prim     = 0;
    bool   line_filled = false;
    std::array<uint8_t, VX_CFG_MEM_BLOCK_SIZE> line_data = {};
    uint32_t line_byte_off = 0;
  };

  struct Slot {
    bool   in_use = false;
    State  state  = State::ISSUE;
    RtuReq req;
    std::array<LaneState, VX_CFG_NUM_THREADS> lanes = {};
    uint32_t pending_mem = 0;
    // Phase 2: true once a CB_YIELD response has been emitted for this
    // slot; suppresses re-emit on subsequent ticks while we wait for the
    // matching CB_ACTION packet. Per-lane cb_pending is *not* cleared on
    // emit — CB_ACTION drain reads it to find which lanes to act on.
    bool   cb_emitted = false;
  };

  explicit Impl(RtuCore* simobject)
    : simobject_(simobject)
    , slots_(VX_CFG_RTU_CONTEXT_POOL)
  {}

  void reset() {
    for (auto& s : slots_) {
      s.in_use = false;
      s.state = State::ISSUE;
      s.pending_mem = 0;
      for (auto& l : s.lanes) {
        l.active = false;
        l.line_filled = false;
      }
    }
    pending_mem_.clear();
    perf_stats_ = RtuCore::PerfStats();
    next_tag_ = 0;
  }

  void tick() {
    drain_mem_rsp();
    drain_requests();
    issue_memory();
    compute_intersections();
    emit_completions();
  }

  void drain_requests() {
    for (auto& ch : simobject_->rtu_req_in) {
      while (!ch.empty()) {
        const RtuReq& req = ch.peek();
        if (req.kind == RtuReqKind::CB_ACTION) {
          // Phase 2: per-lane CB_RET action for an already-parked slot.
          // Match by warp_id alone — Phase 2 limits one outstanding ray
          // per (warp, lane), and the cb_ret packet carries the cb_ret
          // *instruction's* uuid, not the TRACE's. Phase 3 (with per-
          // lane handles) will route by (warp_id, handle).
          int match_idx = -1;
          for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].in_use
                && slots_[i].state == State::AWAIT_CALLBACK
                && slots_[i].req.warp_id == req.warp_id) {
              match_idx = (int)i;
              break;
            }
          }
          if (match_idx < 0) {
            DT(3, "rtu-core cb_action: no matching slot wid=" << req.warp_id
                  << " uuid=" << req.uuid);
            ch.pop();
            continue;
          }
          auto& s = slots_[match_idx];
          for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
            if (((req.tmask_bits >> t) & 1u) == 0) continue;
            LaneState& l = s.lanes[t];
            if (!l.cb_pending) continue;
            uint32_t action = req.cb_action[t];
            if (action == VX_RT_CB_ACCEPT || action == VX_RT_CB_TERMINATE) {
              // Commit the candidate as the closest hit.
              l.hit      = true;
              l.hit_t    = l.cand_t;
              l.hit_u    = l.cand_u;
              l.hit_v    = l.cand_v;
              l.hit_prim = l.cand_prim;
            }
            // IGNORE: leave best_hit unchanged. Phase 2 minimum has
            // single-tri scenes, so any IGNORE collapses to DONE_MISS.
            l.cb_pending = false;
          }
          // Phase 2 minimum: traversal is one-shot. After the callback
          // decision, transition to RESP — Phase 2.1 will loop back to
          // COMPUTE for multi-tri scenes.
          s.state = State::RESP;
          DT(3, "rtu-core cb_action applied: tag=" << s.req.tag);
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
        s.cb_emitted = false;
        for (auto& l : s.lanes) {
          l.active = false;
          l.line_filled = false;
          l.cb_pending = false;
          l.hit = false;
        }
        for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
          if (s.req.tmask_bits & (1u << t)) {
            s.lanes[t].active = true;
          }
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
      if (!s.in_use || s.state != State::ISSUE) continue;
      // Per-lane cache-line fetch (Phase 1: simple — one line per active lane).
      bool any_issued = false;
      bool all_done   = true;
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        if (!s.lanes[t].active) continue;
        if (s.lanes[t].line_filled) continue;
        all_done = false;
        if (port.full()) break;
        uint64_t addr     = uint64_t(s.req.scene_root[t]);
        uint64_t line     = addr & kRtuLineMask;
        uint32_t off      = uint32_t(addr - line);
        s.lanes[t].line_byte_off = off;
        uint32_t tag = next_tag_++;
        MemReq m;
        m.addr    = line;
        m.op      = MemOp::LD;
        m.tag     = tag;
        m.hart_id = 0;
        m.uuid    = s.req.uuid;
        port.send(m);
        pending_mem_[tag] = PendingFill{ uint32_t(&s - &slots_[0]), uint8_t(t) };
        ++s.pending_mem;
        ++perf_stats_.mem_reads;
        any_issued = true;
      }
      if (all_done && s.pending_mem == 0) {
        s.state = State::COMPUTE;
      } else if (any_issued) {
        s.state = State::AWAIT;
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
          std::memcpy(l.line_data.data(), rsp.data->data(),
                      VX_CFG_MEM_BLOCK_SIZE);
        }
        l.line_filled = true;
        if (s.pending_mem > 0) --s.pending_mem;
        // Transition to compute when all lines arrived.
        if (s.pending_mem == 0 && s.state == State::AWAIT) {
          s.state = State::COMPUTE;
        }
        ch.pop();
      }
    }
  }

  void compute_intersections() {
    for (auto& s : slots_) {
      if (!s.in_use || s.state != State::COMPUTE) continue;
      bool any_cb_pending = false;
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        LaneState& l = s.lanes[t];
        if (!l.active) continue;
        const uint8_t* base = l.line_data.data() + l.line_byte_off;
        uint32_t triangle_count = 0;
        std::memcpy(&triangle_count, base, sizeof(uint32_t));
        if (triangle_count == 0) {
          l.hit = false;
          continue;
        }
        const uint8_t* tri_bytes = base + 16;  // skip header
        uint32_t n_tris = std::min(triangle_count, kPhase1MaxTris);
        float best_t = s.req.tmax[t];
        float best_u = 0.f;
        float best_v = 0.f;
        uint32_t best_prim = 0;
        bool any_hit = false;
        bool yield_pending = false;
        float yield_t = 0.f, yield_u = 0.f, yield_v = 0.f;
        uint32_t yield_prim = 0;
        float ro[3] = { s.req.origin_x[t], s.req.origin_y[t], s.req.origin_z[t] };
        float rd[3] = { s.req.dir_x[t],    s.req.dir_y[t],    s.req.dir_z[t]   };
        // Phase 2: a per-tri flags word lives at the tail of each
        // triangle. Bit 0 = OPAQUE; clear → AHS yield instead of immediate
        // commit. Smoke-test path assumes opaque triangles produce the
        // closest hit directly and non-opaque triangles always yield once.
        for (uint32_t i = 0; i < n_tris; ++i) {
          const float* tri = reinterpret_cast<const float*>(tri_bytes + i * kPhase2TriStride);
          uint32_t tri_flags = 0;
          std::memcpy(&tri_flags,
                      tri_bytes + i * kPhase2TriStride + kPhase2TriFlagsOff,
                      sizeof(uint32_t));
          float t_hit = 0.f, u = 0.f, v = 0.f;
          if (ray_triangle(ro, rd, &tri[0], &tri[3], &tri[6],
                           s.req.tmin[t], best_t, t_hit, u, v)) {
            if (t_hit < best_t) {
              if (tri_flags & kPhase2TriFlagOpaque) {
                best_t = t_hit; best_u = u; best_v = v; best_prim = i;
                any_hit = true;
              } else {
                // Stash the candidate, mark the slot for callback yield.
                yield_pending = true;
                yield_t = t_hit; yield_u = u; yield_v = v; yield_prim = i;
                // Don't commit; the dispatcher decides via cb_ret.
                break;  // single-yield Phase 2 — defer further tris to 2.x
              }
            }
          }
        }
        l.hit       = any_hit;
        l.hit_t     = best_t;
        l.hit_u     = best_u;
        l.hit_v     = best_v;
        l.hit_prim  = best_prim;
        if (yield_pending) {
          l.cb_pending = true;
          l.cb_type    = VX_RT_CB_TYPE_ANYHIT;
          l.cand_t     = yield_t;
          l.cand_u     = yield_u;
          l.cand_v     = yield_v;
          l.cand_prim  = yield_prim;
          any_cb_pending = true;
        }
      }
      s.state = any_cb_pending ? State::AWAIT_CALLBACK : State::RESP;
    }
  }

  void emit_completions() {
    if (simobject_->rtu_rsp_out.empty()) return;
    auto& port = simobject_->rtu_rsp_out.at(0);
    for (auto& s : slots_) {
      if (!s.in_use) continue;
      if (s.state == State::AWAIT_CALLBACK) {
        // Phase 2: emit a CB_YIELD rsp carrying the candidate-hit attrs
        // for each yielded lane, then suppress re-emit by flipping the
        // slot-wide cb_emitted flag. Per-lane cb_pending is left alone
        // so the CB_ACTION drain (running on a later tick) knows which
        // lanes participated. CB_ACTION will transition us to RESP.
        if (s.cb_emitted) continue;
        bool any_yield = false;
        for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
          if (s.lanes[t].cb_pending) { any_yield = true; break; }
        }
        if (!any_yield) continue;  // defensive: shouldn't reach here
        if (port.full()) break;
        RtuRsp rsp(s.req);
        rsp.kind = RtuRspKind::CB_YIELD;
        uint32_t cb_mask = 0;
        for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
          const LaneState& l = s.lanes[t];
          if (!l.cb_pending) continue;
          cb_mask |= (1u << t);
          rsp.cb_type[t]            = l.cb_type;
          rsp.hit_t[t]              = l.cand_t;
          rsp.hit_bary_u[t]         = l.cand_u;
          rsp.hit_bary_v[t]         = l.cand_v;
          rsp.hit_primitive_id[t]   = l.cand_prim;
        }
        rsp.cb_active_mask = cb_mask;
        port.send(rsp);
        DT(3, "rtu-core cb_yield: tag=" << s.req.tag
              << ", cb_mask=0x" << std::hex << cb_mask << std::dec);
        s.cb_emitted = true;
        continue;
      }
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
  struct PendingFill { uint32_t slot_idx; uint8_t lane; };

  RtuCore* simobject_;
  std::vector<Slot> slots_;
  std::unordered_map<uint32_t, PendingFill> pending_mem_;
  uint32_t next_tag_ = 0;
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
