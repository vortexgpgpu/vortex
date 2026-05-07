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

#include "tex_core.h"
#include <array>
#include <cstring>
#include <deque>
#include <unordered_map>
#include <vector>
#include <graphics.h>
#include "cluster.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;

namespace {

// Cluster-local TEX-core fan-in (cluster TexBus arb collapses sockets → 1).
// Mirrors `NUM_TEX_CORES` in VX_gpu_pkg / VX_config.
constexpr uint32_t kNumTexCores = 1;

// tcache request ports exposed to the L2 path. Mirrors `TCACHE_NUM_REQS`.
constexpr uint32_t kTcacheNumReqs = 1;

// tcache cache-line size (matches `TCACHE_LINE_SIZE = L1_LINE_SIZE`).
constexpr uint32_t kTcacheLineSize = MEM_BLOCK_SIZE;
constexpr uint64_t kTcacheLineMask = ~uint64_t(MEM_BLOCK_SIZE - 1);

// Per-request slot count. Mirrors `TEX_REQ_QUEUE_SIZE` from VX_config (a
// rounding helper based on NUM_THREADS / NUM_SFU_LANES; 4 is the upper
// bound at this config and well-above what the tex_smoke kernel issues).
constexpr uint32_t kInflight = 8;

} // namespace

// ════════════════════════════════════════════════════════════════════
// TexCore::Impl
// ════════════════════════════════════════════════════════════════════
//
// Reverse-pipeline tick order (matches DXA's tick scheme). Each accepted
// TexReq flows through four functional stages mirroring the RTL:
//
//   tex_arb (drain inputs)
//      └→ tex_addr  (compute per-lane texel addresses + filter params)
//           └→ tex_mem   (issue MemReq per unique cache-line, gather rsps)
//                └→ tex_sampler (apply_filter once all corners arrived)
//                     └→ TexRsp on tex_rsp_out

class TexCore::Impl {
public:
  enum class State : uint8_t { ADDR, MEM, SAMPLE, RESP };

  // Per-lane sample state.
  struct LaneState {
    bool                       active   = false;
    graphics::TexelRequest     trq;             // pure addr/format/filter description
    std::array<uint32_t, 4>    texels   = {};   // raw 32b words from cache
    std::array<bool,     4>    filled   = { false, false, false, false };
    uint32_t                   needed   = 0;    // 1 (POINT) or 4 (BILINEAR)
    uint32_t                   filtered = 0;    // result after apply_filter
  };

  // Per-inflight slot.
  struct Slot {
    bool                                       in_use = false;
    State                                      state  = State::ADDR;
    TexReq                                     req;
    std::array<LaneState, NUM_THREADS>         lanes  = {};
    uint32_t                                   pending_lines = 0; // outstanding MemReqs
    uint64_t                                   issue_cycle  = 0;
  };

  explicit Impl(TexCore* simobject)
    : simobject_(simobject)
    , slots_(kInflight)
    , cycle_(0)
  {}

  void reset() {
    cycle_ = 0;
    perf_stats_ = TexCore::PerfStats();
    for (auto& s : slots_) {
      s.in_use = false;
      s.pending_lines = 0;
      for (auto& l : s.lanes) {
        l.active = false;
        l.filled = { false, false, false, false };
      }
    }
    pending_mem_.clear();
    next_mem_tag_ = 0;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    sampler_dcrs_.write(addr, value);
    sampler_.configure(sampler_dcrs_);
    return 0;
  }

  const TexCore::PerfStats& perf_stats() const { return perf_stats_; }

  void tick() {
    ++cycle_;

    // 1) Drain memory responses → fill per-lane corner buffers.
    drain_mem_rsp();

    // 2) Per-slot state advance.
    for (auto& s : slots_) {
      if (!s.in_use) continue;
      switch (s.state) {
      case State::ADDR:   advance_addr(s);   break;
      case State::MEM:    advance_mem(s);    break;
      case State::SAMPLE: advance_sample(s); break;
      case State::RESP:   advance_resp(s);   break;
      }
    }

    // 3) Drain incoming TexReqs into a free slot (round-robin across the
    //    NUM_TEX_CORES inputs — currently 1).
    drain_req_in();

    // perf: stall when any slot is waiting on cache rsp this cycle
    bool stalled = false;
    for (auto& s : slots_) {
      if (s.in_use && s.state == State::MEM && s.pending_lines > 0) {
        stalled = true;
        break;
      }
    }
    if (stalled) ++perf_stats_.stall_cycles;

    // perf: outstanding-read latency accumulates per inflight cache req
    perf_stats_.mem_latency += pending_mem_.size();
  }

private:
  // ── Stage: tex_arb — drain TexReqs into free slots ──────────────────
  void drain_req_in() {
    for (auto& ch : simobject_->tex_req_in) {
      if (ch.empty()) continue;
      uint32_t free_slot = UINT32_MAX;
      for (uint32_t i = 0; i < slots_.size(); ++i) {
        if (!slots_[i].in_use) { free_slot = i; break; }
      }
      if (free_slot == UINT32_MAX) break; // no slot
      auto& s = slots_[free_slot];
      s.in_use       = true;
      s.state        = State::ADDR;
      s.req          = ch.peek();
      s.issue_cycle  = cycle_;
      s.pending_lines = 0;
      for (auto& l : s.lanes) {
        l.active = false;
        l.filled = { false, false, false, false };
      }
      ch.pop();
      DT(4, simobject_->name() << " accept: uuid=" << s.req.uuid
         << ", stage=" << s.req.stage << ", slot=" << free_slot);
    }
  }

  // ── Stage: tex_addr — compute per-lane TexelRequest ─────────────────
  void advance_addr(Slot& s) {
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
      LaneState& l = s.lanes[t];
      if (!(s.req.tmask_bits & (1u << t))) {
        l.active = false;
        continue;
      }
      l.active   = true;
      l.trq      = sampler_.compute_request(s.req.stage, s.req.u[t], s.req.v[t], s.req.lod[t]);
      l.needed   = (l.trq.filter == VX_TEX_FILTER_BILINEAR) ? 4u : 1u;
      l.filled   = { false, false, false, false };
      l.filtered = 0;
    }
    s.state = State::MEM;
  }

  // ── Stage: tex_mem — issue MemReqs for missing corners ──────────────
  // Issues at most `kTcacheNumReqs` cache requests per tick (mirrors RTL
  // bandwidth limit). Tag layout: high bits = slot id, low bits = (lane,
  // corner) — recovered on response by `pending_mem_`.
  void advance_mem(Slot& s) {
    // Kick off at most kTcacheNumReqs new reads this cycle for this slot.
    uint32_t budget = kTcacheNumReqs;
    bool all_filled = true;

    for (uint32_t t = 0; t < NUM_THREADS && budget > 0; ++t) {
      LaneState& l = s.lanes[t];
      if (!l.active) continue;
      for (uint32_t c = 0; c < l.needed; ++c) {
        if (l.filled[c]) continue;
        all_filled = false;
        // Try to issue MemReq for the cache line containing addr[c].
        auto& req_ch = simobject_->tcache_req_out.at(0);
        if (req_ch.full()) {
          break;
        }

        uint64_t byte_addr = l.trq.addr[c];
        uint64_t cl_addr   = byte_addr & kTcacheLineMask;

        MemReq mreq;
        mreq.addr  = cl_addr;
        mreq.op    = MemOp::LD;
        mreq.tag   = next_mem_tag_++;
        mreq.hart_id   = 0;
        mreq.uuid  = s.req.uuid;

        // Track which (slot, lane, corner) this tag fills.
        PendingFill pf;
        pf.slot   = (uint32_t)(&s - &slots_[0]);
        pf.lane   = uint8_t(t);
        pf.corner = uint8_t(c);
        pf.byte_off = uint32_t(byte_addr - cl_addr);
        pf.stride = uint8_t(l.trq.stride);
        pending_mem_[mreq.tag] = pf;

        req_ch.send(mreq);
        ++s.pending_lines;
        ++perf_stats_.mem_reads;
        --budget;
        if (budget == 0) break;
      }
    }

    if (all_filled && s.pending_lines == 0) {
      s.state = State::SAMPLE;
    }
  }

  // ── Drain memory responses; deposit bytes into per-lane corner ──────
  void drain_mem_rsp() {
    for (auto& ch : simobject_->tcache_rsp_in) {
      while (!ch.empty()) {
        auto& rsp = ch.peek();
        auto it = pending_mem_.find(uint32_t(rsp.tag));
        if (it == pending_mem_.end()) {
          // Stale / unknown tag: drop.
          ch.pop();
          continue;
        }
        const PendingFill pf = it->second;
        pending_mem_.erase(it);

        Slot& s = slots_[pf.slot];
        LaneState& l = s.lanes[pf.lane];

        if (rsp.data) {
          // Extract `stride` bytes at byte_off → uint32_t (zero-extended).
          uint32_t v = 0;
          const uint8_t* src = rsp.data->data() + pf.byte_off;
          uint32_t n = std::min<uint32_t>(pf.stride, sizeof(uint32_t));
          std::memcpy(&v, src, n);
          l.texels[pf.corner] = v;
        } else {
          l.texels[pf.corner] = 0;
        }
        l.filled[pf.corner] = true;
        if (s.pending_lines > 0) --s.pending_lines;
        ch.pop();
      }
    }
  }

  // ── Stage: tex_sampler — apply_filter per active lane ───────────────
  bool slot_all_filled(const Slot& s) const {
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
      const LaneState& l = s.lanes[t];
      if (!l.active) continue;
      for (uint32_t c = 0; c < l.needed; ++c) {
        if (!l.filled[c]) return false;
      }
    }
    return true;
  }

  void advance_sample(Slot& s) {
    if (!slot_all_filled(s)) return;
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
      LaneState& l = s.lanes[t];
      if (!l.active) continue;
      l.filtered = graphics::TextureSampler::apply_filter(l.trq, l.texels.data());
    }
    s.state = State::RESP;
  }

  // ── Stage: response — package texels into a TexRsp ──────────────────
  void advance_resp(Slot& s) {
    auto& rsp_ch = simobject_->tex_rsp_out.at(0);
    if (rsp_ch.full()) return;

    TexRsp rsp;
    rsp.uuid     = s.req.uuid;
    rsp.tag      = s.req.tag;
    rsp.trace    = s.req.trace;
    rsp.block_id = s.req.block_id;
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
      rsp.texels[t] = (s.req.tmask_bits & (1u << t)) ? s.lanes[t].filtered : 0u;
    }
    rsp_ch.send(rsp);
    DT(4, simobject_->name() << " complete: uuid=" << s.req.uuid
       << ", latency=" << (cycle_ - s.issue_cycle));

    // Free slot.
    s.in_use = false;
    s.state  = State::ADDR;
    s.pending_lines = 0;
    for (auto& l : s.lanes) {
      l.active = false;
      l.filled = { false, false, false, false };
    }
  }

  // ── Members ──────────────────────────────────────────────────────────
  struct PendingFill {
    uint32_t slot;
    uint8_t  lane;
    uint8_t  corner;
    uint8_t  stride;
    uint32_t byte_off;
  };

  TexCore*                                  simobject_;
  graphics::TexDCRS                         sampler_dcrs_;
  graphics::TextureSampler                  sampler_{nullptr, nullptr};
  std::vector<Slot>                         slots_;
  std::unordered_map<uint32_t, PendingFill> pending_mem_;
  uint32_t                                  next_mem_tag_ = 0;
  uint64_t                                  cycle_;
  TexCore::PerfStats                        perf_stats_;
};

// ════════════════════════════════════════════════════════════════════
// TexCore — wrappers
// ════════════════════════════════════════════════════════════════════

TexCore::TexCore(const SimContext& ctx, const char* name, Cluster* cluster)
  : SimObject<TexCore>(ctx, name)
  , tex_req_in(kNumTexCores, this)
  , tex_rsp_out(kNumTexCores, this)
  , tcache_req_out(kTcacheNumReqs, this)
  , tcache_rsp_in(kTcacheNumReqs, this)
{
  __unused(cluster);
  impl_ = new Impl(this);
}

TexCore::~TexCore() {
  delete impl_;
}

void TexCore::on_reset() { impl_->reset(); }
void TexCore::on_tick()  { impl_->tick(); }

int TexCore::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

const TexCore::PerfStats& TexCore::perf_stats() const {
  return impl_->perf_stats();
}
