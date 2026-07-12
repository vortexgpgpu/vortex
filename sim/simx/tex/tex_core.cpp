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
#include "gfx_ff_model.h"
#include "socket.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;

namespace {

// TEX-core fan-in: the socket TexBus arb collapses its cores → 1.
constexpr uint32_t kNumTexCores = 1;

// tcache request ports exposed to the L2 path.
constexpr uint32_t kTcacheNumReqs = 1;

// tcache cache-line size.
constexpr uint32_t kTcacheLineSize = VX_CFG_MEM_BLOCK_SIZE;
constexpr uint64_t kTcacheLineMask = ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);

// Per-request inflight slot count (upper bound for this config).
constexpr uint32_t kInflight = 8;

} // namespace

// ════════════════════════════════════════════════════════════════════
// TexCore::Impl
// ════════════════════════════════════════════════════════════════════
//
// Reverse-pipeline tick order. Each accepted TexReq flows through four
// functional stages:
//
//   tex_arb (drain inputs)
//      └→ tex_addr  (compute per-lane texel addresses + filter params)
//           └→ tex_mem   (issue MemReqs, gather rsps; filter on completion)
//                └→ TexRsp on tex_rsp_out

class TexCore::Impl {
public:
  enum class State : uint8_t { ADDR, MEM, RESP };

  // Per-lane sample state. Trilinear (mip-linear) samples two LODs, so up to
  // 8 taps: trq[0]/texels[0..per_lod) for lod0, trq[1]/texels[per_lod..) for
  // lod1, blended by lod_frac.
  struct LaneState {
    bool                       active   = false;
    TexelRequest               trq[2];          // pure addr/format/filter, per LOD
    std::array<uint32_t, 8>    texels   = {};   // raw 32b words from cache
    std::array<bool,     8>    filled   = {};
    uint32_t                   per_lod  = 0;    // 1 (POINT) or 4 (BILINEAR)
    uint32_t                   nlods    = 1;    // 1, or 2 for trilinear
    uint32_t                   needed   = 0;    // per_lod * nlods
    uint32_t                   lod_frac = 0;    // trilinear blend weight (0..255)
    uint32_t                   filtered = 0;    // result after apply_filter
  };

  // Per-inflight slot.
  struct Slot {
    bool                                       in_use = false;
    State                                      state  = State::ADDR;
    TexReq                                     req;
    std::array<LaneState, VX_CFG_NUM_THREADS>         lanes  = {};
    std::array<bool, 8>                        dups   = {}; // per-corner warp-uniform address
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
        l.filled = {};
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
      case State::ADDR: advance_addr(s); break;
      case State::MEM:  advance_mem(s);  break;
      case State::RESP: advance_resp(s); break;
      }
    }

    // 3) Drain incoming TexReqs into a free slot (round-robin across the
    //    VX_CFG_NUM_TEX_CORES inputs — currently 1).
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
        l.filled = {};
      }
      ch.pop();
      DT(4, simobject_->name() << " accept: uuid=" << s.req.uuid
         << ", stage=" << s.req.stage << ", slot=" << free_slot);
    }
  }

  // ── Stage: tex_addr — compute per-lane TexelRequest ─────────────────
  void advance_addr(Slot& s) {
    const bool tri = sampler_.mip_linear(s.req.stage);
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      LaneState& l = s.lanes[t];
      if (!(s.req.tmask_bits & (1u << t))) {
        l.active = false;
        continue;
      }
      l.active = true;
      const uint32_t lod  = s.req.lod[t];
      const uint32_t lod0 = tri ? (lod >> VX_TEX_LOD_FRAC_BITS) : lod;
      l.trq[0]   = sampler_.compute_request(s.req.stage, s.req.u[t], s.req.v[t], lod0);
      l.per_lod  = (l.trq[0].filter == VX_TEX_FILTER_BILINEAR) ? 4u : 1u;
      if (tri) {
        const uint32_t lod1 = (lod0 + 1 < (uint32_t)VX_TEX_LOD_MAX) ? lod0 + 1
                                                                    : (uint32_t)VX_TEX_LOD_MAX;
        l.trq[1]   = sampler_.compute_request(s.req.stage, s.req.u[t], s.req.v[t], lod1);
        l.nlods    = 2;
        l.lod_frac = lod & ((1u << VX_TEX_LOD_FRAC_BITS) - 1);
      } else {
        l.nlods    = 1;
        l.lod_frac = 0;
      }
      l.needed   = l.per_lod * l.nlods;
      l.filled   = {};
      l.filtered = 0;
    }
    this->compute_dups(s);
    s.state = State::MEM;
  }

  // Per-corner byte address of a lane's tap `c`.
  static uint64_t corner_addr(const LaneState& l, uint32_t c) {
    const TexelRequest& trq = l.trq[c < l.per_lod ? 0 : 1];
    return trq.addr[c < l.per_lod ? c : c - l.per_lod];
  }

  // Duplicate-address squash: when every active lane addresses the same
  // texel for a corner, only lane 0 fetches it and the response fans out
  // to all lanes. Anchored on lane 0 being active.
  void compute_dups(Slot& s) {
    s.dups = {};
    const LaneState& l0 = s.lanes[0];
    if (!l0.active) {
      return;
    }
    for (uint32_t c = 0; c < l0.needed; ++c) {
      uint64_t a0 = corner_addr(l0, c);
      bool dup = true;
      for (uint32_t t = 1; t < VX_CFG_NUM_THREADS; ++t) {
        const LaneState& l = s.lanes[t];
        if (!l.active) {
          continue;
        }
        if (corner_addr(l, c) != a0) {
          dup = false;
          break;
        }
      }
      s.dups[c] = dup;
    }
  }

  // ── Stage: tex_mem — issue MemReqs for missing corners ──────────────
  // Issues at most `kTcacheNumReqs` cache requests per tick.
  // Tag layout: high bits = slot id, low bits = (lane, corner) —
  // recovered on response by `pending_mem_`.
  void advance_mem(Slot& s) {
    // Kick off at most kTcacheNumReqs new reads this cycle for this slot.
    uint32_t budget = kTcacheNumReqs;
    bool all_filled = true;

    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS && budget > 0; ++t) {
      LaneState& l = s.lanes[t];
      if (!l.active) continue;
      for (uint32_t c = 0; c < l.needed; ++c) {
        if (l.filled[c]) continue;
        // Squashed duplicate: lane 0's fetch fills this corner on response.
        if (s.dups[c] && t != 0) {
          continue;
        }
        all_filled = false;
        // Try to issue MemReq for the cache line containing addr[c].
        auto& req_ch = simobject_->tcache_req_out.at(0);
        if (req_ch.full()) {
          break;
        }

        // taps 0..per_lod-1 belong to lod0 (trq[0]); per_lod..2*per_lod-1 to lod1.
        uint64_t byte_addr = corner_addr(l, c);
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
        pf.dup    = s.dups[c];
        pf.byte_off = uint32_t(byte_addr - cl_addr);
        pf.stride = uint8_t(l.trq[0].stride);
        pending_mem_[mreq.tag] = pf;

        req_ch.send(mreq);
        ++s.pending_lines;
        ++perf_stats_.mem_reads;
        --budget;
        if (budget == 0) break;
      }
    }

    if (all_filled && s.pending_lines == 0) {
      // Corners complete — filter immediately (the completed batch feeds
      // the sampler combinationally); the response send is the registered
      // stage.
      this->sample_slot(s);
      s.state = State::RESP;
    }
  }

  // ── Drain memory responses; deposit bytes into per-lane corner ──────
  void drain_mem_rsp() {
    // One response per port per cycle (each channel in tcache_rsp_in is a port).
    for (auto& ch : simobject_->tcache_rsp_in) {
      if (ch.empty()) continue;
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

      uint32_t v = 0;
      if (rsp.data) {
        // Extract `stride` bytes at byte_off → uint32_t (zero-extended).
        const uint8_t* src = rsp.data->data() + pf.byte_off;
        uint32_t n = std::min<uint32_t>(pf.stride, sizeof(uint32_t));
        std::memcpy(&v, src, n);
      }
      if (pf.dup) {
        // Squashed duplicate: fan the texel out to every active lane.
        for (auto& l : s.lanes) {
          if (!l.active) continue;
          l.texels[pf.corner] = v;
          l.filled[pf.corner] = true;
        }
      } else {
        LaneState& l = s.lanes[pf.lane];
        l.texels[pf.corner] = v;
        l.filled[pf.corner] = true;
      }
      if (s.pending_lines > 0) --s.pending_lines;
      ch.pop();
    }
  }

  // ── Stage: tex_sampler — apply_filter per active lane ───────────────
  void sample_slot(Slot& s) {
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      LaneState& l = s.lanes[t];
      if (!l.active) continue;
      l.filtered = TextureSampler::apply_filter(l.trq[0], &l.texels[0]);
      if (l.nlods == 2) {
        uint32_t f1 = TextureSampler::apply_filter(l.trq[1], &l.texels[l.per_lod]);
        l.filtered = TexLodLerp(l.filtered, f1, l.lod_frac);
      }
    }
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
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
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
      l.filled = {};
    }
  }

  // ── Members ──────────────────────────────────────────────────────────
  struct PendingFill {
    uint32_t slot;
    uint8_t  lane;
    uint8_t  corner;
    uint8_t  stride;
    bool     dup;
    uint32_t byte_off;
  };

  TexCore*                                  simobject_;
  TexDCRS                         sampler_dcrs_;
  TextureSampler                  sampler_{nullptr, nullptr};
  std::vector<Slot>                         slots_;
  std::unordered_map<uint32_t, PendingFill> pending_mem_;
  uint32_t                                  next_mem_tag_ = 0;
  uint64_t                                  cycle_;
  TexCore::PerfStats                        perf_stats_;
};

// ════════════════════════════════════════════════════════════════════
// TexCore — wrappers
// ════════════════════════════════════════════════════════════════════

TexCore::TexCore(const SimContext& ctx, const char* name, Socket* socket)
  : SimObject<TexCore>(ctx, name)
  , tex_req_in(kNumTexCores, this)
  , tex_rsp_out(kNumTexCores, this)
  , tcache_req_out(kTcacheNumReqs, this)
  , tcache_rsp_in(kTcacheNumReqs, this)
{
  __unused(socket);
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
