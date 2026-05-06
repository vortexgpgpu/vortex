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

#include "om_core.h"
#include <array>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <graphics.h>
#include "cluster.h"
#include "mem_block_pool.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;

namespace {

// Mirrors RTL `OCACHE_*` (default config). See
// hw/rtl/VX_gpu_pkg.sv:1089-1101 and VX_config.toml [ocache].
constexpr uint32_t kOcacheNumReqs  = OCACHE_NUM_BANKS;
constexpr uint32_t kOcacheMemPorts = 1;
constexpr uint32_t kOcacheLineSize = MEM_BLOCK_SIZE;
constexpr uint64_t kOcacheLineMask = ~uint64_t(MEM_BLOCK_SIZE - 1);

// Inflight OmReq slot count. Mirrors `OM_MEM_QUEUE_SIZE`.
constexpr uint32_t kInflight = OM_MEM_QUEUE_SIZE;

// Cores per cluster.
constexpr uint32_t kCoresPerCluster = NUM_SOCKETS * SOCKET_SIZE;

} // namespace

// ════════════════════════════════════════════════════════════════════
// OmCore::Impl
// ════════════════════════════════════════════════════════════════════

class OmCore::Impl {
public:
  // Per-request lifecycle (mirrors the staged dataflow inside VX_om_core:
  // mem_read → ds/blend compute → mem_write).
  enum class State : uint8_t {
    ADDR,         // compute per-lane addresses + read/write enables
    READ_ISSUE,   // issue MemReq{READ} for needed (lane, port)
    READ_GATHER,  // wait for MemRsps; capture dst_color / dst_depthstencil
    COMPUTE,      // run DS test + blend; decide what to write
    WRITE_ISSUE,  // issue MemReq{WRITE} per lane port that needs it
    DONE,         // free slot
  };

  // Per-thread lane state.
  struct LaneState {
    bool      active             = false;
    uint32_t  pos_x              = 0;
    uint32_t  pos_y              = 0;
    bool      face               = false;
    uint32_t  src_color          = 0;
    uint32_t  src_depth          = 0;

    uint64_t  zbuf_addr_byte     = 0;
    uint64_t  cbuf_addr_byte     = 0;

    bool      need_z_read        = false;
    bool      need_c_read        = false;

    bool      z_arrived          = false;
    bool      c_arrived          = false;
    uint32_t  dst_depthstencil   = 0;
    uint32_t  dst_color          = 0;

    bool      ds_pass            = false;
    uint32_t  merged_depthstencil = 0;
    uint32_t  blended_color      = 0;

    bool      need_z_write       = false;
    bool      need_c_write       = false;
    uint32_t  z_write_value      = 0;
    uint32_t  c_write_value      = 0;
    bool      z_write_issued     = false;
    bool      c_write_issued     = false;
    bool      z_read_issued      = false;
    bool      c_read_issued      = false;
  };

  struct Slot {
    bool                                    in_use = false;
    State                                   state  = State::ADDR;
    OmReq                                   req;
    std::array<LaneState, NUM_THREADS>      lanes  = {};
    uint64_t                                issue_cycle = 0;
  };

  // Pending memory request bookkeeping. Each tag identifies a unique
  // (slot, lane, port) triple so the response can be deposited in the right
  // lane buffer. Port: 0 = zbuf (depth+stencil), 1 = cbuf (color).
  struct PendingFill {
    uint32_t slot;
    uint8_t  lane;
    uint8_t  port;        // 0=zbuf, 1=cbuf
    uint32_t byte_off;    // offset within the 64-byte cache line
  };

  explicit Impl(OmCore* simobject)
    : simobject_(simobject)
    , slots_(kInflight)
    , cycle_(0)
  {}

  void reset() {
    cycle_ = 0;
    perf_stats_ = OmCore::PerfStats();
    for (auto& s : slots_) {
      s.in_use = false;
      s.state  = State::ADDR;
      for (auto& l : s.lanes) l = LaneState{};
    }
    pending_mem_.clear();
    next_mem_tag_ = 0;
    rr_req_ = 0;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    dcrs_.write(addr, value);
    depth_stencil_.configure(dcrs_);
    blender_.configure(dcrs_);
    recompute_state();
    return 0;
  }

  const OmCore::PerfStats& perf_stats() const { return perf_stats_; }

  void tick() {
    ++cycle_;

    // 1) Drain ocache responses.
    drain_mem_rsp();

    // 2) Per-slot state advance.
    for (auto& s : slots_) {
      if (!s.in_use) continue;
      switch (s.state) {
      case State::ADDR:        advance_addr(s);        break;
      case State::READ_ISSUE:  advance_read_issue(s);  break;
      case State::READ_GATHER: advance_read_gather(s); break;
      case State::COMPUTE:     advance_compute(s);     break;
      case State::WRITE_ISSUE: advance_write_issue(s); break;
      case State::DONE:        free_slot(s);           break;
      }
    }

    // 3) Drain incoming OmReqs into free slots (round-robin per-core).
    drain_req_in();

    // perf
    bool stalled = false;
    for (auto& s : slots_) {
      if (s.in_use && (s.state == State::READ_GATHER ||
                       s.state == State::READ_ISSUE ||
                       s.state == State::WRITE_ISSUE)) {
        stalled = true;
        break;
      }
    }
    if (stalled) ++perf_stats_.stall_cycles;
    perf_stats_.mem_latency += pending_mem_.size();
  }

private:
  // ── DCR-derived state (cached, recomputed on every dcr_write) ───────
  void recompute_state() {
    zbuf_baseaddr_           = uint64_t(dcrs_.read(VX_DCR_OM_ZBUF_ADDR)) << 6;
    zbuf_pitch_              = dcrs_.read(VX_DCR_OM_ZBUF_PITCH);
    depth_writemask_         = dcrs_.read(VX_DCR_OM_DEPTH_WRITEMASK) & 0x1;
    stencil_front_writemask_ = dcrs_.read(VX_DCR_OM_STENCIL_WRITEMASK) & 0xffff;
    stencil_back_writemask_  = dcrs_.read(VX_DCR_OM_STENCIL_WRITEMASK) >> 16;

    cbuf_baseaddr_ = uint64_t(dcrs_.read(VX_DCR_OM_CBUF_ADDR)) << 6;
    cbuf_pitch_    = dcrs_.read(VX_DCR_OM_CBUF_PITCH);
    uint32_t cbuf_writemask = dcrs_.read(VX_DCR_OM_CBUF_WRITEMASK) & 0xf;
    cbuf_writemask_ = (((cbuf_writemask >> 0) & 0x1) * 0x000000ff)
                    | (((cbuf_writemask >> 1) & 0x1) * 0x0000ff00)
                    | (((cbuf_writemask >> 2) & 0x1) * 0x00ff0000)
                    | (((cbuf_writemask >> 3) & 0x1) * 0xff000000);
    color_read_  = (cbuf_writemask != 0xf);
    color_write_ = (cbuf_writemask != 0x0);
  }

  // ── Stage: ACCEPT (drain per-core inputs into free slots) ───────────
  void drain_req_in() {
    auto& chs = simobject_->om_req_in;
    if (chs.empty()) return;
    for (uint32_t i = 0; i < chs.size(); ++i) {
      uint32_t cid = (rr_req_ + i) % chs.size();
      auto& ch = chs.at(cid);
      if (ch.empty()) continue;

      uint32_t free_slot = UINT32_MAX;
      for (uint32_t s = 0; s < slots_.size(); ++s) {
        if (!slots_[s].in_use) { free_slot = s; break; }
      }
      if (free_slot == UINT32_MAX) break; // no slot

      auto& slot = slots_[free_slot];
      slot.in_use      = true;
      slot.state       = State::ADDR;
      slot.req         = ch.peek();
      slot.issue_cycle = cycle_;
      for (auto& l : slot.lanes) l = LaneState{};
      ch.pop();
      rr_req_ = (cid + 1) % chs.size();
      DT(4, simobject_->name() << " accept: uuid=" << slot.req.uuid
         << ", slot=" << free_slot);
      break;
    }
  }

  // ── Stage: ADDR — compute per-lane addresses + read/write enables ───
  void advance_addr(Slot& s) {
    bool depth_enabled    = depth_stencil_.depth_enabled();
    bool blend_enabled    = blender_.enabled();

    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
      if (!(s.req.tmask_bits & (1u << t))) {
        s.lanes[t].active = false;
        continue;
      }
      LaneState& l = s.lanes[t];
      l.active    = true;
      l.pos_x     = s.req.pos_x[t];
      l.pos_y     = s.req.pos_y[t];
      l.face      = s.req.face[t] != 0;
      l.src_color = s.req.color[t];
      l.src_depth = s.req.depth[t];

      l.zbuf_addr_byte = zbuf_baseaddr_ + uint64_t(l.pos_y) * zbuf_pitch_ + l.pos_x * 4;
      l.cbuf_addr_byte = cbuf_baseaddr_ + uint64_t(l.pos_y) * cbuf_pitch_ + l.pos_x * 4;

      bool stencil_enabled = depth_stencil_.stencil_enabled(l.face);
      l.need_z_read = depth_enabled || stencil_enabled;
      l.need_c_read = color_write_ && (color_read_ || blend_enabled);
    }
    s.state = State::READ_ISSUE;
  }

  // ── Stage: READ_ISSUE — issue MemReq{READ} for needed lane ports ────
  void advance_read_issue(Slot& s) {
    bool any_read_needed = false;
    bool all_issued      = true;

    // Cap MemReq issuance per tick at OCACHE_NUM_REQS (RTL bandwidth).
    uint32_t budget = kOcacheNumReqs;

    for (uint32_t t = 0; t < NUM_THREADS && budget > 0; ++t) {
      LaneState& l = s.lanes[t];
      if (!l.active) continue;

      if (l.need_z_read && !l.z_read_issued) {
        any_read_needed = true;
        if (try_issue_read(s, t, /*port=*/0, l.zbuf_addr_byte)) {
          l.z_read_issued = true;
          if (--budget == 0) break;
        } else {
          all_issued = false;
          break;
        }
      }
      if (l.need_c_read && !l.c_read_issued) {
        any_read_needed = true;
        if (try_issue_read(s, t, /*port=*/1, l.cbuf_addr_byte)) {
          l.c_read_issued = true;
          if (--budget == 0) break;
        } else {
          all_issued = false;
          break;
        }
      }
    }
    // Detect "is anything still un-issued?"
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
      LaneState& l = s.lanes[t];
      if (!l.active) continue;
      if (l.need_z_read && !l.z_read_issued) { all_issued = false; }
      if (l.need_c_read && !l.c_read_issued) { all_issued = false; }
    }

    if (!any_read_needed) {
      // No reads needed — color-write-only bypass path; jump to COMPUTE.
      s.state = State::COMPUTE;
    } else if (all_issued) {
      s.state = State::READ_GATHER;
    }
  }

  // ── Try to issue a read MemReq for (slot, lane, port). ──────────────
  // Returns true if sent, false if backpressured.
  bool try_issue_read(Slot& s, uint32_t lane, uint8_t port, uint64_t byte_addr) {
    auto& req_ch = simobject_->ocache_req_out.at(0);
    if (req_ch.full()) return false;

    uint64_t cl_addr = byte_addr & kOcacheLineMask;
    uint32_t off     = uint32_t(byte_addr - cl_addr);

    MemReq mreq;
    mreq.addr  = cl_addr;
    mreq.write = false;
    mreq.op    = MemOp::READ;
    mreq.type  = AddrType::Global;
    mreq.tag   = next_mem_tag_++;
    mreq.cid   = s.req.tag;
    mreq.uuid  = s.req.uuid;

    PendingFill pf;
    pf.slot     = (uint32_t)(&s - &slots_[0]);
    pf.lane     = uint8_t(lane);
    pf.port     = port;
    pf.byte_off = off;
    pending_mem_[mreq.tag] = pf;

    req_ch.send(mreq);
    ++perf_stats_.mem_reads;
    return true;
  }

  // ── Drain ocache responses → fill per-lane buffers ──────────────────
  void drain_mem_rsp() {
    for (auto& ch : simobject_->ocache_rsp_in) {
      while (!ch.empty()) {
        auto& rsp = ch.peek();
        auto it = pending_mem_.find(uint32_t(rsp.tag));
        if (it == pending_mem_.end()) {
          ch.pop();
          continue;
        }
        const PendingFill pf = it->second;
        pending_mem_.erase(it);

        Slot& s = slots_[pf.slot];
        LaneState& l = s.lanes[pf.lane];

        uint32_t v = 0;
        if (rsp.data) {
          std::memcpy(&v, rsp.data->data() + pf.byte_off, 4);
        }
        if (pf.port == 0) {
          l.dst_depthstencil = v;
          l.z_arrived        = true;
        } else {
          l.dst_color = v;
          l.c_arrived = true;
        }
        ch.pop();
      }
    }
  }

  // ── Stage: READ_GATHER — wait for all needed responses ──────────────
  void advance_read_gather(Slot& s) {
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
      const LaneState& l = s.lanes[t];
      if (!l.active) continue;
      if (l.need_z_read && !l.z_arrived) return;
      if (l.need_c_read && !l.c_arrived) return;
    }
    s.state = State::COMPUTE;
  }

  // ── Stage: COMPUTE — run DS test + blend, decide writes ─────────────
  void advance_compute(Slot& s) {
    bool depth_enabled = depth_stencil_.depth_enabled();
    bool blend_enabled = blender_.enabled();

    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
      LaneState& l = s.lanes[t];
      if (!l.active) continue;

      bool stencil_enabled = depth_stencil_.stencil_enabled(l.face);
      bool ds_active = depth_enabled || stencil_enabled;

      uint32_t merged = 0;
      l.ds_pass = !ds_active
                || depth_stencil_.test(l.face, l.src_depth, l.dst_depthstencil, &merged);
      l.merged_depthstencil = merged;

      l.blended_color = (blend_enabled && l.ds_pass)
                      ? blender_.blend(l.src_color, l.dst_color)
                      : l.src_color;

      // Decide writes (mirrors VX_om_core gating).
      uint32_t stencil_writemask = l.face ? stencil_back_writemask_ : stencil_front_writemask_;
      uint32_t ds_writemask =
          ((depth_enabled && l.ds_pass && depth_writemask_) ? VX_OM_DEPTH_MASK : 0u)
        | (stencil_enabled ? (uint32_t(stencil_writemask) << VX_OM_DEPTH_BITS) : 0u);

      l.need_z_write = (ds_writemask != 0);
      if (l.need_z_write) {
        l.z_write_value = (l.dst_depthstencil & ~ds_writemask)
                        | (l.merged_depthstencil & ds_writemask);
      }

      l.need_c_write = color_write_ && l.ds_pass;
      if (l.need_c_write) {
        // If color_read_ is false (writemask == 0xf), dst_color is unread —
        // we'll still write the full word; the merge is a no-op.
        l.c_write_value = (l.dst_color & ~cbuf_writemask_)
                        | (l.blended_color & cbuf_writemask_);
      }
    }
    s.state = State::WRITE_ISSUE;
  }

  // ── Stage: WRITE_ISSUE — emit per-lane writes (rate-limited) ────────
  void advance_write_issue(Slot& s) {
    bool any_pending = false;
    bool all_done    = true;

    uint32_t budget = kOcacheNumReqs;

    for (uint32_t t = 0; t < NUM_THREADS && budget > 0; ++t) {
      LaneState& l = s.lanes[t];
      if (!l.active) continue;

      if (l.need_z_write && !l.z_write_issued) {
        any_pending = true;
        if (try_issue_write(s, l.zbuf_addr_byte, l.z_write_value)) {
          l.z_write_issued = true;
          if (--budget == 0) break;
        } else {
          all_done = false;
          break;
        }
      }
      if (l.need_c_write && !l.c_write_issued) {
        any_pending = true;
        if (try_issue_write(s, l.cbuf_addr_byte, l.c_write_value)) {
          l.c_write_issued = true;
          if (--budget == 0) break;
        } else {
          all_done = false;
          break;
        }
      }
    }

    // Re-scan for un-issued writes.
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
      const LaneState& l = s.lanes[t];
      if (!l.active) continue;
      if (l.need_z_write && !l.z_write_issued) all_done = false;
      if (l.need_c_write && !l.c_write_issued) all_done = false;
    }

    if (!any_pending && all_done) {
      // Nothing to write at all.
      s.state = State::DONE;
    } else if (all_done) {
      s.state = State::DONE;
    }
  }

  // ── Issue a 4-byte write at byte_addr. Software RMW already merged. ─
  bool try_issue_write(Slot& s, uint64_t byte_addr, uint32_t value) {
    auto& req_ch = simobject_->ocache_req_out.at(0);
    if (req_ch.full()) return false;

    uint64_t cl_addr = byte_addr & kOcacheLineMask;
    uint32_t off     = uint32_t(byte_addr - cl_addr);

    MemReq mreq;
    mreq.addr   = cl_addr;
    mreq.write  = true;
    mreq.op     = MemOp::WRITE;
    mreq.type   = AddrType::Global;
    mreq.tag    = next_mem_tag_++;
    mreq.cid    = s.req.tag;
    mreq.uuid   = s.req.uuid;
    mreq.byteen = uint64_t(0xfu) << off;

    auto blk = make_mem_block();
    std::memcpy(blk->data() + off, &value, 4);
    mreq.data   = blk;

    req_ch.send(mreq);
    ++perf_stats_.mem_writes;
    return true;
  }

  void free_slot(Slot& s) {
    DT(4, simobject_->name() << " complete: uuid=" << s.req.uuid
       << ", latency=" << (cycle_ - s.issue_cycle));
    s.in_use = false;
    s.state  = State::ADDR;
    for (auto& l : s.lanes) l = LaneState{};
  }

  // ── Members ──────────────────────────────────────────────────────────
  OmCore*                                   simobject_;
  graphics::OMDCRS                          dcrs_;
  graphics::DepthTencil                     depth_stencil_;
  graphics::Blender                         blender_;

  // DCR-derived cache (matches existing OmCore semantics).
  uint64_t  cbuf_baseaddr_   = 0;
  uint32_t  cbuf_pitch_      = 0;
  uint32_t  cbuf_writemask_  = 0;
  bool      color_read_      = false;
  bool      color_write_     = false;
  uint64_t  zbuf_baseaddr_   = 0;
  uint32_t  zbuf_pitch_      = 0;
  bool      depth_writemask_ = false;
  uint32_t  stencil_front_writemask_ = 0;
  uint32_t  stencil_back_writemask_  = 0;

  std::vector<Slot>                         slots_;
  std::unordered_map<uint32_t, PendingFill> pending_mem_;
  uint32_t                                  next_mem_tag_ = 0;
  uint32_t                                  rr_req_ = 0;
  uint64_t                                  cycle_;
  OmCore::PerfStats                         perf_stats_;
};

// ════════════════════════════════════════════════════════════════════
// OmCore — wrappers
// ════════════════════════════════════════════════════════════════════

OmCore::OmCore(const SimContext& ctx, const char* name, Cluster* cluster)
  : SimObject<OmCore>(ctx, name)
  , om_req_in(kCoresPerCluster, this)
  , ocache_req_out(kOcacheNumReqs, this)
  , ocache_rsp_in(kOcacheNumReqs, this)
{
  __unused(cluster);
  impl_ = new Impl(this);
}

OmCore::~OmCore() {
  delete impl_;
}

void OmCore::on_reset() { impl_->reset(); }
void OmCore::on_tick()  { impl_->tick(); }

int OmCore::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

const OmCore::PerfStats& OmCore::perf_stats() const {
  return impl_->perf_stats();
}
