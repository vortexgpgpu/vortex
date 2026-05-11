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

#include "dxa_core.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <vector>
#include "core.h"
#include "cluster.h"
#include "socket.h"
#include "mem_block_pool.h"
#include "debug.h"
#include "constants.h"

using namespace vortex;

namespace {

// Number of GMEM ports DxaCore exposes to L2 (one arb output per port).
constexpr uint32_t kDxaMemPorts = std::min<uint32_t>(NUM_DXA_UNITS, L2_NUM_REQS);

// LMEM "word" granularity for splitting DXA writes. The LocalMem bank
// model applies byteen relative to a MEM_BLOCK_SIZE-aligned address (see
// LocalMem::Impl::tick), so each LineWork's destination must lie within
// one MEM_BLOCK_SIZE-aligned region. RTL DXA_LMEM_WORD_SIZE
// (LMEM_NUM_BANKS * XLENB) is the per-cycle-bandwidth unit; mem_block
// granularity here is the byteen-mask scope.
constexpr uint32_t kLmemWordSize = MEM_BLOCK_SIZE;

// GMEM line size + mask.
constexpr uint32_t kGmemLineSize = L1_LINE_SIZE;
constexpr uint64_t kGmemLineMask = ~uint64_t(L1_LINE_SIZE - 1);

// Cores per cluster.
constexpr uint32_t kCoresPerCluster = NUM_SOCKETS * SOCKET_SIZE;

} // namespace

// ════════════════════════════════════════════════════════════════════
// DxaCore::Impl
// ════════════════════════════════════════════════════════════════════

class DxaCore::Impl {
public:
  // ── Descriptor ──────────────────────────────────────────────────────
  struct Descriptor {
    uint64_t base_addr = 0;
    std::array<uint32_t, 5> sizes = {};
    std::array<uint32_t, 4> strides = {};
    uint32_t meta = 0;
    std::array<uint32_t, 5> element_strides = {};
    std::array<uint16_t, 5> tile_sizes = {};
    uint32_t cfill = 0;
    uint32_t smem_stride = 0;
  };

  // ── Per-line work item produced by addr_gen, consumed by gmem_req ───
  // Each entry = one GMEM cache-line read whose payload contributes one or
  // more bytes to a single LMEM word write.
  struct LineWork {
    uint64_t gmem_cl_addr;     // CL-aligned global address
    uint64_t smem_word_addr;   // word-aligned SMEM byte address
    uint32_t cl_byte_offset;   // start of valid bytes within the CL
    uint32_t smem_byte_offset; // destination offset within the SMEM word
    uint32_t valid_length;     // valid bytes
    uint32_t cfill;            // OOB fill value (lane-replicated)
    bool     oob;              // skip GMEM read; use cfill
    bool     last;             // last work item of the transfer
  };

  // ── Inflight slot for a GMEM read ──────────────────────────────────
  struct InflightSlot {
    bool      allocated = false;
    bool      rsp_arrived = false;
    LineWork  work;            // captured at allocation
    std::shared_ptr<mem_block_t> rsp_data;
  };

  // ── Worker (one per slice) ───────────────────────────────────────────
  enum class WState { IDLE, RUNNING };

  struct Worker {
    WState                  state = WState::IDLE;
    uint32_t                worker_id = 0;
    DxaReq                  req;
    Descriptor              desc;
    bool                    is_multicast = false;
    std::vector<uint32_t>   cta_indices;
    uint32_t                smem_stride = 0;
    uint64_t                issue_cycle = 0;

    // addr_gen: pre-enumerated work list (one entry per GMEM CL +
    // multicast replays handled at smem_wr stage).
    std::vector<LineWork>   work_list;
    uint32_t                ag_idx = 0;          // next entry to issue (gmem_req)

    // gmem_req → rsp_buf → smem_wr inflight bookkeeping.
    std::array<InflightSlot, DXA_MAX_INFLIGHT> inflight;
    std::deque<uint32_t>    issued_order;        // tag order, FIFO drain

    // smem_wr multicast replay state.
    uint32_t                mc_cta_idx = 0;      // 0..cta_indices.size()
    uint32_t                writes_emitted = 0;  // count for perf
  };

  // ── Constructor ──────────────────────────────────────────────────────
  explicit Impl(DxaCore* simobject, MemArbiter* gmem_arb)
    : simobject_(simobject)
    , gmem_arb_(gmem_arb)
    , workers_(NUM_DXA_UNITS)
    , cycle_(0)
  {
    for (uint32_t i = 0; i < NUM_DXA_UNITS; ++i)
      workers_[i].worker_id = i;
  }

  void reset() {
    cycle_ = 0;
    queue_.clear();
    perf_stats_ = DxaCore::PerfStats();
    for (auto& w : workers_) {
      w.state = WState::IDLE;
      w.work_list.clear();
      w.cta_indices.clear();
      w.issued_order.clear();
      for (auto& s : w.inflight) { s.allocated = false; s.rsp_arrived = false; s.rsp_data.reset(); }
      w.ag_idx = 0;
      w.mc_cta_idx = 0;
      w.writes_emitted = 0;
    }
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    uint32_t slot = VX_DCR_DXA_DESC_SLOT(addr);
    uint32_t word = VX_DCR_DXA_DESC_WORD(addr);
    auto& d = descriptors_.at(slot);
    switch (word) {
    case VX_DCR_DXA_DESC_BASE_LO_OFF:   d.base_addr = (d.base_addr & 0xffffffff00000000ull) | value; break;
    case VX_DCR_DXA_DESC_BASE_HI_OFF:   d.base_addr = (d.base_addr & 0x00000000ffffffffull) | (uint64_t(value) << 32); break;
    case VX_DCR_DXA_DESC_SIZE0_OFF:     d.sizes[0] = value; break;
    case VX_DCR_DXA_DESC_SIZE1_OFF:     d.sizes[1] = value; break;
    case VX_DCR_DXA_DESC_SIZE2_OFF:     d.sizes[2] = value; break;
    case VX_DCR_DXA_DESC_SIZE3_OFF:     d.sizes[3] = value; break;
    case VX_DCR_DXA_DESC_SIZE4_OFF:     d.sizes[4] = value; break;
    case VX_DCR_DXA_DESC_STRIDE0_OFF:   d.strides[0] = value; break;
    case VX_DCR_DXA_DESC_STRIDE1_OFF:   d.strides[1] = value; break;
    case VX_DCR_DXA_DESC_STRIDE2_OFF:   d.strides[2] = value; break;
    case VX_DCR_DXA_DESC_STRIDE3_OFF:   d.strides[3] = value; break;
    case VX_DCR_DXA_DESC_META_OFF:      d.meta = value; break;
    case VX_DCR_DXA_DESC_ESTRIDE0_OFF:  d.element_strides[0] = value; break;
    case VX_DCR_DXA_DESC_ESTRIDE1_OFF:  d.element_strides[1] = value; break;
    case VX_DCR_DXA_DESC_ESTRIDE2_OFF:  d.element_strides[2] = value; break;
    case VX_DCR_DXA_DESC_ESTRIDE3_OFF:  d.element_strides[3] = value; break;
    case VX_DCR_DXA_DESC_ESTRIDE4_OFF:  d.element_strides[4] = value; break;
    case VX_DCR_DXA_DESC_TILESIZE01_OFF:
      d.tile_sizes[0] = uint16_t(value & 0xffff);
      d.tile_sizes[1] = uint16_t(value >> 16);
      break;
    case VX_DCR_DXA_DESC_TILESIZE23_OFF:
      d.tile_sizes[2] = uint16_t(value & 0xffff);
      d.tile_sizes[3] = uint16_t(value >> 16);
      break;
    case VX_DCR_DXA_DESC_TILESIZE4_OFF: d.tile_sizes[4] = uint16_t(value & 0xffff); break;
    case VX_DCR_DXA_DESC_CFILL_OFF:     d.cfill = value; break;
    case VX_DCR_DXA_DESC_SMEM_STRIDE_OFF: d.smem_stride = value; break;
    default: break;
    }
    return 0;
  }

  void tick() {
    ++cycle_;

    // Drain in reverse-pipeline order (smem_wr → rsp_buf → gmem_req →
    // addr_gen → setup → req-arb) so each stage's tick reads what its
    // upstream produced earlier this cycle, mimicking forward stage flow.

    // 1) Drain GMEM responses → mark inflight slots arrived.
    // gmem_arb_ already routes responses back to RspOut[worker_id] (its
    // input index) and restores the user's tag (the per-worker slot id).
    for (uint32_t worker_id = 0; worker_id < workers_.size(); ++worker_id) {
      auto& ch = gmem_arb_->RspOut.at(worker_id);
      while (!ch.empty()) {
        auto& rsp = ch.peek();
        uint32_t slot_id = uint32_t(rsp.tag) & (DXA_MAX_INFLIGHT - 1);
        auto& w = workers_[worker_id];
        if (slot_id < w.inflight.size() && w.inflight[slot_id].allocated) {
          w.inflight[slot_id].rsp_arrived = true;
          w.inflight[slot_id].rsp_data    = rsp.data;
        }
        ch.pop();
      }
    }

    // 2) Tick workers (smem_wr drain → gmem_req issue).
    for (auto& w : workers_) {
      if (w.state == WState::RUNNING) {
        tick_worker_smem_wr(w);
        tick_worker_gmem_req(w);
      }
    }

    // 3) Dispatch from queue to idle workers.
    for (auto& w : workers_) {
      if (w.state == WState::IDLE && !queue_.empty())
        start_worker(w, queue_.front()), queue_.pop_front();
    }

    // 4) Drain DxaUnit channels → req queue (round-robin).
    drain_req_in();
  }

  const DxaCore::PerfStats& perf_stats() const { return perf_stats_; }

private:
  // ── Descriptor helpers ───────────────────────────────────────────────
  static uint32_t desc_rank(uint32_t meta) {
    uint32_t r = (meta >> VX_DXA_DESC_META_DIM_LSB)
                  & ((1u << VX_DXA_DESC_META_DIM_BITS) - 1u);
    return (r == 0) ? 1u : std::min(r, 5u);
  }

  static uint32_t desc_elem_bytes(uint32_t meta) {
    uint32_t enc = (meta >> VX_DXA_DESC_META_ELEMSZ_LSB)
                    & ((1u << VX_DXA_DESC_META_ELEMSZ_BITS) - 1u);
    return 1u << enc;
  }

  // ── Pull from per-core dxa_req_in[] into queue_ (round-robin) ────────
  void drain_req_in() {
    auto& chs = simobject_->dxa_req_in;
    for (uint32_t i = 0; i < chs.size(); ++i) {
      uint32_t cid = (rr_req_ + i) % chs.size();
      auto& ch = chs.at(cid);
      if (ch.empty()) continue;
      if (queue_.size() >= DXA_QUEUE_SIZE) break;
      queue_.push_back(ch.peek());
      ch.pop();
      rr_req_ = (cid + 1) % chs.size();
      break;
    }
  }

  // ── Setup: pre-enumerate the work list for this transfer ─────────────
  void start_worker(Worker& w, const DxaReq& req) {
    w.req = req;
    w.issue_cycle = cycle_;
    w.work_list.clear();
    w.ag_idx = 0;
    w.issued_order.clear();
    for (auto& s : w.inflight) { s.allocated = false; s.rsp_arrived = false; s.rsp_data.reset(); }
    w.mc_cta_idx = 0;
    w.writes_emitted = 0;

    // Multicast setup.
    w.is_multicast = (__builtin_popcount(req.cta_mask) > 1);
    w.cta_indices.clear();
    if (w.is_multicast) {
      for (uint32_t i = 0; i < 32; ++i) {
        if (req.cta_mask & (1u << i)) w.cta_indices.push_back(i);
      }
    }

    if (req.desc_slot >= VX_DCR_DXA_DESC_COUNT) {
      // Invalid descriptor — release barrier(s) directly. This is the only
      // path that bypasses the LMEM completion flag (no LMEM write to ride
      // on). RTL handles this via the completion module's no-write timeout.
      release_all_barriers(w);
      finish_worker(w);
      return;
    }

    w.desc = descriptors_.at(req.desc_slot);
    w.smem_stride = w.desc.smem_stride;

    enumerate_work_list(w);
    if (w.work_list.empty()) {
      // No work — same edge case as above.
      release_all_barriers(w);
      finish_worker(w);
      return;
    }

    // Mark the *last* work item — its multicast replay's last write carries
    // notify_done.
    w.work_list.back().last = true;

    w.state = WState::RUNNING;
    perf_stats_.gmem_reads += w.work_list.size();

    DT(3, simobject_->name() << "[" << w.worker_id << "] start: core="
       << req.core->id() << ", wid=" << req.wid
       << ", slot=" << req.desc_slot
       << ", lines=" << w.work_list.size()
       << ", multicast=" << w.is_multicast
       << ", num_ctas=" << w.cta_indices.size());
  }

  // Enumerate (CL, smem-word, byte-offset, length, oob) tuples — one per
  // GMEM CL the transfer needs to read. Row-major iteration over outer
  // dims, contiguous span over dim-0, dedup of consecutive same-CL.
  void enumerate_work_list(Worker& w) {
    const auto& desc = w.desc;
    uint32_t rank = desc_rank(desc.meta);
    if (rank < 1 || rank > 5) return;

    uint32_t elem_bytes = desc_elem_bytes(desc.meta);
    std::array<uint32_t, 5> tiles = {};
    for (uint32_t d = 0; d < 5; ++d)
      tiles[d] = (d < rank) ? std::max<uint32_t>(1u, desc.tile_sizes[d]) : 1u;

    uint32_t total_rows = 1;
    for (uint32_t d = 1; d < rank; ++d) total_rows *= tiles[d];
    uint32_t row_elems = tiles[0];

    uint32_t cfill = desc.cfill;
    uint64_t global_prev_cl = ~uint64_t(0);

    for (uint32_t row = 0; row < total_rows; ++row) {
      // Decompose row into outer-dim local offsets (dim1 varies fastest).
      uint32_t outer[4] = {};
      uint32_t rem = row;
      for (uint32_t d = 1; d < rank; ++d) {
        outer[d - 1] = rem % tiles[d];
        rem /= tiles[d];
      }

      // OOB check on outer dims.
      bool row_in_bounds = true;
      for (uint32_t d = 1; d < rank; ++d) {
        if (w.req.coords[d] + outer[d - 1] >= desc.sizes[d]) { row_in_bounds = false; break; }
      }

      // First element GMEM addr in this row.
      uint64_t row_gbase = desc.base_addr + uint64_t(w.req.coords[0]) * elem_bytes;
      for (uint32_t d = 1; d < rank; ++d)
        row_gbase += uint64_t(w.req.coords[d] + outer[d - 1]) * uint64_t(desc.strides[d - 1]);

      // SMEM destination base for this row (dense row-major packing).
      uint64_t row_smem_base = w.req.smem_addr + uint64_t(row * row_elems) * elem_bytes;

      // Walk dim-0 elements; each LineWork carries one MemReq write of at
      // most one LMEM word AND at most one GMEM CL (so byteen fits in the
      // 64-byte mem_block payload).
      for (uint32_t e0 = 0; e0 < row_elems; ) {
        uint64_t gaddr_e = row_gbase + uint64_t(e0) * elem_bytes;
        uint64_t saddr_e = row_smem_base + uint64_t(e0) * elem_bytes;
        uint64_t cl_addr = gaddr_e & kGmemLineMask;
        uint64_t sword   = saddr_e & ~uint64_t(kLmemWordSize - 1);
        uint32_t cl_off  = uint32_t(gaddr_e - cl_addr);
        uint32_t s_off   = uint32_t(saddr_e - sword);
        uint32_t cl_room  = kGmemLineSize - cl_off;
        uint32_t sw_room  = kLmemWordSize - s_off;
        uint32_t row_room = (row_elems - e0) * elem_bytes;
        // Cap to MEM_BLOCK_SIZE (the byteen mask is 64 bits wide and the
        // mem_block_t payload is 64 bytes).
        uint32_t span = std::min({cl_room, sw_room, row_room, uint32_t(MEM_BLOCK_SIZE)});
        // Round to a whole-element multiple — ensures e0 advances by an
        // integer count and avoids off-by-one element splits.
        span -= span % elem_bytes;
        if (span == 0) break;

        bool elem_oob = !row_in_bounds || (w.req.coords[0] + e0 >= desc.sizes[0]);
        // (RTL splits dim-0 at the OOB boundary too; SimX marks OOB at
        // element-0 of the span. Tile widths are typically aligned to
        // dim-0 size, making this exact for the common case.)

        LineWork lw{};
        lw.gmem_cl_addr     = cl_addr;
        lw.smem_word_addr   = sword;
        lw.cl_byte_offset   = cl_off;
        lw.smem_byte_offset = s_off;
        lw.valid_length     = span;
        lw.cfill            = cfill;
        lw.oob              = elem_oob;
        lw.last             = false;

        // Dedup consecutive same-CL (non-OOB only).
        if (!elem_oob && cl_addr == global_prev_cl) {
          ++perf_stats_.gmem_dedup;
        } else if (!elem_oob) {
          global_prev_cl = cl_addr;
        }
        w.work_list.push_back(lw);

        e0 += span / elem_bytes;
      }
    }
  }

  // ── gmem_req: issue MemReqs with available inflight slots ────────────
  void tick_worker_gmem_req(Worker& w) {
    while (w.ag_idx < w.work_list.size()) {
      // Find a free slot.
      uint32_t slot = UINT32_MAX;
      for (uint32_t s = 0; s < w.inflight.size(); ++s) {
        if (!w.inflight[s].allocated) { slot = s; break; }
      }
      if (slot == UINT32_MAX) break; // all slots in flight

      const LineWork& lw = w.work_list[w.ag_idx];

      // OOB lines skip the GMEM request entirely; we synthesize an immediate
      // arrival (rsp data left null — smem_wr will use cfill).
      if (lw.oob) {
        w.inflight[slot].allocated   = true;
        w.inflight[slot].rsp_arrived = true;
        w.inflight[slot].rsp_data.reset();
        w.inflight[slot].work        = lw;
        w.issued_order.push_back(slot);
        ++w.ag_idx;
        continue;
      }

      // Issue real GMEM read. Tag = per-worker slot id (gmem_arb_ prepends
      // its own input-index bits for response routing — we only see slot
      // bits coming back).
      MemReq mreq;
      mreq.addr  = lw.gmem_cl_addr;
      mreq.tag   = slot;
      mreq.hart_id   = w.req.core->id();
      mreq.uuid  = w.req.uuid;

      auto& ch = gmem_arb_->ReqIn.at(w.worker_id);
      if (!ch.try_send(mreq)) break; // arb backpressure

      w.inflight[slot].allocated   = true;
      w.inflight[slot].rsp_arrived = false;
      w.inflight[slot].rsp_data.reset();
      w.inflight[slot].work        = lw;
      w.issued_order.push_back(slot);
      ++w.ag_idx;
    }
  }

  // ── smem_wr: drain ready slots, build LMEM MemReqs ───────────────────
  //
  // In-order mode (default): consume issued_order.front(). Stall on head.
  // OoO mode (-DDXA_OOO_DRAIN_ENABLE): scan inflight[] in slot-ID order
  // and pick the lowest-ID slot whose rsp_arrived is set. This mirrors
  // the RTL's VX_priority_encoder over rsp_arrived bitvector and exposes
  // the same latency-hiding benefit (a later-issued CL can drain first if
  // its GMEM response arrived sooner).
  void tick_worker_smem_wr(Worker& w) {
    if (w.issued_order.empty()) {
      // Nothing in flight — if all addr_gen done, transfer is finished.
      if (w.ag_idx == w.work_list.size()) finish_worker(w);
      return;
    }

#ifdef DXA_OOO_DRAIN_ENABLE
    // PE-equivalent: lowest-ID slot that is allocated and ready.
    int oo_slot = -1;
    for (uint32_t s = 0; s < uint32_t(w.inflight.size()); ++s) {
      if (w.inflight[s].allocated && w.inflight[s].rsp_arrived) {
        oo_slot = int(s);
        break;
      }
    }
    if (oo_slot < 0) return; // nothing ready
    uint32_t slot = uint32_t(oo_slot);
    auto& s = w.inflight[slot];
#else
    uint32_t slot = w.issued_order.front();
    auto& s = w.inflight[slot];
    if (!s.rsp_arrived) return; // wait
#endif

    // Determine destination core's LMEM port.
    uint32_t cluster_local_cid = w.req.core->id() % kCoresPerCluster;
    auto& lmem_ch = simobject_->lmem_req_out.at(cluster_local_cid);
    if (lmem_ch.full()) return; // backpressure

    const LineWork& lw = s.work;

    // Build LMEM MemReq with TLM payload.
    MemReq req;
    req.addr   = lw.smem_word_addr;
    req.op = MemOp::ST;
    req.tag    = w.req.core->id();           // routing tag
    req.hart_id = w.req.core->id();
    req.uuid   = w.req.uuid;
    // Build byteen carefully: (1<<64)-1 is UB, so synthesize ~0 directly when
    // the span fills the whole 64-byte block.
    uint64_t span_mask = (lw.valid_length >= 64)
                       ? ~uint64_t(0)
                       : ((uint64_t(1) << lw.valid_length) - 1ull);
    req.byteen = span_mask << lw.smem_byte_offset;
    auto blk = make_mem_block();
    if (s.rsp_data) {
      // Copy valid bytes from the GMEM CL response into the LMEM word
      // payload, shifted from cl_byte_offset to smem_byte_offset.
      std::memcpy(blk->data() + lw.smem_byte_offset,
                  s.rsp_data->data() + lw.cl_byte_offset,
                  lw.valid_length);
    } else {
      // OOB — fill with cfill pattern (lane-replicated).
      uint32_t pat = lw.cfill;
      for (uint32_t b = 0; b < lw.valid_length; ++b) {
        (*blk)[lw.smem_byte_offset + b] =
            uint8_t((pat >> ((b & 3) * 8)) & 0xff);
      }
    }
    req.data = blk;

    // For multicast, replay across cta_indices; for single, dest = req.smem_addr.
    uint32_t cta_warp_idx = 0;
    if (w.is_multicast) {
      cta_warp_idx = w.cta_indices.at(w.mc_cta_idx);
      req.addr += uint64_t(cta_warp_idx) * w.smem_stride;
    }

    // Set notify_done flag on the LAST LMEM write of the transfer (or
    // last per-CTA replay of the last work item under multicast). The
    // bus-snoop tx_callback registered on the per-core LMEM channel
    // reads this flag at packet delivery and pulses
    // barrier_event_release(notify_bar_id).
    bool is_last_work   = lw.last;
    bool is_last_replay = !w.is_multicast || (w.mc_cta_idx + 1 == w.cta_indices.size());
    if (is_last_work && (w.is_multicast || is_last_replay)) {
      req.flags.dxa_notify_done   = 1;
      req.flags.dxa_notify_bar_id = w.req.bar_id + (w.is_multicast ? cta_warp_idx : 0u);
    }

    lmem_ch.send(req);
    ++w.writes_emitted;
    ++perf_stats_.lmem_writes;

    // Advance multicast cursor or finish this slot.
    if (w.is_multicast && (w.mc_cta_idx + 1) < w.cta_indices.size()) {
      ++w.mc_cta_idx;
    } else {
      w.mc_cta_idx = 0;
      // Slot done — release.
      s.allocated = false;
      s.rsp_arrived = false;
      s.rsp_data.reset();
#ifdef DXA_OOO_DRAIN_ENABLE
      // In OoO mode the drained slot is not necessarily the head; remove it
      // wherever it sits in issued_order.
      auto it = std::find(w.issued_order.begin(), w.issued_order.end(), slot);
      if (it != w.issued_order.end()) w.issued_order.erase(it);
#else
      w.issued_order.pop_front();
#endif
    }
  }

  void release_all_barriers(Worker& w) {
    // bar_id is RAW (encoded); decode at release call site.
    if (w.is_multicast) {
      for (uint32_t cta : w.cta_indices) {
        uint32_t decoded = bar_decode_id(w.req.bar_id + cta, NUM_BARRIERS);
        w.req.core->barrier_event_release(decoded);
      }
    } else {
      uint32_t decoded = bar_decode_id(w.req.bar_id, NUM_BARRIERS);
      w.req.core->barrier_event_release(decoded);
    }
  }

  void finish_worker(Worker& w) {
    if (w.state == WState::RUNNING || w.state == WState::IDLE) {
      uint64_t latency = cycle_ - w.issue_cycle;
      ++perf_stats_.transfers;
      perf_stats_.total_latency += latency;
      DT(3, simobject_->name() << "[" << w.worker_id << "] complete: core="
         << w.req.core->id() << ", bar=" << w.req.bar_id
         << ", writes=" << w.writes_emitted << ", latency=" << latency);
    }
    w.state = WState::IDLE;
    w.work_list.clear();
    w.cta_indices.clear();
    w.issued_order.clear();
    w.ag_idx = 0;
    w.mc_cta_idx = 0;
    w.writes_emitted = 0;
  }

  // ── Members ──────────────────────────────────────────────────────────
  DxaCore*    simobject_;
  MemArbiter* gmem_arb_;
  std::array<Descriptor, VX_DCR_DXA_DESC_COUNT> descriptors_;
  std::deque<DxaReq>     queue_;
  std::vector<Worker>    workers_;
  uint32_t               rr_req_ = 0;
  uint64_t               cycle_;
  DxaCore::PerfStats     perf_stats_;
};

// ════════════════════════════════════════════════════════════════════
// DxaCore — wrappers
// ════════════════════════════════════════════════════════════════════

DxaCore::DxaCore(const SimContext& ctx, const char* name, Cluster* cluster)
  : SimObject<DxaCore>(ctx, name)
  , dxa_req_in(kCoresPerCluster, this)
  , gmem_req_out(kDxaMemPorts, this)
  , gmem_rsp_in(kDxaMemPorts, this)
  , lmem_req_out(kCoresPerCluster, this)
{
  __unused(cluster);

  // Build the GMEM arbiter (NUM_DXA_UNITS workers → kDxaMemPorts L2-facing).
  // Tag layout used by workers: high bit packs worker_id, low bits the
  // per-worker inflight slot. We pass TAG_SEL_IDX so the arb can route
  // responses back to the right input.
  char sname[100];
  snprintf(sname, 100, "%s-gmem-arb", name);
  gmem_arb_ = MemArbiter::Create(sname, ArbiterType::RoundRobin, NUM_DXA_UNITS, kDxaMemPorts);
  for (uint32_t i = 0; i < kDxaMemPorts; ++i) {
    gmem_arb_->ReqOut.at(i).bind(&gmem_req_out.at(i));
    gmem_rsp_in.at(i).bind(&gmem_arb_->RspIn.at(i));
  }

  impl_ = new Impl(this, gmem_arb_.get());
}

DxaCore::~DxaCore() {
  delete impl_;
}

void DxaCore::on_reset() { impl_->reset(); }
void DxaCore::on_tick()  { impl_->tick(); }

int DxaCore::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

const DxaCore::PerfStats& DxaCore::perf_stats() const {
  return impl_->perf_stats();
}
