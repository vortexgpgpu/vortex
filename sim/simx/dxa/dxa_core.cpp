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
#include "dxa_meta.h"
#include "lmem_slot.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <vector>
#include "core.h"
#include "socket.h"
#include "socket.h"
#include "mem_block_pool.h"
#include "debug.h"
#include "constants.h"

using namespace vortex;

// response tags index the in-flight slots by masking
static_assert(VX_CFG_DXA_MAX_INFLIGHT >= 2 && (VX_CFG_DXA_MAX_INFLIGHT & (VX_CFG_DXA_MAX_INFLIGHT - 1)) == 0,
              "DXA_MAX_INFLIGHT must be a power of two of at least 2");
static_assert(VX_CFG_DXA_MAX_XFERS >= 2 && (VX_CFG_DXA_MAX_XFERS & (VX_CFG_DXA_MAX_XFERS - 1)) == 0,
              "DXA_MAX_XFERS must be a power of two of at least 2");

namespace {

// One GMEM port: the DXA joins its socket's L2-facing arb as a single peer.
constexpr uint32_t kDxaMemPorts = 1;

// LMEM "word" granularity for splitting DXA writes. The LocalMem bank
// model applies byteen relative to a VX_CFG_MEM_BLOCK_SIZE-aligned address,
// so each LineWork's destination must lie within one VX_CFG_MEM_BLOCK_SIZE-aligned
// region. mem_block granularity is the byteen-mask scope.
constexpr uint32_t kLmemWordSize = VX_CFG_MEM_BLOCK_SIZE;

// GMEM line size + mask.
constexpr uint32_t kGmemLineSize = VX_CFG_L1_LINE_SIZE;
constexpr uint64_t kGmemLineMask = ~uint64_t(VX_CFG_L1_LINE_SIZE - 1);

// Cores served by one (socket-resident) DXA engine.
constexpr uint32_t kCoresPerSocket = VX_CFG_SOCKET_SIZE;
// A delivered line reaches its SMEM write through two pipeline registers.
constexpr uint32_t kSmemWrLatency = 2;

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
  // Each entry = ONE GMEM cache-line read. Row-major writes contribute it to a
  // single contiguous LMEM word write; a K-major or tiled load fans it out to
  // `km_num_elems` per-element destinations, which smem_wr writes one element
  // per beat (one read per cache line, never re-read per element).
  struct LineWork {
    uint64_t gmem_cl_addr;     // CL-aligned global address
    uint64_t smem_word_addr;   // word-aligned SMEM byte address (element 0)
    uint32_t cl_byte_offset;   // start of valid bytes within the CL (element 0)
    uint32_t smem_byte_offset; // destination offset within the SMEM word (element 0)
    uint32_t valid_length;     // valid bytes per write
    uint32_t cfill;            // OOB fill value (lane-replicated)
    bool     oob;              // skip GMEM read; use cfill
    bool     scatter;          // K-major/tiled: one element written per beat
    uint32_t km_num_elems;     // K-major scatter fan-out (1 = contiguous write)
    uint32_t km_lane_stride;   // SMEM byte stride between scattered elements
    uint8_t  dest_layout;      // DestLayout (Flat/BlockMajor use tiled_dest_elem)
    uint32_t tile_k_row;       // tiled scatter: K-row (dim1 index) for this span
    uint32_t tile_n_base;      // tiled scatter: N (dim0) index where the span starts
  };

  // ── Transfer context: one per transfer a worker has in flight ───────
  // The next transfer issues while earlier ones still drain, so everything
  // smem_wr needs per line lives here, indexed by the line's slot.
  struct Xfer {
    bool                  valid = false;
    bool                  issued = false;     // every line has been issued
    uint32_t              outstanding = 0;    // lines issued, not yet written
    uint64_t              free_cycle = 0;     // first cycle the context can be reused
    DxaReq                req;
    Descriptor            desc;
    bool                  is_multicast = false;
    std::vector<uint32_t> cta_indices;
    uint32_t              smem_stride = 0;
    uint64_t              issue_cycle = 0;
    uint32_t              writes_emitted = 0;
  };

  // ── Inflight slot for a GMEM read ──────────────────────────────────
  struct InflightSlot {
    bool      allocated = false;
    bool      rsp_arrived = false;
    uint64_t  ready_cycle = 0;  // first cycle the line can be written to SMEM
    uint64_t  free_cycle = 0;   // first cycle the slot can carry a new read
    uint32_t  xfer = 0;         // owning transfer context
    LineWork  work;            // captured at allocation
    std::shared_ptr<mem_block_t> rsp_data;
  };

  // ── Worker (one per slice) ───────────────────────────────────────────
  struct Worker {
    uint32_t                worker_id = 0;

    // setup engine: the next transfer is set up while this one issues
    bool                    staged = false;
    DxaReq                  staged_req;
    uint64_t                staged_ready = 0;

    // addr_gen: pre-enumerated work list of the transfer being issued (one
    // entry per GMEM CL; multicast replays are handled at the smem_wr stage).
    bool                    issuing = false;
    uint32_t                issue_xfer = 0;
    std::vector<LineWork>   work_list;
    uint32_t                ag_idx = 0;          // next entry to issue (gmem_req)

    std::array<Xfer, VX_CFG_DXA_MAX_XFERS> xfers;

    // gmem_req → smem_wr inflight bookkeeping.
    std::array<InflightSlot, VX_CFG_DXA_MAX_INFLIGHT> inflight;
    // Tag pools: never-used slots count up from 0, released slots recycle in
    // release order; OOB lines wait for smem_wr in issue order.
    uint32_t                fresh_slots = 0;
    std::deque<uint32_t>    free_fifo;
    std::deque<uint32_t>    oob_fifo;
    uint32_t                stg_slot = UINT32_MAX;   // response stage (slot read)
    uint32_t                pend_slot = UINT32_MAX;  // captured, waiting to drain
    uint32_t                drain_slot = UINT32_MAX; // slot currently draining
    bool                    drain_final = false; // drain_slot is its transfer's last line

    // smem_wr multicast replay state.
    uint32_t                mc_cta_idx = 0;      // 0..cta_indices.size()
    uint32_t                km_elem_idx = 0;     // 0..km_num_elems (K-major scatter)
  };

  // ── Constructor ──────────────────────────────────────────────────────
  explicit Impl(DxaCore* simobject, MemArbiter* gmem_arb)
    : simobject_(simobject)
    , gmem_arb_(gmem_arb)
    , workers_(VX_CFG_NUM_DXA_CORES)
    , cycle_(0)
  {
    for (uint32_t i = 0; i < VX_CFG_NUM_DXA_CORES; ++i)
      workers_[i].worker_id = i;
  }

  void reset() {
    cycle_ = 0;
    queue_.clear();
    // perf_stats_ survives: the MPM counters run from device reset.
    for (auto& w : workers_) {
      w.staged = false;
      w.issuing = false;
      w.work_list.clear();
      w.ag_idx = 0;
      for (auto& x : w.xfers) { x = Xfer(); }
      for (auto& s : w.inflight) { s = InflightSlot(); }
      w.fresh_slots = 0;
      w.free_fifo.clear();
      w.oob_fifo.clear();
      w.stg_slot = UINT32_MAX;
      w.pend_slot = UINT32_MAX;
      w.drain_slot = UINT32_MAX;
      w.drain_final = false;
      w.mc_cta_idx = 0;
      w.km_elem_idx = 0;
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
    case VX_DCR_DXA_DESC_SMEM_STRIDE_OFF:
      // Round up to the CTA slot pitch so multicast destination addresses
      // `issuer_addr + r * smem_stride` stay block-aligned. The LMEM model
      // treats requests as block-aligned with a byteen mask; a non-aligned
      // stride would truncate the address and land writes in the wrong
      // block. The dispatcher applies the same rounding to per-CTA LMEM
      // allocations so the host-visible "same offset in receiver LMEM"
      // contract holds end-to-end.
      d.smem_stride = lmem_slot_pitch(value);
      break;
    default: break;
    }
    return 0;
  }

  void tick() {
    ++cycle_;

    // Drain in reverse-pipeline order (smem_wr → gmem_req → addr_gen →
    // setup → req-arb) so each stage's tick reads what its upstream produced
    // earlier this cycle, mimicking forward stage flow.

    // 1) smem_wr drain, then 2) hand it one line per worker, then 3) issue
    // the next GMEM read. A line leaving pend for the fill buffer frees pend
    // for a capture in the same cycle. gmem_arb_ already routes responses
    // back to RspOut[worker_id] (its input index) and restores the user's tag
    // (the per-worker slot id).
    for (uint32_t worker_id = 0; worker_id < workers_.size(); ++worker_id) {
      auto& w = workers_[worker_id];
      tick_worker_smem_wr(w);
      capture_line(w, gmem_arb_->RspOut.at(worker_id));
      if (w.issuing) {
        tick_worker_gmem_req(w);
      }
    }

    // 3) Start a set-up transfer once the worker has issued the previous
    //    one and holds a free transfer context; otherwise let the worker's
    //    setup engine take the next request.
    for (auto& w : workers_) {
      if (w.staged && !w.issuing && cycle_ >= w.staged_ready) {
        uint32_t xi = free_xfer(w);
        if (xi != UINT32_MAX) {
          w.staged = false;
          start_worker(w, xi, w.staged_req);
        }
      } else if (!w.staged && !queue_.empty()) {
        w.staged_req   = queue_.front();
        w.staged_ready = cycle_ + this->setup_latency(w.staged_req);
        w.staged       = true;
        queue_.pop_front();
      }
    }

    // 4) Drain DxaUnit channels → req queue (round-robin).
    drain_req_in();
  }

  const DxaCore::PerfStats& perf_stats() const { return perf_stats_; }

private:
  // ── Descriptor helpers ───────────────────────────────────────────────
  static uint32_t desc_rank(uint32_t meta) {
    uint32_t r = (meta >> DXA_DESC_META_DIM_LSB)
                  & ((1u << DXA_DESC_META_DIM_BITS) - 1u);
    return (r == 0) ? 1u : std::min(r, 5u);
  }

  static uint32_t desc_elem_bytes(uint32_t meta) {
    uint32_t enc = (meta >> DXA_DESC_META_ELEMSZ_LSB)
                    & ((1u << DXA_DESC_META_ELEMSZ_BITS) - 1u);
    return 1u << enc;
  }

  // Destination SMEM layout (2-bit LAYOUT meta field). Mirrors dxa.h::Layout.
  enum class DestLayout : uint32_t { RowMajor = 0, KMajor = 1, Flat = 2, BlockMajor = 3 };

  static DestLayout desc_layout(uint32_t meta) {
    uint32_t v = (meta >> DXA_DESC_META_LAYOUT_LSB)
                 & ((1u << DXA_DESC_META_LAYOUT_BITS) - 1u);
    return DestLayout(v);
  }

  // K-major destination layout (TMA style): SMEM addr per element is
  // base + i1 * elem_bytes + e0 * tile1 * elem_bytes (instead of the default
  // row-major base + i1 * tile0 * elem_bytes + e0 * elem_bytes).
  static bool desc_dest_kmajor(uint32_t meta) { return desc_layout(meta) == DestLayout::KMajor; }

  // Flat (sparse) / BlockMajor (dense) reuse the K-major SCATTER datapath
  // (contiguous GMEM read → permuted SMEM write) with a richer per-element
  // destination index. The tile geometry rides ESTRIDE2 (set_tile_geometry).
  static bool desc_dest_tiled(uint32_t meta) {
    DestLayout l = desc_layout(meta);
    return l == DestLayout::Flat || l == DestLayout::BlockMajor;
  }

  // WGMMA tile geometry conveyed by set_tile_geometry() via ESTRIDE2:
  // tcN in [15:0], the dense B block K extent (elements) in [31:16].
  static uint32_t desc_geo_tcn(const Descriptor& d) {
    return std::max<uint32_t>(1u, d.element_strides[2] & 0xffffu);
  }
  static uint32_t desc_geo_bkk(const Descriptor& d) {
    return std::max<uint32_t>(1u, d.element_strides[2] >> 16);
  }

  // SMEM element-index for B element (k = K-row, n = N-col) under the bbuf's
  // native layouts. Mirrors vx_tensor.h::b_sp_flat_idx / b_blockmajor_idx
  // (tcK == tcN for canonical WGMMA configs; ratio = 32-bit-word / elem).
  static uint32_t tiled_dest_elem(DestLayout lay, uint32_t k, uint32_t n,
                                  uint32_t ratio, uint32_t tcN, uint32_t n_steps,
                                  uint32_t bkK) {
    if (lay == DestLayout::Flat) {
      uint32_t b_tcK_words = tcN * 2;
      uint32_t blk_words   = tcN * b_tcK_words;
      uint32_t k_word = k / ratio;
      uint32_t elem   = k % ratio;
      uint32_t k_blk  = k_word / b_tcK_words;
      uint32_t kw_in  = k_word % b_tcK_words;
      uint32_t n_blk  = n / tcN;
      uint32_t n_in   = n % tcN;
      uint32_t word_off = (k_blk * n_steps + n_blk) * blk_words + (kw_in * tcN + n_in);
      return word_off * ratio + elem;
    }
    // BlockMajor (dense): within-block N-outer, K-inner; the block K extent
    // follows the consumer's FEDP depth and is conveyed by the descriptor.
    uint32_t kw          = bkK;
    uint32_t b_blk_elems = kw * tcN;
    uint32_t k_blk = k / kw;
    uint32_t r_in  = k % kw;
    uint32_t n_blk = n / tcN;
    uint32_t n_in  = n % tcN;
    return (k_blk * n_steps + n_blk) * b_blk_elems + n_in * kw + r_in;
  }

  // Parameter setup takes longer for higher-rank descriptors: each extra
  // dimension adds a multiply phase.
  uint32_t setup_latency(const DxaReq& req) const {
    uint32_t rank = (req.desc_slot < VX_DCR_DXA_DESC_COUNT)
                  ? desc_rank(descriptors_.at(req.desc_slot).meta) : 1;
    return 2 * std::max<uint32_t>(rank, 2) + 2;
  }

  // ── Pull from per-core dxa_req_in[] into queue_ (round-robin) ────────
  void drain_req_in() {
    auto& chs = simobject_->dxa_req_in;
    for (uint32_t i = 0; i < chs.size(); ++i) {
      uint32_t cid = (rr_req_ + i) % chs.size();
      auto& ch = chs.at(cid);
      if (ch.empty()) continue;
      if (queue_.size() >= VX_CFG_DXA_QUEUE_SIZE) break;
      queue_.push_back(ch.peek());
      ch.pop();
      rr_req_ = (cid + 1) % chs.size();
      break;
    }
  }

  uint32_t free_xfer(const Worker& w) const {
    for (uint32_t i = 0; i < w.xfers.size(); ++i) {
      if (!w.xfers[i].valid && cycle_ >= w.xfers[i].free_cycle)
        return i;
    }
    return UINT32_MAX;
  }

  // ── Setup: pre-enumerate the work list for this transfer ─────────────
  void start_worker(Worker& w, uint32_t xi, const DxaReq& req) {
    auto& x = w.xfers[xi];
    x.req = req;
    x.issue_cycle = cycle_;
    x.writes_emitted = 0;

    // Multicast setup.
    x.is_multicast = (__builtin_popcount(req.cta_mask) > 1);
    x.cta_indices.clear();
    if (x.is_multicast) {
      for (uint32_t i = 0; i < 32; ++i) {
        if (req.cta_mask & (1u << i)) x.cta_indices.push_back(i);
      }
    }

    if (req.desc_slot >= VX_DCR_DXA_DESC_COUNT) {
      // Invalid descriptor — release barrier(s) directly without any LMEM write.
      release_all_barriers(x);
      count_transfer(w, x);
      return;
    }

    x.desc = descriptors_.at(req.desc_slot);
    x.smem_stride = x.desc.smem_stride;

    w.work_list.clear();
    enumerate_work_list(x, w.work_list);
    if (w.work_list.empty()) {
      // No work — same edge case as above.
      release_all_barriers(x);
      count_transfer(w, x);
      return;
    }

    x.valid = true;
    x.issued = false;
    x.outstanding = 0;
    w.issuing = true;
    w.issue_xfer = xi;
    w.ag_idx = 0;
    perf_stats_.gmem_reads += w.work_list.size();

    DT(3, simobject_->name() << "[" << w.worker_id << "] start: core="
       << req.core->id() << ", wid=" << req.wid
       << ", slot=" << req.desc_slot
       << ", xfer=" << xi
       << ", lines=" << w.work_list.size()
       << ", multicast=" << x.is_multicast
       << ", num_ctas=" << x.cta_indices.size());
  }

  // Enumerate (CL, smem-word, byte-offset, length, oob) tuples — one per
  // GMEM CL the transfer needs to read. Row-major iteration over outer
  // dims, contiguous span over dim-0, dedup of consecutive same-CL.
  void enumerate_work_list(const Xfer& x, std::vector<LineWork>& work_list) {
    const auto& desc = x.desc;
    uint32_t rank = desc_rank(desc.meta);
    if (rank < 1 || rank > 5) return;

    uint32_t   elem_bytes  = desc_elem_bytes(desc.meta);
    DestLayout layout      = desc_layout(desc.meta);
    bool       dest_kmajor = (layout == DestLayout::KMajor);
    bool       dest_tiled  = desc_dest_tiled(desc.meta);
    std::array<uint32_t, 5> tiles = {};
    for (uint32_t d = 0; d < 5; ++d)
      tiles[d] = (d < rank) ? std::max<uint32_t>(1u, desc.tile_sizes[d]) : 1u;

    // K-major / tiled (Flat, BlockMajor) scatter require rank ≤ 2.
    if ((dest_kmajor || dest_tiled) && rank > 2) {
      dest_kmajor = false;
      dest_tiled  = false;
      layout      = DestLayout::RowMajor;
    }
    // Both share the single-element scatter datapath (one read → per-element
    // strided/permuted writes); row-major writes the span contiguously.
    bool scatter = dest_kmajor || dest_tiled;

    uint32_t total_rows = 1;
    for (uint32_t d = 1; d < rank; ++d) total_rows *= tiles[d];
    uint32_t row_elems = tiles[0];

    // K-major SMEM per-lane stride = tile1 * elem_bytes.
    uint32_t per_lane_stride_bytes = tiles[1] * elem_bytes;

    // Tiled (Flat/BlockMajor) geometry: ratio = 32-bit-word / elem, tcN from
    // ESTRIDE2, n_steps = row_elems(=xtileN) / tcN.
    uint32_t geo_ratio  = dest_tiled ? std::max<uint32_t>(1u, 4u / elem_bytes) : 1u;
    uint32_t geo_tcN    = dest_tiled ? desc_geo_tcn(desc) : 1u;
    uint32_t geo_bkK    = dest_tiled ? desc_geo_bkk(desc) : 1u;
    uint32_t geo_nsteps = dest_tiled ? std::max<uint32_t>(1u, row_elems / geo_tcN) : 1u;

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
        if (x.req.coords[d] + outer[d - 1] >= desc.sizes[d]) { row_in_bounds = false; break; }
      }

      // First element GMEM addr in this row.
      uint64_t row_gbase = desc.base_addr + uint64_t(x.req.coords[0]) * elem_bytes;
      for (uint32_t d = 1; d < rank; ++d)
        row_gbase += uint64_t(x.req.coords[d] + outer[d - 1]) * uint64_t(desc.strides[d - 1]);

      // SMEM destination base for this row:
      //   Row-major:  base + row_idx * row_elems * elem_bytes  (i1 outer × i0 inner).
      //   K-major:    base + outer[0] * elem_bytes              (lane stride applies per e0 below).
      //   Tiled:      computed per-element below (smem_wr recomputes the permuted dest).
      uint64_t row_smem_base = dest_kmajor
          ? (x.req.smem_addr + uint64_t(outer[0]) * elem_bytes)
          : (x.req.smem_addr + uint64_t(row * row_elems) * elem_bytes);

      // Walk dim-0 elements; each LineWork carries one MemReq write of at
      // most one LMEM word AND at most one GMEM CL (so byteen fits in the
      // 64-byte mem_block payload). K-major scatters each element to its
      // own SMEM destination (lane stride = tile1 * elem_bytes).
      for (uint32_t e0 = 0; e0 < row_elems; ) {
        uint64_t gaddr_e = row_gbase + uint64_t(e0) * elem_bytes;
        uint64_t saddr_e;
        if (dest_tiled) {
          uint32_t de = tiled_dest_elem(layout, outer[0], e0, geo_ratio, geo_tcN, geo_nsteps, geo_bkK);
          saddr_e = x.req.smem_addr + uint64_t(de) * elem_bytes;
        } else if (dest_kmajor) {
          saddr_e = row_smem_base + uint64_t(e0) * uint64_t(per_lane_stride_bytes);
        } else {
          saddr_e = row_smem_base + uint64_t(e0) * elem_bytes;
        }
        uint64_t cl_addr = gaddr_e & kGmemLineMask;
        uint64_t sword   = saddr_e & ~uint64_t(kLmemWordSize - 1);
        uint32_t cl_off  = uint32_t(gaddr_e - cl_addr);
        uint32_t s_off   = uint32_t(saddr_e - sword);
        uint32_t cl_room  = kGmemLineSize - cl_off;
        uint32_t sw_room  = kLmemWordSize - s_off;
        uint32_t row_room = (row_elems - e0) * elem_bytes;
        // The GMEM read span is ALWAYS coalesced to the cache line: dim-0
        // elements are contiguous in global memory, so one cache-line read
        // covers as many as fit. K-major then SCATTERS those elements to
        // strided SMEM destinations (one read → km_num_elems writes); row-major
        // writes them contiguously (bounded by the SMEM word). This mirrors the
        // hardware address generator, which reads per cache line and never
        // re-reads per element.
        uint32_t gspan = scatter
            ? std::min({cl_room, row_room, uint32_t(VX_CFG_MEM_BLOCK_SIZE)})
            : std::min({cl_room, sw_room, row_room, uint32_t(VX_CFG_MEM_BLOCK_SIZE)});
        // Round to a whole-element multiple — ensures e0 advances by an
        // integer count and avoids off-by-one element splits.
        gspan -= gspan % elem_bytes;
        if (gspan == 0) break;
        uint32_t num_elems = gspan / elem_bytes;

        bool elem_oob = !row_in_bounds || (x.req.coords[0] + e0 >= desc.sizes[0]);
        // OOB is determined per-element at the start of the span; tile widths
        // are typically aligned to dim-0 size, so this is exact for the common case.

        LineWork lw{};
        lw.gmem_cl_addr     = cl_addr;
        lw.smem_word_addr   = sword;
        lw.cl_byte_offset   = cl_off;
        lw.smem_byte_offset = s_off;
        // Row-major: one contiguous write of the whole span. Scatter (K-major
        // or tiled Flat/BlockMajor): each element is its own write (elem_bytes).
        lw.valid_length     = scatter ? elem_bytes : gspan;
        lw.cfill            = cfill;
        lw.oob              = elem_oob;
        lw.scatter          = scatter;
        lw.km_num_elems     = scatter ? num_elems : 1;
        lw.km_lane_stride   = dest_kmajor ? per_lane_stride_bytes : 0;
        lw.dest_layout      = uint8_t(layout);
        lw.tile_k_row       = outer[0];   // tiled scatter: K-row (dim1)
        lw.tile_n_base      = e0;          // tiled scatter: N (dim0) span start

        // Dedup consecutive same-CL (non-OOB only).
        if (!elem_oob && cl_addr == global_prev_cl) {
          ++perf_stats_.gmem_dedup;
        } else if (!elem_oob) {
          global_prev_cl = cl_addr;
        }
        work_list.push_back(lw);

        e0 += num_elems;
      }
    }
  }

  // ── Response stage, then smem_wr capture ────────────────────────────
  // The slot table is a RAM with a registered read, so a line (a GMEM
  // response, which has priority, or a pending OOB line) first spends a cycle
  // in a one-entry stage while its slot is read. smem_wr captures the staged
  // line once its line register is free; the stage refills in the same cycle.
  // A response that cannot enter the stage stays on the memory response path
  // and holds the responses queued behind it.
  void capture_line(Worker& w, SimChannel<MemRsp>& ch) {
    if (w.stg_slot != UINT32_MAX && w.pend_slot == UINT32_MAX) {
      auto& s = w.inflight[w.stg_slot];
      s.rsp_arrived = true;
      s.ready_cycle = cycle_ + kSmemWrLatency;
      w.pend_slot   = w.stg_slot;
      w.stg_slot    = UINT32_MAX;
    }

    uint32_t slot = UINT32_MAX;
    if (!ch.empty()) {
      slot = uint32_t(ch.peek().tag) & (VX_CFG_DXA_MAX_INFLIGHT - 1);
      if (!w.inflight[slot].allocated) {
        ch.pop();
        return;
      }
    } else if (!w.oob_fifo.empty()) {
      slot = w.oob_fifo.front();
    }
    if (slot == UINT32_MAX || w.stg_slot != UINT32_MAX) {
      return;
    }

    auto& s = w.inflight[slot];
    if (!ch.empty()) {
      s.rsp_data = ch.peek().data;
      ch.pop();
    } else {
      w.oob_fifo.pop_front();
    }
    w.stg_slot = slot;
  }

  // ── gmem_req: issue MemReqs with available inflight slots ────────────
  void tick_worker_gmem_req(Worker& w) {
    // One AGU step per cycle: advance ag_idx by at most one, issuing at most
    // one GMEM read to this worker's ReqIn port (or one OOB skip). A while-loop
    // here would issue many reads to a single port in one cycle.

    // Next free slot: a never-used one, else the oldest released one (a
    // slot released this cycle is reusable from the next).
    uint32_t slot = UINT32_MAX;
    if (w.fresh_slots < w.inflight.size()) {
      slot = w.fresh_slots;
    } else if (!w.free_fifo.empty() && cycle_ >= w.inflight[w.free_fifo.front()].free_cycle) {
      slot = w.free_fifo.front();
    }

    const LineWork& lw = w.work_list[w.ag_idx];

    if (slot == UINT32_MAX) {
      // All slots in flight. Out-of-bounds lines issue no read, so they are
      // excluded: this counts only reads the slot pool actually held back.
      if (!lw.oob) {
        ++perf_stats_.noslot_stalls;
      }
      return;
    }

    // OOB lines skip the GMEM request entirely; smem_wr takes them in a cycle
    // with no GMEM response (rsp data left null — smem_wr will use cfill).
    if (lw.oob) {
      w.oob_fifo.push_back(slot);
      issue_line(w, slot, lw);
      return;
    }

    // Issue real GMEM read. Tag = per-worker slot id (gmem_arb_ prepends
    // its own input-index bits for response routing — we only see slot
    // bits coming back).
    MemReq mreq;
    mreq.addr  = lw.gmem_cl_addr;
    mreq.tag   = slot;
    const auto& req = w.xfers[w.issue_xfer].req;
    mreq.hart_id   = req.core->id();
    mreq.uuid  = req.uuid;

    auto& ch = gmem_arb_->ReqIn.at(w.worker_id);
    if (!ch.try_send(mreq)) return; // arb backpressure

    issue_line(w, slot, lw);
  }

  void issue_line(Worker& w, uint32_t slot, const LineWork& lw) {
    if (slot == w.fresh_slots) {
      ++w.fresh_slots;
    } else {
      w.free_fifo.pop_front();
    }
    auto& s = w.inflight[slot];
    s.allocated   = true;
    s.rsp_arrived = false;
    s.xfer        = w.issue_xfer;
    s.rsp_data.reset();
    s.work        = lw;
    auto& x = w.xfers[w.issue_xfer];
    ++x.outstanding;
    if (++w.ag_idx == w.work_list.size()) {
      x.issued  = true;
      w.issuing = false;
    }
  }

  // ── smem_wr: drain ready slots, build LMEM MemReqs ───────────────────
  void tick_worker_smem_wr(Worker& w) {
    // Lines drain in arrival order, whatever their transfer. The line that
    // leaves its transfer with nothing else outstanding once the transfer
    // has issued every line is the final one: its write carries notify_done,
    // so it lands after every other write of the transfer.
    if (w.drain_slot == UINT32_MAX) {
      uint32_t p = w.pend_slot;
      if (p == UINT32_MAX || cycle_ < w.inflight[p].ready_cycle)
        return;
      const auto& px = w.xfers[w.inflight[p].xfer];
      w.drain_final = px.issued && (px.outstanding == 1);
      w.drain_slot  = p;
      w.pend_slot   = UINT32_MAX;
    }

    auto& x = w.xfers[w.inflight[w.drain_slot].xfer];

    // Determine destination core's LMEM port group.
    uint32_t socket_local_cid = x.req.core->id() % kCoresPerSocket;

    // One drain beat per cycle covers one aligned LMEM row
    // (DxaCore::LMEM_ROW_SIZE bytes — the full banked row the RTL writes in
    // a single cycle). A row wider than the 64B mem_block byteen scope is
    // emitted as LMEM_PORTS_PER_CORE same-cycle block writes, one per row
    // half on its own LMEM DMA channel. A row narrower than a block
    // (small NT) bounds the per-beat gather instead.

    // A banked-row write is committed atomically, byte enables aside, so a row is
    // never half-written. The row spans LMEM_PORTS_PER_CORE channels that fill
    // independently, so gate the beat on the whole group being ready instead of
    // stranding the half already sent.
    for (uint32_t p = 0; p < DxaCore::LMEM_PORTS_PER_CORE; ++p) {
      if (simobject_->lmem_req_out.at(socket_local_cid * DxaCore::LMEM_PORTS_PER_CORE + p).full())
        return;
    }

    constexpr uint64_t kRowMask = ~uint64_t(DxaCore::LMEM_ROW_SIZE - 1);
    uint64_t beat_row = UINT64_MAX;

    for (uint32_t emitted = 0; emitted < DxaCore::LMEM_PORTS_PER_CORE; ++emitted) {
      uint32_t slot = w.drain_slot;
      if (slot == UINT32_MAX) {
        break; // slot released mid-beat
      }
      auto& s = w.inflight[slot];

      const LineWork& lw = s.work;

      // One cache-line read fans out to its scattered writes, one element per
      // beat: the scatter destinations are not contiguous, so each element is
      // its own byte-masked write. Cursors: km_elem_idx advances one element
      // per beat; mc_cta_idx replays the whole group to each multicast
      // receiver.
      const uint32_t num_elems = lw.km_num_elems ? lw.km_num_elems : 1;
      const uint32_t e0        = w.km_elem_idx;
      const uint32_t wlen      = lw.valid_length;

      uint32_t cta_warp_idx = 0;
      uint64_t cta_off = 0;
      if (x.is_multicast) {
        cta_warp_idx = x.cta_indices.at(w.mc_cta_idx);
        cta_off = uint64_t(cta_warp_idx) * x.smem_stride;
      }

      // Per-element SMEM byte destination. K-major: uniform lane stride from a
      // span base. Tiled (Flat/BlockMajor): permuted index from the WGMMA
      // B-buffer layout formula (idx = within-span dim-0 offset → N = base+idx).
      const bool     tiled    = desc_dest_tiled(x.desc.meta);
      const uint32_t t_eb     = desc_elem_bytes(x.desc.meta);
      const uint32_t t_ratio  = tiled ? std::max<uint32_t>(1u, 4u / t_eb) : 1u;
      const uint32_t t_tcN    = tiled ? desc_geo_tcn(x.desc) : 1u;
      const uint32_t t_bkK    = tiled ? desc_geo_bkk(x.desc) : 1u;
      const uint32_t t_nsteps = tiled ? std::max<uint32_t>(1u, uint32_t(x.desc.tile_sizes[0]) / t_tcN) : 1u;
      auto dest_byte_of = [&](uint32_t idx) -> uint64_t {
        if (tiled) {
          uint32_t de = tiled_dest_elem(DestLayout(lw.dest_layout), lw.tile_k_row,
                                        lw.tile_n_base + idx, t_ratio, t_tcN, t_nsteps, t_bkK);
          return x.req.smem_addr + uint64_t(de) * t_eb + cta_off;
        }
        return lw.smem_word_addr + lw.smem_byte_offset
             + uint64_t(idx) * lw.km_lane_stride + cta_off;
      };

      // The block targeted this write = the block of scatter element e0.
      uint64_t base_byte0 = dest_byte_of(e0);
      uint64_t dword = base_byte0 & ~uint64_t(kLmemWordSize - 1);

      // Row constraint: every write of this beat stays in one aligned LMEM row.
      uint64_t row = base_byte0 & kRowMask;
      if (beat_row == UINT64_MAX) {
        beat_row = row;
      } else if (row != beat_row) {
        break; // next block lands in a different row — next cycle's beat
      }

      // Route this row half to its own LMEM DMA channel.
      uint32_t half = (DxaCore::LMEM_PORTS_PER_CORE > 1)
          ? uint32_t((dword / kLmemWordSize) & (DxaCore::LMEM_PORTS_PER_CORE - 1))
          : 0u;
      auto& lmem_ch = simobject_->lmem_req_out.at(
          socket_local_cid * DxaCore::LMEM_PORTS_PER_CORE + half);
      if (lmem_ch.full()) {
        break; // backpressure
      }

      // Build LMEM MemReq with TLM payload.
      MemReq req;
      req.addr   = dword;
      req.op = MemOp::ST;
      req.tag    = x.req.core->id();           // routing tag
      req.hart_id = x.req.core->id();
      req.uuid   = x.req.uuid;
      uint64_t byteen = 0;
      auto blk = make_mem_block();
      const uint32_t pat = lw.cfill;

      // A contiguous span is one write; a scatter span writes element e0.
      const bool     beat_scatter = lw.scatter;
      const uint32_t e_end = beat_scatter ? (e0 + 1) : num_elems;
      uint32_t ee = e0;
      for (; ee < e_end; ++ee) {
        uint64_t dest_byte = dest_byte_of(ee);
        if ((dest_byte & ~uint64_t(kLmemWordSize - 1)) != dword) break;
        if ((dest_byte & kRowMask) != row) {
          break;
        }
        uint32_t doff  = uint32_t(dest_byte - dword);
        uint64_t emask = (wlen >= 64) ? ~uint64_t(0) : ((uint64_t(1) << wlen) - 1ull);
        byteen |= emask << doff;
        if (s.rsp_data) {
          std::memcpy(blk->data() + doff,
                      s.rsp_data->data() + lw.cl_byte_offset + ee * wlen,
                      wlen);
        } else {
          for (uint32_t b = 0; b < wlen; ++b)
            (*blk)[doff + b] = uint8_t((pat >> ((b & 3) * 8)) & 0xff);
        }
      }
      req.byteen = byteen;
      req.data = blk;

      // notify_done on the LAST block write of the transfer — when the write
      // reached the last scatter element of the final line, per receiver.
      bool is_last_elem   = (ee == num_elems);
      bool is_last_replay = !x.is_multicast || (w.mc_cta_idx + 1 == x.cta_indices.size());
      if (w.drain_final && is_last_elem && (x.is_multicast || is_last_replay)) {
        req.flags.dxa_notify_done   = 1;
        req.flags.dxa_notify_bar_id = x.req.bar_id + (x.is_multicast ? cta_warp_idx : 0u);
      }

      lmem_ch.send(req);
      ++x.writes_emitted;
      // Count write transactions, not port requests: one per banked-row beat.
      // A beat spans LMEM_PORTS_PER_CORE ports of the same row, so only its
      // first emission is counted.
      if (0 == emitted) {
        ++perf_stats_.lmem_writes;
      }

      // Advance scatter cursor (to the next unwritten element); then multicast
      // cursor; then release the slot.
      if (!is_last_elem) {
        w.km_elem_idx = ee;
      } else {
        w.km_elem_idx = 0;
        if (x.is_multicast && (w.mc_cta_idx + 1) < x.cta_indices.size()) {
          ++w.mc_cta_idx;
        } else {
          w.mc_cta_idx = 0;
          // Slot done — release.
          s.allocated = false;
          s.rsp_arrived = false;
          s.free_cycle = cycle_ + 1;
          w.free_fifo.push_back(slot);
          s.rsp_data.reset();
          --x.outstanding;
          if (w.drain_final) {
            finish_xfer(w, x);
          }
          w.drain_slot = UINT32_MAX;
        }
      }
      if (beat_scatter) {
        break; // a scatter beat writes one element
      }
    } // per-tick row-beat emission loop
  }

  void release_all_barriers(const Xfer& x) {
    // bar_id is RAW (encoded); decode at release call site.
    if (x.is_multicast) {
      for (uint32_t cta : x.cta_indices) {
        uint32_t decoded = bar_decode_id(x.req.bar_id + cta, VX_CFG_NUM_BARRIERS);
        x.req.core->barrier_event_release(decoded);
      }
    } else {
      uint32_t decoded = bar_decode_id(x.req.bar_id, VX_CFG_NUM_BARRIERS);
      x.req.core->barrier_event_release(decoded);
    }
  }

  void count_transfer(const Worker& w, const Xfer& x) {
    __unused(w);
    uint64_t latency = cycle_ - x.issue_cycle;
    ++perf_stats_.transfers;
    perf_stats_.total_latency += latency;
    DT(3, simobject_->name() << "[" << w.worker_id << "] complete: core="
       << x.req.core->id() << ", bar=" << x.req.bar_id
       << ", writes=" << x.writes_emitted << ", latency=" << latency);
  }

  void finish_xfer(Worker& w, Xfer& x) {
    count_transfer(w, x);
    x.valid = false;
    x.free_cycle = cycle_ + 1;
    x.cta_indices.clear();
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

DxaCore::DxaCore(const SimContext& ctx, const char* name, Socket* socket)
  : SimObject<DxaCore>(ctx, name)
  , dxa_req_in(kCoresPerSocket, this)
  , gmem_req_out(kDxaMemPorts, this)
  , gmem_rsp_in(kDxaMemPorts, this)
  , lmem_req_out(kCoresPerSocket * LMEM_PORTS_PER_CORE, this)
{
  __unused(socket);

  // Build the GMEM arbiter (VX_CFG_NUM_DXA_CORES workers → kDxaMemPorts L2-facing).
  // Tag layout used by workers: high bit packs worker_id, low bits the
  // per-worker inflight slot. We pass TAG_SEL_IDX so the arb can route
  // responses back to the right input.
  char sname[100];
  snprintf(sname, 100, "%s-gmem-arb", name);
  gmem_arb_ = MemArbiter::Create(sname, ArbiterType::RoundRobin, VX_CFG_NUM_DXA_CORES, kDxaMemPorts);
  for (uint32_t i = 0; i < kDxaMemPorts; ++i) {
  #ifdef SIMX_SOCKET_UNIT_MMU
    // The unit MMU's lookup starts from a registered request: the DXA owns it.
    snprintf(sname, 100, "%s-gmem-out%d", name, i);
    auto out_reg = MemReqSlice::Create(sname, 1);
    gmem_arb_->ReqOut.at(i).bind(&out_reg->In);
    out_reg->Out.bind(&gmem_req_out.at(i));
    gmem_out_regs_.push_back(out_reg);
  #else
    gmem_arb_->ReqOut.at(i).bind(&gmem_req_out.at(i));
  #endif
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
