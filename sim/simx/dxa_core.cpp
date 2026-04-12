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
#include <deque>
#include <unordered_set>
#include <vector>
#include "core.h"
#include "cluster.h"
#include "debug.h"
#include "constants.h"

using namespace vortex;

// Number of DXA GMEM output ports toward L2: min(NUM_DXA_UNITS, L2_NUM_REQS).
// Mirrors DXA_MEM_PORTS = min($NUM_DXA_UNITS, up($NUM_CORES/$SOCKET_SIZE)*$L1_MEM_PORTS)
// from VX_config.toml, computed locally so VX_config.h need not be regenerated.
static constexpr uint32_t kDxaMemPorts = std::min<uint32_t>(NUM_DXA_UNITS, L2_NUM_REQS);

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

  // ── Copy geometry ────────────────────────────────────────────────────
  struct CopyCfg {
    uint32_t rank;
    uint32_t elem_bytes;
    std::array<uint32_t, 5> tiles = {};  // tile sizes per dimension (unused dims = 1)
  };

  // ── Per-request context ──────────────────────────────────────────────
  struct Request {
    Core*                   core = nullptr;
    DxaCore::TraceData::Ptr td;
  };

  // ── Slice state machine ──────────────────────────────────────────────
  enum class SliceState { IDLE, GMEM_FETCH, SMEM_WRITE };

  struct ActiveTransfer {
    Request    req;
    CopyCfg    cfg;
    // GMEM_FETCH — addresses precomputed by emulation, replayed here
    std::vector<uint64_t> gmem_lines;  // deduped cache-line addresses (from exe_data)
    uint32_t gmem_send_idx = 0;        // next line to issue
    uint32_t gmem_pending  = 0;        // issued, no rsp yet
    uint64_t issue_cycle   = 0;
    // SMEM_WRITE (timing only; data already written by execute_copy)
    uint32_t smem_block_idx = 0;       // next XLENB-aligned SMEM word signal to send
    // Multicast replay state
    bool     is_multicast = false;
    std::vector<uint32_t> cta_indices; // CTA warp indices from cta_mask
    uint32_t mc_cta_idx   = 0;        // current CTA being replayed
    uint32_t smem_stride  = 0;        // byte stride between CTAs' SMEM regions
  };

  struct DxaSlice {
    SliceState     state     = SliceState::IDLE;
    uint32_t       slice_idx = 0;
    ActiveTransfer active_xfer;
  };

  static constexpr uint32_t kQueueDepth    = 8;
  static constexpr uint32_t kMaxOutstanding = 8;

  // ── Constructor ──────────────────────────────────────────────────────
  explicit Impl(DxaCore* simobject, MemArbiter* arb)
    : simobject_(simobject)
    , arb_(arb)
    , slices_(NUM_DXA_UNITS)
    , cycle_(0)
  {
    for (uint32_t i = 0; i < NUM_DXA_UNITS; ++i)
      slices_[i].slice_idx = i;
  }

  // ── Public methods ───────────────────────────────────────────────────

  void reset() {
    queue_.clear();
    cycle_ = 0;
    perf_stats_ = DxaCore::PerfStats();
    for (auto& s : slices_) {
      s.state = SliceState::IDLE;
      s.active_xfer = ActiveTransfer();
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

  bool submit(Core* core, DxaCore::TraceData::Ptr td) {
    if (queue_.size() >= kQueueDepth)
      return false;
    queue_.push_back({core, td});
    DT(4, simobject_->name() << " submit: core=" << core->id()
       << ", slot=" << td->desc_slot << ", bar=" << td->bar_id);
    return true;
  }

  // Emulation only: build GMEM address list, read tile, write to SMEM.
  // Returns TraceData with routing fields pre-filled and emulation fields computed.
  // Always returns non-null; invalid descriptor → tile0=tile1=0, no gmem/smem blocks
  // (start_slice() releases the barrier via the zero-blocks path).
  DxaCore::TraceData::Ptr execute_copy_pub(Core* core, uint32_t desc_slot,
                                           uint64_t smem_addr,
                                           const uint32_t coords[5]) {
    auto exe_data       = std::make_shared<DxaCore::TraceData>();
    exe_data->desc_slot = desc_slot;
    exe_data->smem_addr = smem_addr;
    std::copy(coords, coords + 5, exe_data->coords);

    if (desc_slot >= VX_DCR_DXA_DESC_COUNT)
      return exe_data; // invalid slot — zero tile/blocks, bar released in start_slice
    const auto& desc = descriptors_.at(desc_slot);
    CopyCfg cfg;
    if (!build_copy_cfg(desc, &cfg))
      return exe_data; // invalid config — same

    exe_data->tile0      = cfg.tiles[0];
    exe_data->tile1      = cfg.tiles[1];
    exe_data->elem_bytes = cfg.elem_bytes;

    // Build GMEM cache-line address list matching RTL VX_dxa_addr_gen + VX_dxa_dedup behavior.
    // RTL addr_gen iterates "rows" (innermost tile0 strips) sequentially over all outer dims.
    // Each row is a contiguous tile0*elem_bytes span in GMEM; outer dims advance via strides.
    // RTL dedup merges consecutive same-CL entries across the entire transfer.
    uint32_t line_size = std::max<uint32_t>(1, L1_LINE_SIZE);
    uint64_t line_mask = ~uint64_t(line_size - 1);
    uint64_t global_prev_cl = ~uint64_t(0);

    // Total "rows" = product of tile sizes for dims 1..rank-1.
    uint32_t total_rows = 1;
    for (uint32_t d = 1; d < cfg.rank; ++d)
      total_rows *= cfg.tiles[d];

    for (uint32_t row = 0; row < total_rows; ++row) {
      // Decompose row into outer-dim local offsets (dim1 varies fastest).
      uint32_t outer[4] = {};
      uint32_t rem = row;
      for (uint32_t d = 1; d < cfg.rank; ++d) {
        outer[d - 1] = rem % cfg.tiles[d];
        rem /= cfg.tiles[d];
      }

      // Check OOB for outer dims.
      bool row_in_bounds = true;
      for (uint32_t d = 1; d < cfg.rank; ++d) {
        if (coords[d] + outer[d - 1] >= desc.sizes[d]) {
          row_in_bounds = false;
          break;
        }
      }
      if (!row_in_bounds) continue; // OOB row: RTL uses cfill, no GMEM reads

      // First element GMEM address in this row.
      uint64_t row_gbase = desc.base_addr + uint64_t(coords[0]) * cfg.elem_bytes;
      for (uint32_t d = 1; d < cfg.rank; ++d)
        row_gbase += uint64_t(coords[d] + outer[d - 1]) * uint64_t(desc.strides[d - 1]);

      // CL range for this row.
      uint64_t row_first_cl = row_gbase & line_mask;
      uint64_t row_last_cl  = (row_gbase + uint64_t(cfg.tiles[0]) * cfg.elem_bytes - 1) & line_mask;
      for (uint64_t cl = row_first_cl; cl <= row_last_cl; cl += line_size) {
        if (cl != global_prev_cl) {
          exe_data->gmem_lines.push_back(cl);
          global_prev_cl = cl;
        } else {
          ++exe_data->gmem_dedup_hits;
        }
      }
    }

    // Build SMEM write block list: DXA_SMEM_WORD_SIZE-aligned addresses, sorted.
    constexpr uint32_t word_size = LMEM_NUM_BANKS * XLENB;
    uint64_t word_mask = ~uint64_t(word_size - 1);
    std::unordered_set<uint64_t> smem_seen;
    uint32_t total_elems = 1;
    for (uint32_t d = 0; d < cfg.rank; ++d)
      total_elems *= cfg.tiles[d];
    for (uint32_t lin = 0; lin < total_elems; ++lin) {
      uint64_t saddr = smem_addr + uint64_t(lin) * cfg.elem_bytes;
      for (uint64_t b = saddr & word_mask; b < saddr + cfg.elem_bytes; b += word_size)
        smem_seen.insert(b);
    }
    exe_data->smem_blocks.assign(smem_seen.begin(), smem_seen.end());
    std::sort(exe_data->smem_blocks.begin(), exe_data->smem_blocks.end());

    execute_copy({core, exe_data}, desc, cfg);
    return exe_data;
  }

  // Emulation only — no SimChannel activity.
  void gmem_read(Core* core, uint64_t addr, void* data, uint32_t size) {
    core->mem_read(data, addr, size);
  }

  void tick() {
    ++cycle_;

    // Drain GMEM responses routed back by the internal MemArbiter (one channel per slice).
    for (uint32_t sidx = 0; sidx < slices_.size(); ++sidx) {
      auto& rsp_ch = arb_->RspOut.at(sidx);
      while (!rsp_ch.empty()) {
        rsp_ch.pop();
        --slices_[sidx].active_xfer.gmem_pending;
      }
    }

    // Advance active slices (output before input — reverse pipeline order).
    for (auto& slice : slices_) {
      if (slice.state != SliceState::IDLE)
        tick_slice(slice);
    }

    // Dispatch pending requests to newly-idle slices.
    for (auto& slice : slices_) {
      if (slice.state == SliceState::IDLE && !queue_.empty())
        start_slice(slice);
    }
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

  bool build_copy_cfg(const Descriptor& desc, CopyCfg* cfg) const {
    uint32_t rank = desc_rank(desc.meta);
    if (rank < 1 || rank > 5) return false;
    cfg->rank       = rank;
    cfg->elem_bytes = desc_elem_bytes(desc.meta);
    for (uint32_t d = 0; d < 5; ++d)
      cfg->tiles[d] = (d < rank) ? std::max<uint32_t>(1, desc.tile_sizes[d]) : 1u;
    return true;
  }

  // ── Functional copy — emulation only, no timing side effects ─────────
  // Called once at GMEM→SMEM transition so data is in LocalMem RAM
  // before the first timing signal is sent on lmem_req_out.
  // Supports 1D-5D with multicast (cta_mask replication).
  void execute_copy(const Request& req, const Descriptor& desc, const CopyCfg& cfg) {
    auto& td = *req.td;
    // Compute total elements and iterate with a flat linear index.
    uint32_t total = 1;
    for (uint32_t d = 0; d < cfg.rank; ++d)
      total *= cfg.tiles[d];

    // Determine which CTAs to write to (multicast).
    uint32_t cta_mask = td.cta_mask;
    bool is_multicast = __builtin_popcount(cta_mask) > 1;
    // Build list of CTA indices from cta_mask.
    std::vector<uint32_t> cta_indices;
    if (is_multicast) {
      for (uint32_t w = 0; w < 32; ++w) {
        if (cta_mask & (1u << w))
          cta_indices.push_back(w);
      }
    }

    for (uint32_t lin = 0; lin < total; ++lin) {
      // Decompose linear index into per-dimension local offsets (row-major).
      // local[0] varies fastest.
      uint32_t local[5] = {};
      uint32_t rem = lin;
      for (uint32_t d = 0; d < cfg.rank; ++d) {
        local[d] = rem % cfg.tiles[d];
        rem /= cfg.tiles[d];
      }

      // Compute GMEM address.
      uint32_t idx[5];
      bool in_bounds = true;
      for (uint32_t d = 0; d < cfg.rank; ++d) {
        idx[d] = td.coords[d] + local[d];
        if (idx[d] >= desc.sizes[d])
          in_bounds = false;
      }

      uint64_t gaddr = desc.base_addr + uint64_t(idx[0]) * cfg.elem_bytes;
      for (uint32_t d = 1; d < cfg.rank; ++d)
        gaddr += uint64_t(idx[d]) * uint64_t(desc.strides[d - 1]);

      // SMEM address: dense row-major packing.
      uint64_t saddr_base = td.smem_addr + uint64_t(lin) * cfg.elem_bytes;

      uint64_t value = 0;
      if (in_bounds) {
        gmem_read(req.core, gaddr, &value, cfg.elem_bytes);
      } else {
        value = desc.cfill;
      }

      if (is_multicast) {
        // Replicate to each CTA's SMEM region.
        for (uint32_t cta_idx : cta_indices) {
          uint64_t saddr = saddr_base + uint64_t(cta_idx) * desc.smem_stride;
          req.core->mem_write(&value, saddr, cfg.elem_bytes);
        }
      } else {
        req.core->mem_write(&value, saddr_base, cfg.elem_bytes);
      }
    }
  }

  // ── Slice lifecycle ──────────────────────────────────────────────────

  bool start_slice(DxaSlice& slice) {
    if (queue_.empty()) return false;

    auto req = queue_.front();
    queue_.pop_front();

    // Replay precomputed geometry and addresses from emulation trace.
    auto& td            = *req.td;
    auto& xfer          = slice.active_xfer;
    xfer.req            = req;
    xfer.cfg.tiles[0]   = td.tile0;
    xfer.cfg.tiles[1]   = td.tile1;
    xfer.cfg.elem_bytes = td.elem_bytes;
    xfer.cfg.rank       = 0; // unused in timing path
    xfer.gmem_lines     = td.gmem_lines;
    xfer.gmem_send_idx  = 0;
    xfer.gmem_pending   = 0;
    xfer.smem_block_idx = 0;
    xfer.issue_cycle    = cycle_;

    // Multicast setup.
    uint32_t cta_mask = td.cta_mask;
    xfer.is_multicast = (__builtin_popcount(cta_mask) > 1);
    xfer.cta_indices.clear();
    xfer.mc_cta_idx = 0;
    if (xfer.is_multicast) {
      for (uint32_t w = 0; w < 32; ++w) {
        if (cta_mask & (1u << w))
          xfer.cta_indices.push_back(w);
      }
      if (td.desc_slot < VX_DCR_DXA_DESC_COUNT)
        xfer.smem_stride = descriptors_.at(td.desc_slot).smem_stride;
    }

    perf_stats_.gmem_reads  += xfer.gmem_lines.size();
    perf_stats_.gmem_dedup  += td.gmem_dedup_hits;
    slice.state = xfer.gmem_lines.empty() ? SliceState::SMEM_WRITE
                                          : SliceState::GMEM_FETCH;

    DT(3, simobject_->name() << " start: core=" << req.core->id()
       << ", slot=" << td.desc_slot
       << ", gmem_lines=" << xfer.gmem_lines.size()
       << ", total_elems=" << td.tile0 * td.tile1
       << ", multicast=" << xfer.is_multicast
       << ", num_ctas=" << xfer.cta_indices.size());
    return true;
  }

  void tick_slice(DxaSlice& slice) {
    auto&    xfer = slice.active_xfer;
    uint32_t sidx = slice.slice_idx;

    // ── GMEM_FETCH ───────────────────────────────────────────────────
    if (slice.state == SliceState::GMEM_FETCH) {
      // Issue reads across shared ports (round-robin).
      while (xfer.gmem_send_idx < xfer.gmem_lines.size()
             && xfer.gmem_pending < kMaxOutstanding) {
        MemReq mreq;
        mreq.addr  = xfer.gmem_lines[xfer.gmem_send_idx];
        mreq.write = false;
        mreq.type  = AddrType::Global;
        mreq.tag   = 0; // arbiter encodes slice index into tag for response routing
        mreq.cid   = 0;
        mreq.uuid  = 0;
        // Send to this slice's arbiter input; arbiter routes to an available L2 port.
        if (!arb_->ReqIn.at(sidx).try_send(mreq))
          break; // arbiter input full — backpressure
        ++xfer.gmem_send_idx;
        ++xfer.gmem_pending;
      }

      // Transition when all lines sent and all responses received.
      if (xfer.gmem_send_idx == xfer.gmem_lines.size()
          && xfer.gmem_pending == 0) {
        // Data already in LocalMem RAM — execute_copy() ran at issue time.
        slice.state = SliceState::SMEM_WRITE;
        DT(4, simobject_->name() << "[" << sidx << "] gmem fetch done");
      }
      return;
    }

    // ── SMEM_WRITE (timing only — data already in LocalMem RAM) ──────
    // Emits one DxaSmemReq per XLENB-aligned SMEM word block, matching
    // RTL VX_dxa_cl2smem.sv word-granular SMEM writes with byte enables.
    // For multicast, each block is replayed for each CTA before advancing.
    if (slice.state == SliceState::SMEM_WRITE) {
      const auto& smem_blocks = xfer.req.td->smem_blocks;
      uint32_t total_blocks = smem_blocks.size();
      uint32_t num_ctas = xfer.is_multicast ? xfer.cta_indices.size() : 1;
      uint32_t total_writes = total_blocks * num_ctas;

      // Check if all writes are done.
      if (xfer.smem_block_idx >= total_blocks) {
        // All timing signals sent.
        if (total_blocks == 0) {
          // Edge case: no SMEM blocks — release barrier(s) directly.
          if (xfer.is_multicast) {
            for (uint32_t cta_idx : xfer.cta_indices)
              xfer.req.core->barrier_event_release(xfer.req.td->bar_id + cta_idx);
          } else {
            xfer.req.core->barrier_event_release(xfer.req.td->bar_id);
          }
        }
        uint64_t latency = cycle_ - xfer.issue_cycle;
        ++perf_stats_.transfers;
        perf_stats_.total_latency += latency;
        perf_stats_.lmem_writes   += total_writes;
        DT(3, simobject_->name() << "[" << sidx << "] complete: core="
           << xfer.req.core->id() << ", bar=" << xfer.req.td->bar_id
           << ", smem_blocks=" << total_blocks << ", latency=" << latency);
        slice.state = SliceState::IDLE;
        xfer = ActiveTransfer();
        return;
      }

      uint32_t cores_per_cluster = NUM_SOCKETS * SOCKET_SIZE;
      uint32_t lmem_idx = xfer.req.core->id() % cores_per_cluster;
      if (simobject_->lmem_req_out.at(lmem_idx).full())
        return; // backpressure from LocalMem

      bool is_last_block = (xfer.smem_block_idx + 1 == total_blocks);

      if (xfer.is_multicast) {
        // Multicast: replay current block to current CTA.
        uint32_t cta_warp_idx = xfer.cta_indices[xfer.mc_cta_idx];
        DxaCore::SmemReq sreq;
        sreq.addr    = smem_blocks[xfer.smem_block_idx]
                     + uint64_t(cta_warp_idx) * xfer.smem_stride;
        sreq.size    = XLENB;
        sreq.bar_id  = xfer.req.td->bar_id + cta_warp_idx; // bar_stride = 1
        sreq.core    = xfer.req.core;
        sreq.is_last = is_last_block; // per-CTA last triggers that CTA's barrier
        simobject_->lmem_req_out.at(lmem_idx).send(sreq);

        ++xfer.mc_cta_idx;
        if (xfer.mc_cta_idx >= num_ctas) {
          xfer.mc_cta_idx = 0;
          ++xfer.smem_block_idx;
        }
      } else {
        // Non-multicast: one write per block.
        DxaCore::SmemReq sreq;
        sreq.addr    = smem_blocks[xfer.smem_block_idx];
        sreq.size    = XLENB;
        sreq.bar_id  = xfer.req.td->bar_id;
        sreq.core    = xfer.req.core;
        sreq.is_last = is_last_block;
        simobject_->lmem_req_out.at(lmem_idx).send(sreq);
        ++xfer.smem_block_idx;
      }
    }
  }

  // ── Members ──────────────────────────────────────────────────────────
  DxaCore*     simobject_;
  MemArbiter*  arb_;           // NUM_DXA_UNITS inputs → kDxaMemPorts outputs (owned by DxaCore)
  std::array<Descriptor, VX_DCR_DXA_DESC_COUNT> descriptors_;
  std::deque<Request>    queue_;
  std::vector<DxaSlice>  slices_;
  uint64_t               cycle_;
  DxaCore::PerfStats     perf_stats_;
};

// ════════════════════════════════════════════════════════════════════
// DxaCore — delegation wrappers
// ════════════════════════════════════════════════════════════════════

DxaCore::DxaCore(const SimContext& ctx, const char* name, Cluster* cluster)
  : SimObject(ctx, name)
  , gmem_req_out(kDxaMemPorts, this)
  , gmem_rsp_in(kDxaMemPorts, this)
  , lmem_req_out(NUM_SOCKETS * SOCKET_SIZE, this)
{
  __unused(cluster);
  // Create arbiter: NUM_DXA_UNITS slice inputs → kDxaMemPorts L2-facing outputs.
  char sname[100];
  snprintf(sname, 100, "%s-arb", name);
  arb_ = MemArbiter::Create(sname, ArbiterType::RoundRobin, NUM_DXA_UNITS, kDxaMemPorts);
  // Chain arbiter outputs through DxaCore's external GMEM channels.
  for (uint32_t i = 0; i < kDxaMemPorts; ++i) {
    arb_->ReqOut.at(i).bind(&gmem_req_out.at(i));
    gmem_rsp_in.at(i).bind(&arb_->RspIn.at(i));
  }
  impl_ = new Impl(this, arb_.get());
}

DxaCore::~DxaCore() {
  delete impl_;
}

void DxaCore::reset() {
  impl_->reset();
}

void DxaCore::tick() {
  impl_->tick();
}

int DxaCore::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

DxaCore::TraceData::Ptr DxaCore::execute_copy(Core* core, uint32_t desc_slot,
                                               uint64_t smem_addr,
                                               const uint32_t coords[5]) {
  return impl_->execute_copy_pub(core, desc_slot, smem_addr, coords);
}

bool DxaCore::submit(Core* core, TraceData::Ptr td) {
  return impl_->submit(core, td);
}

void DxaCore::gmem_read(Core* core, uint64_t addr, void* data, uint32_t size) {
  impl_->gmem_read(core, addr, data, size);
}

const DxaCore::PerfStats& DxaCore::perf_stats() const {
  return impl_->perf_stats();
}
