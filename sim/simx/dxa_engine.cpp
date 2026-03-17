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

#ifdef EXT_DXA_ENABLE

#include "dxa_engine.h"
#include <algorithm>
#include <limits>
#include "core.h"

namespace vortex {

namespace {

// ── Timing constants ────────────────────────────────────────────────
// These are approximate cycle costs used by the SimX model.  They do
// NOT need to be cycle-exact w.r.t. RTL; they just need to produce a
// similar-enough completion time for barrier-release ordering.

constexpr uint32_t kDecodeCycles        = 2;   // FSM decode + desc lookup
constexpr uint32_t kCompletionCycles    = 1;   // final done signaling

// ── g2s pipeline constants (line-granularity worker) ─────────────────
// Tile iterator emits 1 row/cycle.  rd_ctrl issues GMEM line reads with
// up to kMaxOutstanding concurrent slots.  wr_ctrl splits each GMEM line
// into SMEM words (RATIO_SPLIT = GMEM_LINE / SMEM_WORD writes per line).
constexpr uint32_t kSetupCycles         = 6;   // sequential multiply pipeline
constexpr uint32_t kMaxOutstanding      = 8;   // matches RTL MAX_OUTSTANDING
constexpr uint32_t kGmemReadLatency     = 8;   // L2 cache-line read latency (approximate)

// ── s2g serial constants (legacy path, not pipelined yet) ───────────
constexpr uint32_t kSmemReadCycles      = 2;
constexpr uint32_t kGmemWriteCycles     = 6;
constexpr uint32_t kElemIssueCycles     = 1;

inline uint64_t ceil_div(uint64_t n, uint64_t d) {
  return (n + d - 1) / d;
}

inline uint32_t desc_rank(uint32_t meta) {
  uint32_t r = (meta >> VX_DXA_DESC_META_DIM_LSB) & ((1u << VX_DXA_DESC_META_DIM_BITS) - 1u);
  if (r == 0) return 1;
  return std::min(r, 5u);
}

inline uint32_t desc_elem_bytes(uint32_t meta) {
  uint32_t enc = (meta >> VX_DXA_DESC_META_ELEMSZ_LSB) & ((1u << VX_DXA_DESC_META_ELEMSZ_BITS) - 1u);
  return 1u << enc;
}

} // namespace

// ════════════════════════════════════════════════════════════════════
// DxaEngine implementation
// ════════════════════════════════════════════════════════════════════

DxaEngine::DxaEngine(Core* core)
  : core_(core)
  , has_active_(false) {}

void DxaEngine::reset() {
  queue_.clear();
  has_active_ = false;
  active_xfer_ = ActiveTransfer();
}

int DxaEngine::dcr_write(uint32_t addr, uint32_t value) {
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
  default: break;
  }
  return 0;
}

const DxaEngine::Descriptor& DxaEngine::read_descriptor(uint32_t slot) const {
  return descriptors_.at(slot);
}

bool DxaEngine::build_copy_cfg(const Descriptor& desc, CopyCfg* cfg) const {
  uint32_t rank = desc_rank(desc.meta);
  if (rank > 2) {
    // Current phase supports rank-1 and rank-2 copy paths only.
    return false;
  }
  cfg->rank       = rank;
  cfg->elem_bytes = desc_elem_bytes(desc.meta);
  cfg->tile0      = std::max<uint32_t>(1, desc.tile_sizes[0]);
  cfg->tile1      = (rank >= 2) ? std::max<uint32_t>(1, desc.tile_sizes[1]) : 1u;
  return true;
}

bool DxaEngine::estimate_transfer(uint32_t slot, uint32_t* total_elems, uint32_t* elem_bytes) const {
  if (slot >= VX_DCR_DXA_DESC_COUNT)
    return false;
  const auto& desc = descriptors_.at(slot);
  CopyCfg cfg;
  if (!build_copy_cfg(desc, &cfg))
    return false;
  if (total_elems) *total_elems = cfg.tile0 * cfg.tile1;
  if (elem_bytes)  *elem_bytes  = cfg.elem_bytes;
  return true;
}

bool DxaEngine::execute_copy(uint32_t slot, uint32_t smem_addr, const uint32_t coords[5], uint32_t* bytes_copied) {
  if (slot >= VX_DCR_DXA_DESC_COUNT)
    return false;
  const auto& desc = descriptors_.at(slot);
  CopyCfg cfg;
  if (!build_copy_cfg(desc, &cfg))
    return false;

  uint64_t copied = 0;
  for (uint32_t y = 0; y < cfg.tile1; ++y) {
    for (uint32_t x = 0; x < cfg.tile0; ++x) {
      uint32_t i0 = coords[0] + x;
      uint32_t i1 = coords[1] + y;
      uint64_t saddr = uint64_t(smem_addr) + (uint64_t(y) * cfg.tile0 + x) * cfg.elem_bytes;
      bool in_bounds = (i0 < desc.sizes[0]) && ((cfg.rank < 2) || (i1 < desc.sizes[1]));
      if (in_bounds) {
        uint64_t gaddr = desc.base_addr + uint64_t(i0) * cfg.elem_bytes;
        if (cfg.rank >= 2) gaddr += uint64_t(i1) * uint64_t(desc.strides[0]);
        uint64_t value = 0;
        // always g2s (s2g path removed)
        core_->dcache_read(&value, gaddr, cfg.elem_bytes);
        core_->dcache_write(&value, saddr, cfg.elem_bytes);
      } else {
        uint64_t fill = desc.cfill;
        core_->dcache_write(&fill, saddr, cfg.elem_bytes);
      }
      copied += cfg.elem_bytes;
    }
  }

  if (bytes_copied) *bytes_copied = uint32_t(copied);
  return true;
}

bool DxaEngine::issue(uint32_t desc_slot,
                      uint32_t smem_addr,
                      const uint32_t coords[5],
                      uint32_t bar_id) {
  if (queue_.size() >= kQueueDepth) {
    return false;
  }

  Request req;
  req.desc_slot = desc_slot;
  req.smem_addr = smem_addr;
  req.bar_id = bar_id;
  for (uint32_t i = 0; i < req.coords.size(); ++i) {
    req.coords.at(i) = coords[i];
  }
  queue_.push_back(req);
  return true;
}

void DxaEngine::tick() {
  if (!has_active_) {
    if (!this->start_next_request()) {
      return;
    }
  }
  this->progress_active_request();
}

bool DxaEngine::start_next_request() {
  if (queue_.empty()) {
    return false;
  }

  auto req = queue_.front();
  queue_.pop_front();

  uint32_t total_elems = 0;
  uint32_t elem_bytes = 0;
  uint32_t total_cycles = 0;
  if (!this->decode_request(req, &total_elems, &elem_bytes, &total_cycles,
                            nullptr, nullptr, nullptr, nullptr, nullptr)) {
    core_->barrier_event_release(req.bar_id);
    return false;
  }

  active_xfer_.req = req;
  active_xfer_.total_elems = total_elems;
  active_xfer_.elem_bytes = elem_bytes;
  active_xfer_.cycles_left = std::max<uint32_t>(1, total_cycles);
  has_active_ = true;

  return true;
}

bool DxaEngine::decode_request(const Request& req,
                               uint32_t* total_elems,
                               uint32_t* elem_bytes,
                               uint32_t* total_cycles,
                               uint64_t* gmem_reads_out,
                               uint64_t* smem_writes_out,
                               uint64_t* gmem_rsp_blk_out,
                               uint64_t* gmem_req_blk_out,
                               uint64_t* smem_wr_blk_out) const {
  Core::DxaTransferInfo info = {};
  if (!core_->dxa_estimate(req.desc_slot, &info)) {
    return false;
  }

  bool is_s2g = false;  // Always g2s; s2g removed with flags
  uint64_t total = 0;
  uint64_t num_gmem_reads_v = 0;
  uint64_t num_smem_writes_v = 0;
  uint64_t gmem_rsp_blk_v = 0;
  uint64_t gmem_req_blk_v = 0;
  uint64_t smem_wr_blk_v = 0;

  if (is_s2g) {
    uint64_t per_elem = uint64_t(kElemIssueCycles + kSmemReadCycles + kGmemWriteCycles);
    total = kDecodeCycles + per_elem * info.total_elems + kCompletionCycles;
  } else {
    // g2s: line-granularity pipelined model with read dedup + write packing.
    uint64_t total_bytes = uint64_t(info.total_elems) * info.elem_bytes;
    uint32_t line_size = std::max<uint32_t>(1, L1_LINE_SIZE);
    uint32_t smem_word = std::max<uint32_t>(1, LMEM_NUM_BANKS * (XLEN / 8));
    uint32_t tile_line = info.tile0 * info.elem_bytes;  // T: bytes per tile row

    // ---- GMEM reads with LLB dedup ----
    // LLB dedup applies when T <= G (single GMEM line per row).
    // Count unique GMEM line addresses across all tile rows.
    if (tile_line <= line_size && info.stride0 > 0 && info.tile1 > 1) {
      // Simulate LLB: iterate rows, count address changes.
      uint64_t gmem_base_with_coord = info.gmem_base
          + uint64_t(req.coords[0]) * info.elem_bytes
          + uint64_t(req.coords[1]) * info.stride0;
      uint64_t prev_line = gmem_base_with_coord >> __builtin_ctz(line_size);
      num_gmem_reads_v = 1;  // first row always reads
      for (uint32_t r = 1; r < info.tile1; ++r) {
        uint64_t row_addr = gmem_base_with_coord + uint64_t(r) * info.stride0;
        uint64_t cur_line = row_addr >> __builtin_ctz(line_size);
        if (cur_line != prev_line) {
          ++num_gmem_reads_v;
          prev_line = cur_line;
        }
      }
    } else {
      // No dedup or multi-line rows: standard formula.
      num_gmem_reads_v = ceil_div(total_bytes, line_size);
    }

    // ---- SMEM writes with packing ----
    // Packed writes: consecutive data fills SMEM words densely.
    // Account for initial SMEM misalignment.
    uint32_t smem_off = req.smem_addr & (smem_word - 1);
    num_smem_writes_v = ceil_div(total_bytes + smem_off, smem_word);

    // GMEM read throughput: limited by outstanding slots.
    uint64_t excess = (num_gmem_reads_v > kMaxOutstanding) ? (num_gmem_reads_v - kMaxOutstanding) : 0;
    uint64_t read_stall_cycles = excess * ceil_div(kGmemReadLatency, kMaxOutstanding);

    // Drain: wait for last outstanding read.
    uint64_t drain_cycles = kGmemReadLatency;

    // Write throughput: 1 SMEM word per cycle.
    uint64_t write_cycles = num_smem_writes_v;

    // Pipeline: reads and writes overlap. Total ≈ setup + max(read, write) + drain.
    uint64_t read_total = read_stall_cycles + num_gmem_reads_v;
    uint64_t pipeline_cycles = std::max(read_total, write_cycles);

    total = kSetupCycles + kDecodeCycles + pipeline_cycles + drain_cycles + kCompletionCycles;

    // Breakdown: attribute stall cycles to bottleneck source.
    gmem_rsp_blk_v = drain_cycles;     // waiting for GMEM responses to drain
    gmem_req_blk_v = read_stall_cycles; // backpressure from outstanding limit
    if (write_cycles > read_total) {
      smem_wr_blk_v = write_cycles - read_total; // SMEM write was the bottleneck
    }
  }

  if (total_elems) *total_elems = info.total_elems;
  if (elem_bytes) *elem_bytes = info.elem_bytes;
  if (total_cycles) *total_cycles = uint32_t(std::min<uint64_t>(total, std::numeric_limits<uint32_t>::max()));
  if (gmem_reads_out) *gmem_reads_out = num_gmem_reads_v;
  if (smem_writes_out) *smem_writes_out = num_smem_writes_v;
  if (gmem_rsp_blk_out) *gmem_rsp_blk_out = gmem_rsp_blk_v;
  if (gmem_req_blk_out) *gmem_req_blk_out = gmem_req_blk_v;
  if (smem_wr_blk_out) *smem_wr_blk_out = smem_wr_blk_v;
  return true;
}

void DxaEngine::progress_active_request() {
  if (!has_active_) {
    return;
  }

  if (active_xfer_.cycles_left > 1) {
    --active_xfer_.cycles_left;
    return;
  }

  this->complete_active_request();
}

void DxaEngine::complete_active_request() {
  uint32_t bytes_copied = 0;
  (void)this->execute_copy(active_xfer_.req.desc_slot,
                           active_xfer_.req.smem_addr,
                           active_xfer_.req.coords.data(),
                           &bytes_copied);
  (void)bytes_copied;
  core_->barrier_event_release(active_xfer_.req.bar_id);

  has_active_ = false;
  active_xfer_ = ActiveTransfer();
}

} // namespace vortex

#endif // EXT_DXA_ENABLE

