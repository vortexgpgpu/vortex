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
#include <limits>
#include "core.h"
#include "cluster.h"
#include "debug.h"
#include "constants.h"

using namespace vortex;

namespace {

// ── Timing constants ────────────────────────────────────────────────
// Approximate cycle costs used by the countdown timing model.
// These intentionally match dxa_engine.cpp values so behaviour is
// unchanged; the Phase-2 L2 SimChannel path will replace them.

constexpr uint32_t kDecodeCycles        = 2;
constexpr uint32_t kCompletionCycles    = 1;

constexpr uint32_t kSetupCycles         = 6;
constexpr uint32_t kMaxOutstanding      = 8;
constexpr uint32_t kGmemReadLatency     = 8;

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
// DxaCore
// ════════════════════════════════════════════════════════════════════

DxaCore::DxaCore(const SimContext& ctx, const char* name, Cluster* cluster)
  : SimObject(ctx, name)
  , cluster_(cluster)
  , slices_(NUM_DXA_UNITS)
  , cycle_(0)
  , perf_stats_()
{}

void DxaCore::reset() {
  queue_.clear();
  cycle_ = 0;
  perf_stats_ = PerfStats();
  for (auto& slice : slices_) {
    slice.has_active = false;
    slice.active_xfer = ActiveTransfer();
  }
}

int DxaCore::dcr_write(uint32_t addr, uint32_t value) {
  uint32_t slot = VX_DCR_DXA_DESC_SLOT(addr);
  uint32_t word = VX_DCR_DXA_DESC_WORD(addr);
  auto& d = descriptors_.at(slot);
  switch (word) {
  case VX_DCR_DXA_DESC_BASE_LO_OFF:    d.base_addr = (d.base_addr & 0xffffffff00000000ull) | value; break;
  case VX_DCR_DXA_DESC_BASE_HI_OFF:    d.base_addr = (d.base_addr & 0x00000000ffffffffull) | (uint64_t(value) << 32); break;
  case VX_DCR_DXA_DESC_SIZE0_OFF:      d.sizes[0] = value; break;
  case VX_DCR_DXA_DESC_SIZE1_OFF:      d.sizes[1] = value; break;
  case VX_DCR_DXA_DESC_SIZE2_OFF:      d.sizes[2] = value; break;
  case VX_DCR_DXA_DESC_SIZE3_OFF:      d.sizes[3] = value; break;
  case VX_DCR_DXA_DESC_SIZE4_OFF:      d.sizes[4] = value; break;
  case VX_DCR_DXA_DESC_STRIDE0_OFF:    d.strides[0] = value; break;
  case VX_DCR_DXA_DESC_STRIDE1_OFF:    d.strides[1] = value; break;
  case VX_DCR_DXA_DESC_STRIDE2_OFF:    d.strides[2] = value; break;
  case VX_DCR_DXA_DESC_STRIDE3_OFF:    d.strides[3] = value; break;
  case VX_DCR_DXA_DESC_META_OFF:       d.meta = value; break;
  case VX_DCR_DXA_DESC_ESTRIDE0_OFF:   d.element_strides[0] = value; break;
  case VX_DCR_DXA_DESC_ESTRIDE1_OFF:   d.element_strides[1] = value; break;
  case VX_DCR_DXA_DESC_ESTRIDE2_OFF:   d.element_strides[2] = value; break;
  case VX_DCR_DXA_DESC_ESTRIDE3_OFF:   d.element_strides[3] = value; break;
  case VX_DCR_DXA_DESC_ESTRIDE4_OFF:   d.element_strides[4] = value; break;
  case VX_DCR_DXA_DESC_TILESIZE01_OFF:
    d.tile_sizes[0] = uint16_t(value & 0xffff);
    d.tile_sizes[1] = uint16_t(value >> 16);
    break;
  case VX_DCR_DXA_DESC_TILESIZE23_OFF:
    d.tile_sizes[2] = uint16_t(value & 0xffff);
    d.tile_sizes[3] = uint16_t(value >> 16);
    break;
  case VX_DCR_DXA_DESC_TILESIZE4_OFF:  d.tile_sizes[4] = uint16_t(value & 0xffff); break;
  case VX_DCR_DXA_DESC_CFILL_OFF:      d.cfill = value; break;
  default: break;
  }
  return 0;
}

bool DxaCore::submit(Core* core,
                     uint32_t desc_slot,
                     uint32_t smem_addr,
                     const uint32_t coords[5],
                     uint32_t bar_id) {
  if (queue_.size() >= kQueueDepth)
    return false;
  Request req;
  req.core      = core;
  req.desc_slot = desc_slot;
  req.smem_addr = smem_addr;
  req.bar_id    = bar_id;
  for (uint32_t i = 0; i < 5; ++i)
    req.coords[i] = coords[i];
  queue_.push_back(req);
  DT(4, this->name() << " submit: core=" << core->id()
     << " slot=" << desc_slot << " bar=" << bar_id);
  return true;
}

void DxaCore::tick() {
  ++cycle_;

  // Dispatch pending requests to idle slices.
  for (auto& slice : slices_) {
    if (!slice.has_active && !queue_.empty()) {
      start_slice(slice);
    }
  }

  // Advance all active slices.
  for (auto& slice : slices_) {
    if (slice.has_active) {
      tick_slice(slice);
    }
  }
}

bool DxaCore::start_slice(DxaSlice& slice) {
  if (queue_.empty())
    return false;

  auto req = queue_.front();
  queue_.pop_front();

  uint32_t total_elems = 0, elem_bytes = 0, total_cycles = 0;
  uint64_t gmem_reads = 0, smem_writes = 0;
  if (!decode_request(req, &total_elems, &elem_bytes, &total_cycles,
                      &gmem_reads, &smem_writes)) {
    // Descriptor invalid – release barrier immediately and skip.
    DT(3, this->name() << " invalid descriptor slot=" << req.desc_slot
       << ", releasing bar=" << req.bar_id);
    req.core->barrier_event_release(req.bar_id);
    return false;
  }

  slice.active_xfer.req          = req;
  slice.active_xfer.total_elems  = total_elems;
  slice.active_xfer.elem_bytes   = elem_bytes;
  slice.active_xfer.cycles_left  = std::max<uint32_t>(1, total_cycles);
  slice.active_xfer.issue_cycle  = cycle_;
  slice.has_active               = true;

  perf_stats_.gmem_reads  += gmem_reads;
  perf_stats_.smem_writes += smem_writes;

  DT(3, this->name() << " start: core=" << req.core->id()
     << " slot=" << req.desc_slot
     << " cycles=" << total_cycles
     << " gmem_reads=" << gmem_reads
     << " smem_writes=" << smem_writes);
  return true;
}

void DxaCore::tick_slice(DxaSlice& slice) {
  if (!slice.has_active)
    return;

  if (slice.active_xfer.cycles_left > 1) {
    --slice.active_xfer.cycles_left;
    return;
  }

  // Transfer complete: execute functional copy then release barrier.
  execute_copy(slice.active_xfer.req);
  slice.active_xfer.req.core->barrier_event_release(slice.active_xfer.req.bar_id);

  uint64_t latency = cycle_ - slice.active_xfer.issue_cycle;
  ++perf_stats_.transfers;
  perf_stats_.total_latency += latency;

  DT(3, this->name() << " complete: core=" << slice.active_xfer.req.core->id()
     << " bar=" << slice.active_xfer.req.bar_id
     << " latency=" << latency);

  slice.has_active = false;
  slice.active_xfer = ActiveTransfer();
}

// ────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────

bool DxaCore::build_copy_cfg(const Descriptor& desc, CopyCfg* cfg) const {
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

bool DxaCore::decode_request(const Request& req,
                              uint32_t* total_elems,
                              uint32_t* elem_bytes,
                              uint32_t* total_cycles,
                              uint64_t* gmem_reads_out,
                              uint64_t* smem_writes_out) const {
  if (req.desc_slot >= VX_DCR_DXA_DESC_COUNT)
    return false;
  const auto& d = descriptors_.at(req.desc_slot);
  CopyCfg cfg;
  if (!build_copy_cfg(d, &cfg))
    return false;

  const uint32_t total_elems_v = cfg.tile0 * cfg.tile1;
  const uint32_t elem_bytes_v  = cfg.elem_bytes;
  const uint32_t tile0         = cfg.tile0;
  const uint32_t tile1         = cfg.tile1;
  const uint32_t stride0       = (cfg.rank >= 2) ? d.strides[0] : 0;
  const uint64_t gmem_base     = d.base_addr;

  uint64_t total = 0;
  uint64_t num_gmem_reads_v  = 0;
  uint64_t num_smem_writes_v = 0;

  // g2s: line-granularity pipelined model.
  {
    uint64_t total_bytes = uint64_t(total_elems_v) * elem_bytes_v;
    uint32_t line_size   = std::max<uint32_t>(1, L1_LINE_SIZE);
    uint32_t smem_word   = std::max<uint32_t>(1, LMEM_NUM_BANKS * (XLEN / 8));
    uint32_t tile_line   = tile0 * elem_bytes_v;

    // GMEM reads with LLB dedup.
    if (tile_line <= line_size && stride0 > 0 && tile1 > 1) {
      uint64_t base_off = gmem_base
          + uint64_t(req.coords[0]) * elem_bytes_v
          + uint64_t(req.coords[1]) * stride0;
      uint64_t prev_line = base_off >> __builtin_ctz(line_size);
      num_gmem_reads_v = 1;
      for (uint32_t r = 1; r < tile1; ++r) {
        uint64_t row_addr = base_off + uint64_t(r) * stride0;
        uint64_t cur_line = row_addr >> __builtin_ctz(line_size);
        if (cur_line != prev_line) {
          ++num_gmem_reads_v;
          prev_line = cur_line;
        }
      }
    } else {
      num_gmem_reads_v = ceil_div(total_bytes, line_size);
    }

    // SMEM writes with packing.
    uint32_t smem_off      = req.smem_addr & (smem_word - 1);
    num_smem_writes_v      = ceil_div(total_bytes + smem_off, smem_word);

    // Throughput: reads limited by MSHR outstanding slots.
    uint64_t excess            = (num_gmem_reads_v > kMaxOutstanding)
                                 ? (num_gmem_reads_v - kMaxOutstanding) : 0;
    uint64_t read_stall_cycles = excess * ceil_div(kGmemReadLatency, kMaxOutstanding);
    uint64_t drain_cycles      = kGmemReadLatency;
    uint64_t write_cycles      = num_smem_writes_v;
    uint64_t read_total        = read_stall_cycles + num_gmem_reads_v;
    uint64_t pipeline_cycles   = std::max(read_total, write_cycles);

    total = kSetupCycles + kDecodeCycles + pipeline_cycles + drain_cycles + kCompletionCycles;
    __unused(kSmemReadCycles);
    __unused(kGmemWriteCycles);
    __unused(kElemIssueCycles);
  }

  if (total_elems)    *total_elems    = total_elems_v;
  if (elem_bytes)     *elem_bytes     = elem_bytes_v;
  if (total_cycles)   *total_cycles   = uint32_t(std::min<uint64_t>(
                                          total, std::numeric_limits<uint32_t>::max()));
  if (gmem_reads_out)  *gmem_reads_out  = num_gmem_reads_v;
  if (smem_writes_out) *smem_writes_out = num_smem_writes_v;
  return true;
}

bool DxaCore::execute_copy(const Request& req) {
  if (req.desc_slot >= VX_DCR_DXA_DESC_COUNT)
    return false;
  const auto& desc = descriptors_.at(req.desc_slot);
  CopyCfg cfg;
  if (!build_copy_cfg(desc, &cfg))
    return false;

  for (uint32_t y = 0; y < cfg.tile1; ++y) {
    for (uint32_t x = 0; x < cfg.tile0; ++x) {
      uint32_t i0 = req.coords[0] + x;
      uint32_t i1 = req.coords[1] + y;
      uint64_t saddr = uint64_t(req.smem_addr)
                     + (uint64_t(y) * cfg.tile0 + x) * cfg.elem_bytes;
      bool in_bounds = (i0 < desc.sizes[0])
                    && ((cfg.rank < 2) || (i1 < desc.sizes[1]));
      if (in_bounds) {
        uint64_t gaddr = desc.base_addr + uint64_t(i0) * cfg.elem_bytes;
        if (cfg.rank >= 2)
          gaddr += uint64_t(i1) * uint64_t(desc.strides[0]);
        uint64_t value = 0;
        req.core->dcache_read(&value, gaddr, cfg.elem_bytes);
        req.core->dcache_write(&value, saddr, cfg.elem_bytes);
      } else {
        uint64_t fill = desc.cfill;
        req.core->dcache_write(&fill, saddr, cfg.elem_bytes);
      }
    }
  }
  return true;
}