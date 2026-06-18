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

#include "dtcu_tma.h"
#include "dtcu_params.h"
#include "types.h"
#include "tensor_cfg.h"
#include <cstring>
#include <cassert>
#include <unordered_set>
#include <algorithm>

using namespace vortex;

namespace vt = vortex::tensor;

namespace {

constexpr uint32_t DTCU_TILE_K_WORDS = 8;

inline uint32_t elem_size_bytes(uint32_t fmt_id) {
  switch (fmt_id) {
    case vt::fp32::id:  return 4;
    case vt::fp16::id:  return 2;
    case vt::bf16::id:  return 2;
    case vt::int32::id: return 4;
    case vt::int8::id:  return 1;
    case vt::uint8::id: return 1;
    case vt::int4::id:  return 1;
    case vt::uint4::id: return 1;
    default:            return 4;
  }
}

inline uint64_t line_base(uint64_t addr) {
  return addr & ~uint64_t(L2_LINE_SIZE - 1);
}

// Similar to mem_coalescer: same line is combined, unaligned accesses split into
// two lines. Preserves first-touch order.
inline void coalesce_to_lines(const std::vector<uint64_t>& addrs, uint32_t bytes, std::vector<uint64_t>& out_lines) {
  std::unordered_set<uint64_t> seen_lines;
  seen_lines.reserve(addrs.size() * 2);

  for (auto addr : addrs) {
    uint64_t l0 = line_base(addr);
    uint64_t l1 = line_base(addr + bytes - 1);

    if (seen_lines.insert(l0).second) {
      out_lines.push_back(l0);
    }

    if (l1 != l0 && seen_lines.insert(l1).second) {
      out_lines.push_back(l1);
    }
  }
}

} // namespace

DtcuTma::DtcuTma(Dtcu& parent)
  : mem_req_out(&parent)
  , mem_rsp_in(&parent)
  , dtcu_(parent)
  , ram_(nullptr)
  , tag_alloc_(1)
  , pending_tag_(0)
{}

DtcuTma::~DtcuTma() {
  //--
}

void DtcuTma::reset() {
  pending_tag_ = 0;
  out_req_lines_.clear();
  out_req_idx_ = 0;
  tma_store_inflight_tags_.clear();
  tma_store_active_ = false;
  tma_store_accum_idx_ = 0;
  tma_store_baseD_ = 0;
  tma_state_ = TmaState::IDLE;
  tma_req_lines_.clear();
  tma_req_idx_ = 0;
  tma_inflight_tags_.clear();
  tma_target_buf_ = 0;
  tma_k_ = 0;
  tma_fill_left_ = 0;
  tma_addrgen_left_ = 0;
}

// Drain all responses that have arrived this cycle (multiple may be outstanding).
void DtcuTma::drain_responses() {
  while (!mem_rsp_in.empty()) {
    auto rsp = mem_rsp_in.peek();
    if (rsp.tag == pending_tag_) {
      mem_rsp_in.pop();
      pending_tag_ = 0;
    } else if (tma_inflight_tags_.count(rsp.tag)) {
      mem_rsp_in.pop();
      tma_inflight_tags_.erase(rsp.tag);
    } else if (tma_store_inflight_tags_.count(rsp.tag)) {
      mem_rsp_in.pop();
      tma_store_inflight_tags_.erase(rsp.tag);
    } else {
      break; // unknown tag (should not happen) — avoid spinning
    }
  }
}

void DtcuTma::issue_mem_req(uint64_t addr, bool write) {
  MemReq req;
  req.addr  = addr;
  req.write = write;
  req.tag   = tag_alloc_++;
  req.cid   = 0;
  req.uuid  = 0;

  // Track both read and write requests until MemRsp arrives.
  pending_tag_ = req.tag;

  // 1-cycle latency for memory access
  // For same-level interconnect, other files (cache_sim) use a 1-cycle latency for the entire request-response round trip too.
  mem_req_out.send(req);
}

// Same as issue_mem_req but tracks the request under a prefetch tag, so a prefetch
// can be in flight independently of the main (descriptor/output) path.
void DtcuTma::issue_mem_req_tma_(uint64_t addr, bool write) {
  MemReq req;
  req.addr  = addr;
  req.write = write;
  req.tag   = tag_alloc_++;
  req.cid   = 0;
  req.uuid  = 0;
  tma_inflight_tags_.insert(req.tag);
  mem_req_out.send(req);
}

// Output-store write, tracked under the store channel's tag set so it is in flight
// independently of the descriptor and load paths.
void DtcuTma::issue_mem_req_store_(uint64_t addr) {
  MemReq req;
  req.addr  = addr;
  req.write = true;
  req.tag   = tag_alloc_++;
  req.cid   = 0;
  req.uuid  = 0;
  tma_store_inflight_tags_.insert(req.tag);
  mem_req_out.send(req);
}

void DtcuTma::read_desc(uint64_t desc_addr) {
  assert(ram_ && "RAM must be attached before DTCU use");
  ram_->read(&dtcu_.desc_, desc_addr, sizeof(Dtcu::Desc));
}

// Helper functions to calculate current tile's base addresses for A/B/C/D based on
// the current tile indices and descriptor (owned by the compute core).
uint64_t DtcuTma::calculate_base_A_(uint32_t k_idx) const {
  uint32_t in_sz = elem_size_bytes(dtcu_.desc_.fmt_s);
  uint64_t row = uint64_t(dtcu_.tile_m_idx_) * dtcu_.tile_m_;
  uint64_t col = uint64_t(k_idx) * dtcu_.tile_k_;
  return dtcu_.desc_.ptrA + (row * dtcu_.desc_.ldmA + col) * in_sz;
}

uint64_t DtcuTma::calculate_base_B_(uint32_t k_idx) const {
  uint32_t in_sz = elem_size_bytes(dtcu_.desc_.fmt_s);
  uint64_t row = uint64_t(k_idx) * dtcu_.tile_k_;
  uint64_t col = uint64_t(dtcu_.tile_n_idx_) * dtcu_.tile_n_;
  return dtcu_.desc_.ptrB + (row + col * dtcu_.desc_.ldmB) * in_sz;
}

uint64_t DtcuTma::calculate_base_C_() const {
  uint32_t out_sz = elem_size_bytes(dtcu_.desc_.fmt_d);
  uint64_t row = uint64_t(dtcu_.tile_m_idx_) * dtcu_.tile_m_;
  uint64_t col = uint64_t(dtcu_.tile_n_idx_) * dtcu_.tile_n_;
  return dtcu_.desc_.ptrC + (row * dtcu_.desc_.ldmC + col) * out_sz;
}

uint64_t DtcuTma::calculate_base_D_() const {
  uint32_t out_sz = elem_size_bytes(dtcu_.desc_.fmt_d);
  uint64_t row = uint64_t(dtcu_.tile_m_idx_) * dtcu_.tile_m_;
  uint64_t col = uint64_t(dtcu_.tile_n_idx_) * dtcu_.tile_n_;
  return dtcu_.desc_.ptrD + (row * dtcu_.desc_.ldmD + col) * out_sz;
}

void DtcuTma::load_operands_into(uint32_t buf_idx, uint32_t k_idx) {
  const Dtcu::Desc& desc = dtcu_.desc_;
  const uint32_t tile_m = dtcu_.tile_m_;
  const uint32_t tile_n = dtcu_.tile_n_;
  uint32_t in_sz = elem_size_bytes(desc.fmt_s);
  uint32_t elems_per_word = 4 / in_sz;

  // Initialize Accumulators Buffer on the first K tile
  if (k_idx == 0) {
    auto& accum = dtcu_.accum_buf_[dtcu_.accum_compute_idx_];
    if (desc.flags & 0x1) {
      // No pre-load for accumulator
      std::fill(accum.begin(), accum.end(), 0.0f);
    } else {
      // Pre-loaded accumulator
      uint64_t baseC = calculate_base_C_();
      for (uint32_t m = 0; m < tile_m; ++m) {
        for (uint32_t n = 0; n < tile_n; ++n) {
          uint64_t addr = baseC + (uint64_t(m) * desc.ldmC + n) * 4;
          float value = 0.0f;
          ram_->read(&value, addr, 4);
          accum[m * tile_n + n] = value;
        }
      }
    }
  }

  // Load A Buffer (row_major), same mapping as kernel/include/vx_tensor.h
  uint64_t baseA = calculate_base_A_(k_idx);
  auto& a_buf = dtcu_.a_buf_[buf_idx];
  for (uint32_t m = 0; m < tile_m; ++m) {
    for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
      uint64_t addr = baseA + (uint64_t(m) * desc.ldmA + uint64_t(kw) * elems_per_word) * in_sz;
      uint32_t word = 0;
      ram_->read(&word, addr, 4);
      a_buf[m * DTCU_TILE_K_WORDS + kw] = word;
    }
  }

  // Load B Buffer (col_major)
  uint64_t baseB = calculate_base_B_(k_idx);
  auto& b_buf = dtcu_.b_buf_[buf_idx];
  for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
    for (uint32_t n = 0; n < tile_n; ++n) {
      uint64_t addr = baseB + (uint64_t(kw) * elems_per_word + uint64_t(n) * desc.ldmB) * in_sz;
      uint32_t word = 0;
      ram_->read(&word, addr, 4);
      b_buf[kw * tile_n + n] = word;
    }
  }
}

// --------------------- L2 timing model for memory traffic -------------------
// Compute which cache lines are touched by A/B/C/D, then issue one MemReq per
// unique cache line.

// Build the operand (A/B/C) cache-line request list for a given K tile.
void DtcuTma::build_op_req_lines_(uint32_t k_idx, std::vector<uint64_t>& out_lines) {
  out_lines.clear();

  const Dtcu::Desc& desc = dtcu_.desc_;
  const uint32_t tile_m = dtcu_.tile_m_;
  const uint32_t tile_n = dtcu_.tile_n_;
  const uint32_t in_sz  = elem_size_bytes(desc.fmt_s);
  const uint32_t elems_per_word = 4 / in_sz;

  // Match current RAM access granularity
  constexpr uint32_t WORD_BYTES = 4;

  std::vector<uint64_t> op_addrs;
  op_addrs.reserve(tile_m * DTCU_TILE_K_WORDS + DTCU_TILE_K_WORDS * tile_n + tile_m * tile_n);

  // A - row_major
  uint64_t baseA = calculate_base_A_(k_idx);
  for (uint32_t m = 0; m < tile_m; ++m) {
    for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
      uint64_t addr = baseA + (uint64_t(m) * desc.ldmA + uint64_t(kw) * elems_per_word) * in_sz;
      op_addrs.push_back(addr);
    }
  }

  // B - col_major
  uint64_t baseB = calculate_base_B_(k_idx);
  for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
    for (uint32_t n = 0; n < tile_n; ++n) {
      uint64_t addr = baseB + (uint64_t(kw) * elems_per_word + uint64_t(n) * desc.ldmB) * in_sz;
      op_addrs.push_back(addr);
    }
  }

  // C - row_major (only on the first K tile when accumulator is pre-loaded)
  if (k_idx == 0 && (desc.flags & 0x1) == 0) {
    uint64_t baseC = calculate_base_C_();
    for (uint32_t m = 0; m < tile_m; ++m) {
      for (uint32_t n = 0; n < tile_n; ++n) {
        uint64_t addr = baseC + (uint64_t(m) * desc.ldmC + n) * 4;
        op_addrs.push_back(addr);
      }
    }
  }

  // Coalesce while preserving first-touch order
  coalesce_to_lines(op_addrs, WORD_BYTES, out_lines);
}

// Build the output (D) cache-line request list for the current output tile.
void DtcuTma::build_out_req_lines_(std::vector<uint64_t>& out_lines) {
  out_lines.clear();

  const Dtcu::Desc& desc = dtcu_.desc_;
  const uint32_t tile_m = dtcu_.tile_m_;
  const uint32_t tile_n = dtcu_.tile_n_;

  constexpr uint32_t WORD_BYTES = 4;

  std::vector<uint64_t> out_addrs;
  out_addrs.reserve(tile_m * tile_n);

  // D output (row_major)
  uint64_t baseD = calculate_base_D_();
  for (uint32_t m = 0; m < tile_m; ++m) {
    for (uint32_t n = 0; n < tile_n; ++n) {
      uint64_t addr = baseD + (uint64_t(m) * desc.ldmD + n) * 4;
      out_addrs.push_back(addr);
    }
  }

  coalesce_to_lines(out_addrs, WORD_BYTES, out_lines);
}

// Start prefetching one K tile's operands (A/B and, on the first K tile, C) into
// the given buffer. Builds the cache-line request list and arms the load channel.
void DtcuTma::start_prefetch(uint32_t buf_idx, uint32_t k_idx) {
  tma_target_buf_ = buf_idx;
  tma_k_ = k_idx;
  dtcu_.buf_ready_[buf_idx] = false;
  build_op_req_lines_(k_idx, tma_req_lines_);
  tma_req_idx_ = 0;
  dtcu_.total_op_reqs_ += tma_req_lines_.size();
  tma_addrgen_left_ = DTCU_ADDRGEN_CYCLES;
  tma_state_ = TmaState::ADDRGEN;
}

// Cycles to write one K tile's fetched data into the operand buffers (A+B), plus
// the accumulator init on the first K tile. Models banked scratchpad write BW.
uint32_t DtcuTma::buffer_fill_cycles_(uint32_t k_idx) const {
  uint32_t words = dtcu_.tile_m_ * DTCU_TILE_K_WORDS + DTCU_TILE_K_WORDS * dtcu_.tile_n_; // A + B
  if (k_idx == 0) {
    words += dtcu_.tile_m_ * dtcu_.tile_n_; // accumulator init (C-load or zero) writes accum_buf_
  }
  return (words + DTCU_BUF_BW - 1) / DTCU_BUF_BW + DTCU_BUF_LATENCY;
}

// Advance the engine by one cycle. A single shared L2 port issues at most one
// request per cycle: the load (operand-prefetch) channel has priority and the
// output-store channel uses the port only when the load channel did not. Both share
// the DTCU_MAX_OUTSTANDING budget; responses retire in drain_responses().
void DtcuTma::tick() {
  bool port_used = false; // a request was sent to mem_req_out this cycle

  // ---- Load channel (operand prefetch) ----
  switch (tma_state_) {
  case TmaState::IDLE:
    break;
  case TmaState::ADDRGEN:
    // AGU per-tile address + cache-line-list setup (per-tile latency).
    if (tma_addrgen_left_ > 0) {
      --tma_addrgen_left_;
      ++dtcu_.tma_addrgen_cycles_;
    } else {
      tma_state_ = TmaState::FETCH;
    }
    break;
  case TmaState::FETCH: {
    // Multiple-outstanding: issue up to one request/cycle while under the outstanding
    // limit and lines remain. L2 cache_sim models bank contention among them.
    uint32_t inflight = tma_inflight_tags_.size() + tma_store_inflight_tags_.size();
    if (tma_req_idx_ < tma_req_lines_.size()
        && inflight < DTCU_MAX_OUTSTANDING
        && !mem_req_out.full()) { // respect L2 input backpressure (channel full → stall)
      issue_mem_req_tma_(tma_req_lines_[tma_req_idx_], false);
      ++tma_req_idx_;
      port_used = true;
    } else if (!tma_inflight_tags_.empty()) {
      ++dtcu_.tma_mem_wait_cycles_; // backpressured or all issued; waiting on responses
    }
    if (tma_req_idx_ >= tma_req_lines_.size() && tma_inflight_tags_.empty()) {
      // All operand lines fetched: spend buffer-write (SRAM fill) cycles.
      tma_fill_left_ = buffer_fill_cycles_(tma_k_);
      tma_state_ = TmaState::FILL;
    }
    break;
  }
  case TmaState::FILL:
    if (tma_fill_left_ > 0) {
      --tma_fill_left_;
      ++dtcu_.tma_buffer_write_cycles_;
    } else {
      // Functional fill from RAM, then mark the buffer ready.
      load_operands_into(tma_target_buf_, tma_k_);
      dtcu_.buf_ready_[tma_target_buf_] = true;
      tma_state_ = TmaState::IDLE;
    }
    break;
  }

  // ---- Store channel (output D write-back, background, load-priority) ----
  if (tma_store_active_) {
    uint32_t inflight = tma_inflight_tags_.size() + tma_store_inflight_tags_.size();
    if (!port_used && out_req_idx_ < out_req_lines_.size()
        && inflight < DTCU_MAX_OUTSTANDING && !mem_req_out.full()) {
      issue_mem_req_store_(out_req_lines_[out_req_idx_]);
      ++out_req_idx_;
    } else if (out_req_idx_ < out_req_lines_.size() || !tma_store_inflight_tags_.empty()) {
      ++dtcu_.tma_store_wait_cycles_; // port taken by load / budget full / waiting on responses
    }
    if (out_req_idx_ >= out_req_lines_.size() && tma_store_inflight_tags_.empty()) {
      // All store lines written back: do the functional store, then free the channel.
      store_output();
      tma_store_active_ = false;
    }
  }
}

// Hand off the current output tile's D store to the store channel. Builds the D
// cache-line list and snapshots the tile's base address + accumulator buffer NOW,
// because the compute core advances its tile indices for the next tile while this
// store is still draining in the background.
void DtcuTma::start_store(uint32_t accum_idx) {
  build_out_req_lines_(out_req_lines_);
  dtcu_.total_out_reqs_ += out_req_lines_.size();
  out_req_idx_ = 0;
  tma_store_accum_idx_ = accum_idx;
  tma_store_baseD_ = calculate_base_D_(); // snapshot (current tile indices)
  tma_store_active_ = true;
}

// Functional write-back of the stored accumulator buffer, using the snapshot taken
// at start_store() (tile indices may have since advanced for the next tile).
void DtcuTma::store_output() {
  const uint32_t ldmD = dtcu_.desc_.ldmD;
  const uint32_t tile_m = dtcu_.tile_m_;
  const uint32_t tile_n = dtcu_.tile_n_;
  const auto& accum = dtcu_.accum_buf_[tma_store_accum_idx_];
  for (uint32_t m = 0; m < tile_m; ++m) {
    for (uint32_t n = 0; n < tile_n; ++n) {
      uint64_t addr = tma_store_baseD_ + (uint64_t(m) * ldmD + n) * 4;
      float value = accum[m * tile_n + n];
      ram_->write(&value, addr, 4);
    }
  }
}
