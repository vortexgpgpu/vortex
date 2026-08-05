// Copyright © 2019-2026
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
#include <iostream>
#include <cstring>
#include <cassert>
#include <unordered_set>
#include <algorithm>

using namespace vortex;

namespace vt = vortex::tensor;

namespace {

constexpr uint64_t kLineMask = uint64_t(VX_CFG_L2_LINE_SIZE - 1);

using vt::elem_size_bytes;

inline uint64_t line_base(uint64_t addr) {
  return addr & ~kLineMask;
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
  , tag_alloc_(1)
  , pending_tag_(0)
{}

DtcuTma::~DtcuTma() {
  //--
}

void DtcuTma::reset() {
  pending_tag_ = 0;
  desc_addr_snapshot_ = 0;
  desc_lines_.clear();
  desc_data_.clear();
  out_req_lines_.clear();
  out_req_data_.clear();
  out_req_byteen_.clear();
  out_req_idx_ = 0;
  out_rsp_count_ = 0;
  tma_store_tags_.clear();
  tma_store_active_ = false;
  tma_store_accum_idx_ = 0;
  tma_store_m_ = 0;
  tma_store_n_ = 0;
  tma_store_baseD_ = 0;
  tma_store_accread_left_ = 0;
  tma_state_ = TmaState::IDLE;
  tma_req_lines_.clear();
  tma_req_idx_ = 0;
  tma_inflight_tags_.clear();
  tma_tag_line_.clear();
  tma_line_data_.clear();
  tma_target_buf_ = 0;
  tma_m_ = 0;
  tma_n_ = 0;
  tma_k_ = 0;
  tma_accum_ = 0;
  tma_fill_left_ = 0;
  tma_fill_acc_left_ = 0;
  tma_addrgen_left_ = 0;
}

// Drain all responses that arrived this cycle (multiple may be outstanding). TLM:
// load responses carry the cache-line payload (MemRsp.data) which we stash by line
// address for later assembly into the scratchpad; store responses just retire.
void DtcuTma::drain_responses() {
  while (!mem_rsp_in.empty()) {
    auto rsp = mem_rsp_in.peek();
    if (pending_tag_ != 0 && rsp.tag == pending_tag_) {
      // Descriptor line response.
      desc_data_[line_base(desc_addr_snapshot_)] = rsp.data;
      pending_tag_ = 0;
      mem_rsp_in.pop();
    } else if (tma_tag_line_.count(rsp.tag)) {
      // Operand prefetch line response: keep the bytes for the FILL assembly.
      uint64_t line = tma_tag_line_[rsp.tag];
      tma_line_data_[line] = rsp.data;
      tma_inflight_tags_.erase(rsp.tag);
      tma_tag_line_.erase(rsp.tag);
      mem_rsp_in.pop();
    } else if (tma_store_tags_.count(rsp.tag)) {
      // D-store acknowledgement: no payload, it only tells us the line has landed.
      tma_store_tags_.erase(rsp.tag);
      ++out_rsp_count_;
      mem_rsp_in.pop();
    } else {
      break; // unknown tag (should not happen) — avoid spinning
    }
  }
}

// Issue a TLM load for one cache line (no payload; data returns in the response).
void DtcuTma::issue_load_(uint64_t line_addr, std::unordered_set<uint64_t>& tagset,
                          std::unordered_map<uint64_t, uint64_t>* tag_line) {
  uint32_t tag = uint32_t(tag_alloc_++);
  MemReq req(MemOp::LD, line_addr);
  req.tag = tag;
  req.byteen = ~uint64_t(0);
  tagset.insert(tag);
  if (tag_line) (*tag_line)[tag] = line_addr;
  mem_req_out.send(req);
}

// Issue a TLM store for one cache line carrying its filled block + byte-enable.
void DtcuTma::issue_store_(uint64_t line_addr, const std::shared_ptr<mem_block_t>& data, uint64_t byteen) {
  uint32_t tag = uint32_t(tag_alloc_++);
  // Opt this store into a response (MemFlags::strsp, honoured by the cache's
  // need_core_rsp). Writes are otherwise fire-and-forget, which would leave the engine
  // unable to tell when D actually landed -- and the descriptor's completion flag has
  // to be ordered after that. drain_responses() counts the acks.
  MemReq req(MemOp::ST, line_addr, data, byteen, tag);
  req.flags.strsp = 1;
  tma_store_tags_.insert(tag);
  mem_req_out.send(req);
}

// Write the descriptor's completion flag. A single 4-byte masked store into the
// descriptor line -- byte-enable keeps the rest of the descriptor untouched, which
// matters because the caller may reuse it. Fire-and-forget: nothing is ordered after
// this, and its own visibility is what the consumer polls for.
void DtcuTma::issue_done_flag(uint64_t desc_addr) {
  const uint64_t addr = desc_addr + DTENSOR_DONE_OFFSET;
  const uint64_t line = line_base(addr);
  const uint32_t off  = uint32_t(addr - line);

  auto block = make_mem_block();
  std::memset(block->data(), 0, block->size());
  const uint32_t one = 1;
  std::memcpy(block->data() + off, &one, sizeof(one));

  uint64_t byteen = 0;
  for (uint32_t b = 0; b < sizeof(one); ++b)
    byteen |= (uint64_t(1) << (off + b));

  MemReq req(MemOp::ST, line, block, byteen, uint32_t(tag_alloc_++));
  mem_req_out.send(req);
}

// Read n bytes at a global address from a collected line map, spanning lines if needed.
void DtcuTma::read_from_lines_(const std::unordered_map<uint64_t, std::shared_ptr<mem_block_t>>& lines,
                               uint64_t addr, void* dst, uint32_t n) const {
  uint8_t* d = static_cast<uint8_t*>(dst);
  uint32_t done = 0;
  while (done < n) {
    uint64_t a = addr + done;
    uint64_t line = line_base(a);
    uint32_t off = uint32_t(a - line);
    uint32_t chunk = std::min(n - done, uint32_t(VX_CFG_L2_LINE_SIZE) - off);
    auto it = lines.find(line);
    assert(it != lines.end() && it->second && "DTCU TLM: line not fetched");
    std::memcpy(d + done, it->second->data() + off, chunk);
    done += chunk;
  }
}

// Descriptor fetch (single-outstanding). v3.0 descriptors are 64-byte aligned, so the
// 64-byte Desc lands in one cache line; the response payload carries it.
void DtcuTma::issue_desc_req(uint64_t desc_addr) {
  desc_addr_snapshot_ = desc_addr;
  desc_data_.clear();
  uint32_t tag = uint32_t(tag_alloc_++);
  pending_tag_ = tag;
  MemReq req(MemOp::LD, line_base(desc_addr));
  req.tag = tag;
  req.byteen = ~uint64_t(0);
  mem_req_out.send(req);
}

void DtcuTma::read_desc(uint64_t desc_addr) {
  // Assemble the descriptor from the fetched line payload (no ram_ backdoor).
  read_from_lines_(desc_data_, desc_addr, &dtcu_.desc_, sizeof(Dtcu::Desc));
  desc_data_.clear();
}

// Helper functions to calculate current tile's base addresses for A/B/C/D based on
// the current tile indices and descriptor (owned by the compute core).
uint64_t DtcuTma::calculate_base_A_(uint32_t k_idx) const {
  uint32_t in_sz = elem_size_bytes(dtcu_.desc_.fmt_s);
  uint64_t row = uint64_t(tma_m_) * dtcu_.tile_m_; // armed fetch's tile coordinate
  uint64_t col = uint64_t(k_idx) * dtcu_.tile_k_;
  return dtcu_.desc_.ptrA + (row * dtcu_.desc_.ldmA + col) * in_sz;
}

uint64_t DtcuTma::calculate_base_B_(uint32_t k_idx) const {
  uint32_t in_sz = elem_size_bytes(dtcu_.desc_.fmt_s);
  uint64_t row = uint64_t(k_idx) * dtcu_.tile_k_;
  uint64_t col = uint64_t(tma_n_) * dtcu_.tile_n_;
  return dtcu_.desc_.ptrB + (row + col * dtcu_.desc_.ldmB) * in_sz;
}

uint64_t DtcuTma::calculate_base_C_() const {
  uint32_t out_sz = elem_size_bytes(dtcu_.desc_.fmt_d);
  uint64_t row = uint64_t(tma_m_) * dtcu_.tile_m_;
  uint64_t col = uint64_t(tma_n_) * dtcu_.tile_n_;
  return dtcu_.desc_.ptrC + (row * dtcu_.desc_.ldmC + col) * out_sz;
}

uint64_t DtcuTma::calculate_base_D_() const {
  uint32_t out_sz = elem_size_bytes(dtcu_.desc_.fmt_d);
  uint64_t row = uint64_t(tma_store_m_) * dtcu_.tile_m_; // armed store's tile coordinate
  uint64_t col = uint64_t(tma_store_n_) * dtcu_.tile_n_;
  return dtcu_.desc_.ptrD + (row * dtcu_.desc_.ldmD + col) * out_sz;
}

// ------------------------------- ragged edges -------------------------------
// M/N/K need not be multiples of the native tile. An edge tile covers coordinates
// past the matrix; the engine clamps them out of the operand fetch (zero-filling the
// scratchpad, exactly what the DXA copy engine does with cfill) and masks them out of
// the D store so it never writes a byte the caller does not own.
//
// These predicates are shared by build_op_req_lines_ (which lines get REQUESTED) and
// load_operands_into (which addresses get READ). read_from_lines_ asserts on a line
// that was never requested, so the two MUST agree element for element.

bool DtcuTma::row_in_bounds_(uint32_t m_idx, uint32_t m) const {
  return uint64_t(m_idx) * dtcu_.tile_m_ + m < dtcu_.desc_.M;
}

bool DtcuTma::col_in_bounds_(uint32_t n_idx, uint32_t n) const {
  return uint64_t(n_idx) * dtcu_.tile_n_ + n < dtcu_.desc_.N;
}

// How many of the elements packed into K word *kw* of K tile *k_idx* are inside the
// matrix, 0..elems_per_word. A word is PARTIALLY valid when K is not a multiple of
// elems_per_word (e.g. fp16 packs 2 elements per word, so an odd K splits a word).
uint32_t DtcuTma::k_word_valid_elems_(uint32_t k_idx, uint32_t kw) const {
  const uint32_t epw = 4 / elem_size_bytes(dtcu_.desc_.fmt_s);
  const uint64_t k0  = uint64_t(k_idx) * dtcu_.tile_k_ + uint64_t(kw) * epw;
  if (k0 >= dtcu_.desc_.K)
    return 0;
  return uint32_t(std::min<uint64_t>(epw, dtcu_.desc_.K - k0));
}

// Zero the element lanes of *word* that lie past K. Every supported input format uses
// the all-zeros bit pattern for zero, so a masked lane contributes 0*0 = 0 to the dot
// product — which is why a K edge needs no store-side handling, unlike M/N.
static inline uint32_t mask_k_word(uint32_t word, uint32_t valid_elems, uint32_t in_sz) {
  const uint32_t bits = valid_elems * in_sz * 8;
  if (bits == 0)  return 0;
  if (bits >= 32) return word;
  return word & ((uint32_t(1) << bits) - 1);
}

// Assemble one K tile's operands (A/B and, on the first K tile, the C accumulator)
// from the fetched cache lines into the scratchpad. TLM: read words from the line
// payloads collected during FETCH, NOT from a ram_ backdoor.
void DtcuTma::load_operands_into(uint32_t buf_idx, uint32_t k_idx) {
  const Dtcu::Desc& desc = dtcu_.desc_;
  const uint32_t tile_m = dtcu_.tile_m_;
  const uint32_t tile_n = dtcu_.tile_n_;
  uint32_t in_sz = elem_size_bytes(desc.fmt_s);
  uint32_t elems_per_word = 4 / in_sz;

  // Initialize accumulator buffer on the first K tile (target from the kick).
  if (k_idx == 0) {
    auto& accum = dtcu_.accum_buf_[tma_accum_];
    if (desc.flags & 0x1) {
      std::fill(accum.begin(), accum.end(), 0.0f);
    } else {
      uint64_t baseC = calculate_base_C_();
      for (uint32_t m = 0; m < tile_m; ++m) {
        for (uint32_t n = 0; n < tile_n; ++n) {
          // Past the matrix: C was never fetched, so seed 0. The result is discarded
          // by the store mask anyway; zeroing keeps the accumulator deterministic.
          if (!row_in_bounds_(tma_m_, m) || !col_in_bounds_(tma_n_, n)) {
            accum[m * tile_n + n] = 0.0f;
            continue;
          }
          uint64_t addr = baseC + (uint64_t(m) * desc.ldmC + n) * 4;
          // Raw 4-byte copy into the accumulator slot: preserves the bit pattern for
          // both fp32 and int32 outputs.
          read_from_lines_(tma_line_data_, addr, &accum[m * tile_n + n], 4);
        }
      }
    }
  }

  // Load A buffer (row_major), same mapping as the kernel header.
  uint64_t baseA = calculate_base_A_(k_idx);
  auto& a_buf = dtcu_.shm_a_[buf_idx];
  for (uint32_t m = 0; m < tile_m; ++m) {
    const bool row_ok = row_in_bounds_(tma_m_, m);
    for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
      const uint32_t valid = k_word_valid_elems_(k_idx, kw);
      uint32_t word = 0;
      if (row_ok && valid != 0) {
        uint64_t addr = baseA + (uint64_t(m) * desc.ldmA + uint64_t(kw) * elems_per_word) * in_sz;
        read_from_lines_(tma_line_data_, addr, &word, 4);
        word = mask_k_word(word, valid, in_sz);
      }
      a_buf[m * DTCU_TILE_K_WORDS + kw] = word;
    }
  }

  // Load B buffer (col_major). Physical row stride is fixed at DTCU_TILE_N_MAX.
  uint64_t baseB = calculate_base_B_(k_idx);
  auto& b_buf = dtcu_.shm_b_[buf_idx];
  for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
    const uint32_t valid = k_word_valid_elems_(k_idx, kw);
    for (uint32_t n = 0; n < tile_n; ++n) {
      uint32_t word = 0;
      if (valid != 0 && col_in_bounds_(tma_n_, n)) {
        uint64_t addr = baseB + (uint64_t(kw) * elems_per_word + uint64_t(n) * desc.ldmB) * in_sz;
        read_from_lines_(tma_line_data_, addr, &word, 4);
        word = mask_k_word(word, valid, in_sz);
      }
      b_buf[kw * DTCU_TILE_N_MAX + n] = word;
    }
  }

  // Done with this tile's fetched lines.
  tma_line_data_.clear();
}

// --------------------- L2 timing model for memory traffic -------------------
// Compute which cache lines are touched by A/B/C/D, then issue one MemReq per
// unique cache line.

void DtcuTma::build_op_req_lines_(uint32_t k_idx, std::vector<uint64_t>& out_lines) {
  out_lines.clear();

  const Dtcu::Desc& desc = dtcu_.desc_;
  const uint32_t tile_m = dtcu_.tile_m_;
  const uint32_t tile_n = dtcu_.tile_n_;
  const uint32_t in_sz  = elem_size_bytes(desc.fmt_s);
  const uint32_t elems_per_word = 4 / in_sz;

  constexpr uint32_t WORD_BYTES = 4;

  std::vector<uint64_t> op_addrs;
  op_addrs.reserve(tile_m * DTCU_TILE_K_WORDS + DTCU_TILE_K_WORDS * tile_n + tile_m * tile_n);

  // A - row_major. Coordinates past the matrix are not fetched (see the ragged-edge
  // predicates above); load_operands_into zero-fills them instead.
  uint64_t baseA = calculate_base_A_(k_idx);
  for (uint32_t m = 0; m < tile_m; ++m) {
    if (!row_in_bounds_(tma_m_, m))
      continue;
    for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
      if (k_word_valid_elems_(k_idx, kw) == 0)
        continue;
      uint64_t addr = baseA + (uint64_t(m) * desc.ldmA + uint64_t(kw) * elems_per_word) * in_sz;
      op_addrs.push_back(addr);
    }
  }

  // B - col_major
  uint64_t baseB = calculate_base_B_(k_idx);
  for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
    if (k_word_valid_elems_(k_idx, kw) == 0)
      continue;
    for (uint32_t n = 0; n < tile_n; ++n) {
      if (!col_in_bounds_(tma_n_, n))
        continue;
      uint64_t addr = baseB + (uint64_t(kw) * elems_per_word + uint64_t(n) * desc.ldmB) * in_sz;
      op_addrs.push_back(addr);
    }
  }

  // C - row_major (only on the first K tile when accumulator is pre-loaded)
  if (k_idx == 0 && (desc.flags & 0x1) == 0) {
    uint64_t baseC = calculate_base_C_();
    for (uint32_t m = 0; m < tile_m; ++m) {
      if (!row_in_bounds_(tma_m_, m))
        continue;
      for (uint32_t n = 0; n < tile_n; ++n) {
        if (!col_in_bounds_(tma_n_, n))
          continue;
        uint64_t addr = baseC + (uint64_t(m) * desc.ldmC + n) * 4;
        op_addrs.push_back(addr);
      }
    }
  }

  coalesce_to_lines(op_addrs, WORD_BYTES, out_lines);
}

void DtcuTma::build_out_req_lines_(std::vector<uint64_t>& out_lines) {
  out_lines.clear();

  const Dtcu::Desc& desc = dtcu_.desc_;
  const uint32_t tile_m = dtcu_.tile_m_;
  const uint32_t tile_n = dtcu_.tile_n_;

  constexpr uint32_t WORD_BYTES = 4;

  std::vector<uint64_t> out_addrs;
  out_addrs.reserve(tile_m * tile_n);

  // D output (row_major) — snapshot base taken in start_store(). An edge tile's
  // coordinates past the matrix are dropped here and left byte-disabled in
  // build_store_payload_, so the store never touches memory outside D.
  uint64_t baseD = tma_store_baseD_;
  for (uint32_t m = 0; m < tile_m; ++m) {
    if (!row_in_bounds_(tma_store_m_, m))
      continue;
    for (uint32_t n = 0; n < tile_n; ++n) {
      if (!col_in_bounds_(tma_store_n_, n))
        continue;
      uint64_t addr = baseD + (uint64_t(m) * desc.ldmD + n) * 4;
      out_addrs.push_back(addr);
    }
  }

  coalesce_to_lines(out_addrs, WORD_BYTES, out_lines);
}

// Build the per-line ST payloads (cache-line block + byte-enable) from the snapshot
// accumulator buffer. Each D word is scattered into the line(s) it falls in.
void DtcuTma::build_store_payload_() {
  const uint32_t ldmD   = dtcu_.desc_.ldmD;
  const uint32_t tile_m = dtcu_.tile_m_;
  const uint32_t tile_n = dtcu_.tile_n_;
  const auto& accum = dtcu_.accum_buf_[tma_store_accum_idx_];

  // Map each output line addr to its slot in out_req_lines_/_data_/_byteen_.
  std::unordered_map<uint64_t, uint32_t> line_idx;
  line_idx.reserve(out_req_lines_.size() * 2);
  out_req_data_.assign(out_req_lines_.size(), nullptr);
  out_req_byteen_.assign(out_req_lines_.size(), 0);
  for (uint32_t i = 0; i < out_req_lines_.size(); ++i) {
    line_idx[out_req_lines_[i]] = i;
    out_req_data_[i] = make_mem_block();
    std::memset(out_req_data_[i]->data(), 0, out_req_data_[i]->size());
  }

  for (uint32_t m = 0; m < tile_m; ++m) {
    if (!row_in_bounds_(tma_store_m_, m))
      continue; // edge tile: no address was emitted, leave those bytes disabled
    for (uint32_t n = 0; n < tile_n; ++n) {
      if (!col_in_bounds_(tma_store_n_, n))
        continue;
      uint64_t addr = tma_store_baseD_ + (uint64_t(m) * ldmD + n) * 4;
      uint32_t bits;
      std::memcpy(&bits, &accum[m * tile_n + n], 4); // raw fp32/int32 bit pattern
      // Scatter the 4 bytes into their line(s), setting byte-enable.
      for (uint32_t b = 0; b < 4; ++b) {
        uint64_t a = addr + b;
        uint64_t line = line_base(a);
        uint32_t off = uint32_t(a - line);
        uint32_t i = line_idx[line];
        (*out_req_data_[i])[off] = uint8_t((bits >> (8 * b)) & 0xff);
        out_req_byteen_[i] |= (uint64_t(1) << off);
      }
    }
  }
}

void DtcuTma::start_prefetch(uint32_t buf_idx, uint32_t m_idx, uint32_t n_idx, uint32_t k_idx, uint32_t accum_idx) {
  tma_target_buf_ = buf_idx;
  tma_m_ = m_idx;
  tma_n_ = n_idx;
  tma_k_ = k_idx;
  tma_accum_ = accum_idx;
  dtcu_.buf_ready_[buf_idx] = false;
  build_op_req_lines_(k_idx, tma_req_lines_);
  tma_req_idx_ = 0;
  tma_line_data_.clear();
  tma_tag_line_.clear();
  tma_inflight_tags_.clear();
  dtcu_.total_op_reqs_ += tma_req_lines_.size();
  tma_addrgen_left_ = DTCU_ADDRGEN_CYCLES;
  tma_state_ = TmaState::ADDRGEN;
}

uint32_t DtcuTma::buffer_fill_cycles_(uint32_t k_idx) const {
  // Operand A+B fill into the operand scratchpad, at its bank count (fill and read hit
  // the same banked SRAM, 1 word/bank/cycle -- DTCU_SMEM_BANKS, one source of truth).
  const uint32_t op_words = dtcu_.tile_m_ * DTCU_TILE_K_WORDS + DTCU_TILE_K_WORDS * dtcu_.tile_n_;
  uint32_t cycles = (op_words + DTCU_SMEM_BANKS - 1) / DTCU_SMEM_BANKS;
  if (k_idx == 0) {
    cycles += fill_acc_cycles_(k_idx);
  }
  return cycles + DTCU_BUF_LATENCY;
}

// Accumulator-init share of the K0 fill: writes accum_buf_ -- a SEPARATE SRAM from
// the operand scratchpad, draining at DTCU_ACC_BANKS. Counted as tma_acc_init.
uint32_t DtcuTma::fill_acc_cycles_(uint32_t k_idx) const {
  if (k_idx != 0)
    return 0;
  const uint32_t acc_words = dtcu_.tile_m_ * dtcu_.tile_n_;
  return (acc_words + DTCU_ACC_BANKS - 1) / DTCU_ACC_BANKS;
}

// Advance the engine by one cycle. A single shared L2 port issues at most one request
// per cycle: the load (operand-prefetch) channel has priority; the output-store
// channel uses the port only when the load channel did not. Loads are bounded by
// DTCU_MAX_OUTSTANDING (responses retire in drain_responses()); stores are
// fire-and-forget and bounded only by the port and the request queue.
void DtcuTma::tick() {
  bool port_used = false;

  // ---- Load channel (operand prefetch) ----
  switch (tma_state_) {
  case TmaState::IDLE:
    break;
  case TmaState::ADDRGEN:
    if (tma_addrgen_left_ > 0) {
      --tma_addrgen_left_;
      ++dtcu_.tma_addrgen_cycles_;
    } else {
      tma_state_ = TmaState::FETCH;
    }
    break;
  case TmaState::FETCH: {
    uint32_t inflight = tma_inflight_tags_.size();
    if (tma_req_idx_ < tma_req_lines_.size()
        && inflight < DTCU_MAX_OUTSTANDING
        && !mem_req_out.full()) {
      issue_load_(tma_req_lines_[tma_req_idx_], tma_inflight_tags_, &tma_tag_line_);
      ++tma_req_idx_;
      port_used = true;
    } else if (!tma_inflight_tags_.empty()) {
      ++dtcu_.tma_mem_wait_cycles_;
    }
    if (tma_req_idx_ >= tma_req_lines_.size() && tma_inflight_tags_.empty()) {
      tma_fill_left_ = buffer_fill_cycles_(tma_k_);
      tma_fill_acc_left_ = fill_acc_cycles_(tma_k_);
      tma_state_ = TmaState::FILL;
    }
    break;
  }
  case TmaState::FILL:
    if (tma_fill_left_ > 0) {
      --tma_fill_left_;
      if (tma_fill_acc_left_ > 0) { // attribute the acc-init share first
        --tma_fill_acc_left_;
        ++dtcu_.tma_acc_init_cycles_;
      } else {
        ++dtcu_.tma_op_fill_cycles_;
      }
    } else {
      // Assemble the buffers from the fetched line payloads, then mark ready.
      load_operands_into(tma_target_buf_, tma_k_);
      dtcu_.buf_ready_[tma_target_buf_] = true;
      tma_state_ = TmaState::IDLE;
    }
    break;
  }

  // ---- Store channel (output D write-back, background, load-priority) ----
  if (tma_store_active_) {
    // acc-SRAM read streams in parallel with the L2 writes (separate resource).
    if (tma_store_accread_left_ > 0)
      --tma_store_accread_left_;

    if (!port_used && out_req_idx_ < out_req_lines_.size() && !mem_req_out.full()) {
      issue_store_(out_req_lines_[out_req_idx_], out_req_data_[out_req_idx_], out_req_byteen_[out_req_idx_]);
      ++out_req_idx_;
    } else if (out_req_idx_ < out_req_lines_.size()) {
      ++dtcu_.tma_store_issue_stall_cycles_;
    }
    // Done only when every D line has been ACKNOWLEDGED (not merely issued) and the acc
    // read has drained, so the store lasts max(acc read, mem write + ack) cycles. The
    // ack condition is what lets the caller order the completion flag after the data.
    if (out_rsp_count_ >= out_req_lines_.size() && tma_store_accread_left_ == 0) {
      out_req_data_.clear();
      out_req_byteen_.clear();
      tma_store_active_ = false;
    }
  }
}

// Hand off output tile (m_idx, n_idx)'s D store to the store channel. Snapshots the
// base address + accumulator payload NOW (build_store_payload_ copies the bytes), so
// the accumulator buffer is immediately reusable -- the cross-tile lookahead's
// C-preload relies on this. TODO: the acc-SRAM port conflict between this store's
// modeled acc read and a concurrent lookahead C-preload write is not modeled.
void DtcuTma::start_store(uint32_t accum_idx, uint32_t m_idx, uint32_t n_idx) {
  tma_store_accum_idx_ = accum_idx;
  tma_store_m_ = m_idx;
  tma_store_n_ = n_idx;
  tma_store_baseD_ = calculate_base_D_(); // snapshot (armed coordinates)
  build_out_req_lines_(out_req_lines_);
  dtcu_.total_out_reqs_ += out_req_lines_.size();
  build_store_payload_();
  out_req_idx_ = 0;
  out_rsp_count_ = 0;
  tma_store_tags_.clear();
  // Reading the tile out of the accumulator SRAM to feed the store: tile_m*tile_n fp32
  // words at DTCU_ACC_BANKS words/cycle. A separate resource from the L2 store port, so
  // it streams alongside the memory writes -- store completes at max(acc read, mem write).
  tma_store_accread_left_ = (dtcu_.tile_m_ * dtcu_.tile_n_ + DTCU_ACC_BANKS - 1) / DTCU_ACC_BANKS;
  tma_store_active_ = true;
}
