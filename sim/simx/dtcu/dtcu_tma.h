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

#pragma once

#include <simobject.h>
#include "types.h"
#include "mem_block_pool.h"
#include "dtcu.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>

namespace vortex {

// DtcuTma: the DTCU's tensor-memory engine. It owns the single L2 memory port and
// performs ALL of the DTCU's memory movement -- descriptor fetch, operand (A/B/C)
// prefetch into the scratchpad, and output (D) store-back -- including per-tile
// address generation and cache-line coalescing.
//
// v3.0 TLM data path (NO ram_ backdoor): a load MemReq carries no payload; its
// MemRsp.data (a shared_ptr<mem_block_t> cache line) carries the fetched bytes,
// which we copy into the scratchpad. A store MemReq carries a filled mem_block_t +
// byte-enable mask, which the modeled memory hierarchy writes. Data and timing thus
// travel the same path (the postdoc's "no shortcut" goal).
//
// Plain helper class (not a SimObject): the owning Dtcu drives it from on_tick().
class DtcuTma {
public:
  // Memory port (public so the Cluster can bind it to the L2 arbiter).
  SimChannel<MemReq> mem_req_out;
  SimChannel<MemRsp> mem_rsp_in;

  explicit DtcuTma(Dtcu& parent);
  ~DtcuTma();

  void reset();

  // Drain all memory responses that arrived this cycle (call once per tick).
  void drain_responses();

  // Descriptor fetch path is single-outstanding: true when its response is back.
  bool main_done() const { return pending_tag_ == 0; }

  // Descriptor fetch: issue the load request, then assemble the descriptor from the
  // returned cache line(s) once the response(s) are back.
  void issue_desc_req(uint64_t desc_addr);
  void read_desc(uint64_t desc_addr);

  // Operand prefetch (load channel): arm a K tile, advance one cycle, query state.
  void start_prefetch(uint32_t buf_idx, uint32_t k_idx);
  void tick();
  bool load_idle() const { return tma_state_ == TmaState::IDLE; }

  // Output store (store channel): hand off the current tile's D store; it then runs
  // in the background inside tick() (multiple-outstanding, lower priority than the
  // load channel) and writes the accumulator back to memory when complete.
  void start_store(uint32_t accum_idx);
  bool store_active() const { return tma_store_active_; }
  bool store_idle() const { return !tma_store_active_; }

private:
  // TMA prefetch sub-engine state (loads one K tile's operands into a buffer).
  enum class TmaState {
    IDLE,
    ADDRGEN, // AGU computes per-tile addresses + cache-line list (per-tile setup)
    FETCH,   // issue operand cache-line requests (multiple-outstanding) + retire responses
    FILL     // writing fetched lines into the operand/accumulator buffer (SRAM)
  };

  Dtcu& dtcu_;   // back-reference to the owning compute core (scratchpad + geometry)

  uint64_t tag_alloc_;
  uint64_t pending_tag_; // descriptor-fetch single-outstanding request tag

  // Descriptor fetch: the line(s) covering the 64-byte descriptor, captured from the
  // response payload, then assembled in read_desc().
  uint64_t desc_addr_snapshot_ = 0;
  std::vector<uint64_t> desc_lines_;                              // line addrs requested
  std::unordered_map<uint64_t, std::shared_ptr<mem_block_t>> desc_data_; // line -> bytes

  // Store channel (output D write-back): runs in the background, overlapped with
  // the next tile's prefetch/compute. Multiple-outstanding, shares the outstanding
  // budget with the load channel but yields the port to it (load priority).
  std::vector<uint64_t> out_req_lines_;
  std::vector<std::shared_ptr<mem_block_t>> out_req_data_;   // per-line ST payload
  std::vector<uint64_t> out_req_byteen_;                     // per-line byte-enable mask
  uint32_t out_req_idx_ = 0;
  std::unordered_set<uint64_t> tma_store_inflight_tags_; // outstanding store-write tags
  bool     tma_store_active_ = false;
  uint32_t tma_store_accum_idx_ = 0;
  uint64_t tma_store_baseD_ = 0;
  uint32_t tma_store_accread_left_ = 0; // acc-SRAM read to feed the store, at DTCU_ACC_BANKS/cyc

  // Load channel (operand prefetch).
  TmaState tma_state_ = TmaState::IDLE;
  std::vector<uint64_t> tma_req_lines_;
  uint32_t tma_req_idx_ = 0;
  std::unordered_set<uint64_t> tma_inflight_tags_;          // outstanding prefetch tags
  std::unordered_map<uint64_t, uint64_t> tma_tag_line_;     // prefetch tag -> line addr
  std::unordered_map<uint64_t, std::shared_ptr<mem_block_t>> tma_line_data_; // line -> bytes
  uint32_t tma_target_buf_ = 0;
  uint32_t tma_k_ = 0;
  uint32_t tma_fill_left_ = 0;
  uint32_t tma_addrgen_left_ = 0;

  // Issue helpers (TLM): loads carry no payload (data returns in the response);
  // stores carry the filled cache-line block + byteen.
  void issue_load_(uint64_t addr, std::unordered_set<uint64_t>& tagset,
                   std::unordered_map<uint64_t, uint64_t>* tag_line);
  void issue_store_(uint64_t addr, const std::shared_ptr<mem_block_t>& data, uint64_t byteen);

  // Read n bytes (n<=4) at a global byte address from a collected line map, spanning
  // a line boundary if necessary. Replaces the old ram_->read of operand words.
  void read_from_lines_(const std::unordered_map<uint64_t, std::shared_ptr<mem_block_t>>& lines,
                        uint64_t addr, void* dst, uint32_t n) const;

  uint64_t calculate_base_A_(uint32_t k_idx) const;
  uint64_t calculate_base_B_(uint32_t k_idx) const;
  uint64_t calculate_base_C_() const;
  uint64_t calculate_base_D_() const;

  void build_op_req_lines_(uint32_t k_idx, std::vector<uint64_t>& out_lines);
  void build_out_req_lines_(std::vector<uint64_t>& out_lines);
  void build_store_payload_(); // fill out_req_data_/out_req_byteen_ from the accumulator

  void load_operands_into(uint32_t buf_idx, uint32_t k_idx);
  uint32_t buffer_fill_cycles_(uint32_t k_idx) const;
};

} // namespace vortex
