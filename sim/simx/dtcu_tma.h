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

#pragma once

#include <simobject.h>
#include "mem_sim.h"
#include "mem.h"
#include "dtcu.h"
#include <vector>
#include <unordered_set>

namespace vortex {

// DtcuTma: the DTCU's tensor-memory engine. It owns the L2 memory port and RAM
// handle and performs ALL of the DTCU's memory movement -- descriptor fetch,
// operand (A/B/C) prefetch into the scratchpad, and output (D) store-back --
// including per-tile address generation and cache-line coalescing.
//
// It is a plain helper class (not a SimObject): the owning Dtcu drives it from
// Dtcu::tick(). The memory channels live here so the port is physically on the
// TMA side (matches how a future RTL block would attach to L2). The scratchpad
// buffers stay in Dtcu; the engine reaches them through the Dtcu& back-reference.
class DtcuTma {
public:
  // Memory port (public so the Cluster can bind it to L2).
  SimChannel<MemReq> mem_req_out;
  SimChannel<MemRsp> mem_rsp_in;

  explicit DtcuTma(Dtcu& parent);
  ~DtcuTma();

  void attach_ram(RAM* ram) { ram_ = ram; }
  void reset();

  // Drain all memory responses that arrived this cycle (call once per tick).
  void drain_responses();

  // Descriptor fetch path is single-outstanding: true when its response is back.
  bool main_done() const { return pending_tag_ == 0; }

  // Descriptor fetch: issue the timing request, then read it functionally.
  void issue_desc_req(uint64_t desc_addr) { issue_mem_req(desc_addr, false); }
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
  RAM*  ram_;

  uint64_t tag_alloc_;
  uint64_t pending_tag_; // descriptor-fetch single-outstanding request tag

  // Store channel (output D write-back): runs in the background, overlapped with
  // the next tile's prefetch/compute. Multiple-outstanding, shares the outstanding
  // budget with the load channel but yields the port to it (load priority).
  std::vector<uint64_t> out_req_lines_;
  uint32_t out_req_idx_ = 0;
  std::unordered_set<uint64_t> tma_store_inflight_tags_; // outstanding store-write tags
  bool     tma_store_active_ = false;     // a store is in progress (issuing or draining)
  uint32_t tma_store_accum_idx_ = 0;      // accumulator buffer being stored
  uint64_t tma_store_baseD_ = 0;          // snapshot of the D tile base addr at handoff

  // Load channel (operand prefetch).
  TmaState tma_state_ = TmaState::IDLE;
  std::vector<uint64_t> tma_req_lines_;
  uint32_t tma_req_idx_ = 0;
  std::unordered_set<uint64_t> tma_inflight_tags_; // outstanding prefetch tags (multiple-outstanding)
  uint32_t tma_target_buf_ = 0;
  uint32_t tma_k_ = 0;
  uint32_t tma_fill_left_ = 0;    // remaining buffer-fill (SRAM write) cycles
  uint32_t tma_addrgen_left_ = 0; // remaining address-generation (AGU setup) cycles

  void issue_mem_req(uint64_t addr, bool write);
  void issue_mem_req_tma_(uint64_t addr, bool write);
  void issue_mem_req_store_(uint64_t addr);
  void store_output();

  uint64_t calculate_base_A_(uint32_t k_idx) const;
  uint64_t calculate_base_B_(uint32_t k_idx) const;
  uint64_t calculate_base_C_() const;
  uint64_t calculate_base_D_() const;

  void build_op_req_lines_(uint32_t k_idx, std::vector<uint64_t>& out_lines);
  void build_out_req_lines_(std::vector<uint64_t>& out_lines);

  void load_operands_into(uint32_t buf_idx, uint32_t k_idx);
  uint32_t buffer_fill_cycles_(uint32_t k_idx) const;
};

} // namespace vortex
