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

#include "instr_trace.h"

namespace vortex {

class Core;

// Operand collector partition. Each (issue_lane, opc_idx) pair owns the
// integer + float register files for the warps it serves.
// Routing math:
//   lane = wid % ISSUE_WIDTH       — selects which Operands instance
//   wis  = wid / ISSUE_WIDTH       — warp-in-slice index
//   opc  = wis % NUM_OPCS          — which OpcUnit within the lane
//   slot = wis / NUM_OPCS          — local slot inside this OpcUnit
class OpcUnit : public SimObject<OpcUnit> {
public:
  SimChannel<instr_trace_t *> Input;
  SimChannel<instr_trace_t *> Output;

  OpcUnit(const SimContext& ctx, const char* name,
          uint32_t num_warp_slots, uint32_t num_threads);
  virtual ~OpcUnit();

  // Functional regfile writeback applied at unit-tick.
  void writeback(instr_trace_t* trace, uint32_t wid);

  // Read one source operand for `wid` into `out[t]`
  void read_src(std::vector<reg_data_t>& out,
                uint32_t wid,
                uint32_t src_index,
                const RegOpd& reg) const;

  uint32_t total_stalls() const {
    return total_stalls_;
  }

  // DTM debug-only accessor: returns a mutable ref to the integer
  // register file entry for warp slot `slot`, register `reg`, lane `lane`.
  // Used exclusively by sim/simx/dtm/debug_module.cpp (single-hart debug,
  // typically slot=0 lane=0).
  Word& dtm_ireg(uint32_t slot, uint32_t reg, uint32_t lane = 0) {
    return regs_.at(slot).ireg_file.at(reg).at(lane);
  }

protected:
  virtual void on_reset();
  virtual void on_tick();

private:
  struct warp_regs_t {
    std::vector<std::vector<Word>>     ireg_file;   // [reg][thread]
    std::vector<std::vector<uint64_t>> freg_file;   // [reg][thread]
    warp_regs_t(uint32_t num_threads)
      : ireg_file(MAX_NUM_REGS, std::vector<Word>(num_threads, 0))
      , freg_file(MAX_NUM_REGS, std::vector<uint64_t>(num_threads, 0))
    {}
    void reset();
  };

  std::vector<warp_regs_t> regs_;     // indexed by warp_slot
  uint32_t num_threads_;

  uint32_t total_stalls_ = 0;
  instr_trace_t* cur_trace_ = nullptr;
  uint64_t       release_cycle_ = 0;

  friend class SimObject<OpcUnit>;
  friend class Operands;  // Operands::get_exit_code reads regs_ directly
};

} // namespace vortex
