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
#include "opc_unit.h"

namespace vortex {

class Core;

// Operand collection stage. One instance per issue lane.
class Operands : public SimObject<Operands> {
public:
  SimChannel<instr_trace_t*> Input;
  SimChannel<instr_trace_t*> Output;

  Operands(const SimContext &ctx, const char* name, Core* core);

  virtual ~Operands();

  // Capture register operands at issue time into trace->src_data.
  void fetch_operands(instr_trace_t* trace);

  // Apply trace->dst_data to the regfile at unit-tick time.
  void writeback(instr_trace_t* trace);

  // the program exit code by RISC-V tests.
  int get_exit_code() const;

  uint32_t total_stalls() const;

  // DTM debug-only accessor: returns a mutable ref to the integer
  // register file entry for warp `wid`, register `reg` (lane 0).
  // Used exclusively by sim/simx/dtm/debug_module.cpp.
  // The (lane, opc, slot) split for `wid` mirrors Operands::wid_to_opc_idx;
  // here we are inside one issue-lane already.
  Word& dtm_ireg(uint32_t wid, uint32_t reg);

protected:
  virtual void on_reset();
  virtual void on_tick();

private:
  Core* core_;
  std::vector<OpcUnit::Ptr> opc_units_;
  TraceArbiter::Ptr rsp_arb_;

  friend class SimObject<Operands>;
};

} // namespace vortex
