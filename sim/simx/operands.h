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
  SimPortIn<instr_trace_t*> Input;
  SimPortOut<instr_trace_t*> Output;

  Operands(const SimContext &ctx, const char* name, Core* core);

  virtual ~Operands();

  // Capture register operands at issue time into trace->src_data.
  void fetch_operands(instr_trace_t* trace);

  // Apply trace->dst_data to the regfile at unit-tick time.
  void writeback(instr_trace_t* trace);

  // Returns the program exit code (as written by RISC-V tests).
  int get_exit_code() const;

  uint32_t total_stalls() const;

  // DTM debug-only accessor: returns a mutable ref to the integer register
  // file entry for warp `wid`, register `reg` (lane 0).
  // Used exclusively by sim/simx/dtm/debug_module.cpp.
  Word& dtm_ireg(uint32_t wid, uint32_t reg);

protected:
  void on_reset();
  void on_tick();
  bool busy() const;

private:
  Core* core_;
  std::vector<std::unique_ptr<OpcUnit>> opc_units_;
  // With several collectors, their outputs ahead of the round-robin merge,
  // which registers its output.
  SimPortInArray<instr_trace_t*> merge_in_;
  Arbiter merge_arb_;
  // First cycle the buffer's head is offered: the cycle after the previous
  // head left.
  uint64_t head_ready_ = 0;

  friend class SimObject<Operands>;
};

} // namespace vortex
