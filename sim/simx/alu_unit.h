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

#include "func_unit.h"

namespace vortex {

class AluUnit : public FuncUnit<VX_CFG_NUM_ALU_BLOCKS> {
public:
  // Cycles the multiply/divide pipeline adds over the integer result stage.
  static constexpr uint32_t kMulDivLatency = 3;

  // Partial bandwidth: a lane-gather stage registers the results once more.
  static constexpr bool kGather = (VX_CFG_NUM_ALU_BLOCKS != VX_CFG_ISSUE_WIDTH)
                               || (VX_CFG_NUM_ALU_LANES != VX_CFG_SIMD_WIDTH);

  AluUnit(const SimContext& ctx, const char* name, Core*);

  // Resolved branch of a stalled warp, registered toward the scheduler.
  SimChannel<uint32_t> branch_ctl_out;
  // Full bandwidth: the dispatch queue feeds the unit directly, so an op
  // returns its issue credit as the unit accepts it.
  SimChannel<instr_trace_t*> ReleaseOut;

protected:
  void on_reset() override;
  void on_tick() override;

private:
  // Per-unit functional execution. Called only from this unit's tick().
  // Reads operands from trace->src_data, populates trace->dst_data,
  // mutates warp.PC for branches.
  void execute(instr_trace_t* trace);

  uint32_t latency_of(const instr_trace_t* trace) const;

  // Cycles a result entering the block's output on `ready` waits there behind
  // the results already headed to it, one leaving per cycle.
  uint32_t output_wait(uint32_t b, uint64_t ready);

  // Per block: the cycles on which results already sent reach the output, in
  // the order they reach it.
  std::array<std::vector<uint64_t>, VX_CFG_NUM_ALU_BLOCKS> out_ready_;
  // Per block: the first cycle the output is free after the results already
  // gone from out_ready_.
  std::array<uint64_t, VX_CFG_NUM_ALU_BLOCKS> out_free_{};
};

}
