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

#include <deque>
#include "func_unit.h"

namespace vortex {

class FpuUnit : public FuncUnit<VX_CFG_NUM_FPU_BLOCKS> {
public:
  FpuUnit(const SimContext& ctx, const char* name, Core*);

  // Partial bandwidth: a lane-gather stage takes results off the response
  // port. Otherwise the commit arbiter reads the response port directly.
  static constexpr bool kGather = (VX_CFG_NUM_FPU_BLOCKS != VX_CFG_ISSUE_WIDTH)
                               || (VX_CFG_NUM_FPU_LANES != VX_CFG_SIMD_WIDTH);

  // Full bandwidth: the dispatch queue feeds the unit directly, so an op
  // returns its issue credit as the unit accepts it.
  SimChannel<instr_trace_t*> ReleaseOut;

protected:
  void on_reset() override;
  void on_tick() override;

private:
  // Per-unit functional execution. Called only from this unit's tick().
  void execute(instr_trace_t* trace);

  uint32_t latency_of(const instr_trace_t* trace) const;

  // Cycles an operation holds its block's input: its class serves the block's
  // lanes on fewer processing elements, one batch of lanes per cycle.
  uint32_t batches_of(const instr_trace_t* trace) const;

  std::array<uint32_t, VX_CFG_NUM_FPU_BLOCKS> batch_idx_{};

  // Result pipelines, in the order of the block's response arbiter inputs;
  // the divider and the square root share one input, the divider first.
  enum : uint32_t { kPipeFma, kPipeDiv, kPipeCvt, kPipeNcp, kPipeFma64, kPipeSqrt, kNumPipes };
  // respond() indexes arbiter inputs 0..4 directly as pipes; sqrt shares div's input.
  static_assert(kPipeFma64 == 4 && kPipeSqrt == kNumPipes - 1, "arbiter input order");
  uint32_t pipe_of(const instr_trace_t* trace) const;

  // Hands the block one result per cycle, round-robin over the classes whose
  // result has reached its output buffer.
  void respond(uint32_t b);

  struct Result {
    uint64_t       ready; // cycle it reaches the response arbiter
    instr_trace_t* trace;
  };
  std::array<std::array<std::deque<Result>, kNumPipes>, VX_CFG_NUM_FPU_BLOCKS> results_;
  std::array<uint32_t, VX_CFG_NUM_FPU_BLOCKS> num_results_{};
  std::array<uint32_t, VX_CFG_NUM_FPU_BLOCKS> rsp_grant_{};
};

}
