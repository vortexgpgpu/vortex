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

#include <array>
#include "func_unit.h"

namespace vortex {

class AluUnit : public FuncUnit<VX_CFG_NUM_ALU_BLOCKS> {
public:
  // Cycles the multiply/divide pipeline adds over the integer result stage:
  // the 32-bit multiplier (VX_multiplier) or, opted in, the DPI models
  // (VX_CFG_IMUL_DPI_ENABLE / VX_CFG_IDIV_DPI_ENABLE).
  static constexpr uint32_t kMulDivLatency = 3;

  // XLEN 64 multiplies run on VX_mul_multipass: 1 (MULW), 3 (MUL) or 4
  // (MULH*) passes through the 32-bit multiplier, one a cycle while the
  // operation holds the block's input, then the multiplier and one
  // accumulate stage after the last pass.
#if defined(VX_CFG_XLEN_64) && !VX_CFG_IMUL_DPI_ENABLED
  static constexpr bool kMultiPassMul = true;
#else
  static constexpr bool kMultiPassMul = false;
#endif

  // VX_serial_div, one operation at a time per block, as VX_alu_muldiv
  // instantiates it with WIDTHN = XLEN: a step that takes the operands'
  // magnitudes, WIDTHN division steps, and a step that applies the result
  // signs. The unit is busy for its steps; its result is ready the cycle
  // after the last one, and it takes the next operation the cycle after it
  // hands the result on (unless the DPI model is opted in).
  static constexpr bool kSerialDiv = !VX_CFG_IDIV_DPI_ENABLED;
  static constexpr uint32_t kSerialDivSteps = 1 + VX_CFG_XLEN + 1;  // magnitudes, WIDTHN, signs
  static constexpr uint32_t kSerialDivLatency = kSerialDivSteps + 1;
  static constexpr uint32_t kMulLatency = kMulDivLatency + (kMultiPassMul ? 1 : 0);

  // Partial bandwidth: a lane-gather stage registers the results once more.
  static constexpr bool kGather = (VX_CFG_NUM_ALU_BLOCKS != VX_CFG_ISSUE_WIDTH)
                               || (VX_CFG_NUM_ALU_LANES != VX_CFG_SIMD_WIDTH);

  AluUnit(const SimContext& ctx, const char* name, Core*);

  // Resolved branch of a stalled warp, registered toward the scheduler.
  SimChannel<uint32_t> branch_ctl_out;

protected:
  void on_reset() override;
  void on_tick() override;

private:
  // Per-unit functional execution. Called only from this unit's tick().
  // Reads operands from trace->src_data, populates trace->dst_data,
  // mutates warp.PC for branches.
  void execute(instr_trace_t* trace);

  uint32_t latency_of(const instr_trace_t* trace) const;

  // Passes a multiply takes on the multiplier (1 for every other operation).
  static uint32_t mul_passes(const instr_trace_t* trace);

  // Per block, the passes the multiply at its input has issued so far.
  std::array<uint32_t, VX_CFG_NUM_ALU_BLOCKS> mul_pass_{};

  // Per block, the first cycle the iterative divider takes a new operation.
  std::array<uint64_t, VX_CFG_NUM_ALU_BLOCKS> div_free_{};
};

}
