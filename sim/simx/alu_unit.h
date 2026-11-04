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
  // the pipelined multiplier (VX_multiplier at XLEN 32) or, opted in, the
  // DPI models (VX_CFG_IMUL_DPI_ENABLE / VX_CFG_IDIV_DPI_ENABLE).
  static constexpr uint32_t kMulDivLatency = 3;

  // The iterative units of VX_alu_muldiv, each one operation at a time per
  // block: VX_serial_mul (XLEN 64; an XLEN+1-bit operand, a bit a cycle) and
  // VX_serial_div (a bit of the XLEN-bit quotient a cycle). An operation's
  // result is ready its width + 1 cycles after it enters, and the unit takes
  // the next one the cycle after it hands the result on.
#if defined(VX_CFG_XLEN_64) && !VX_CFG_IMUL_DPI_ENABLED
  static constexpr bool kSerialMul = true;
#else
  static constexpr bool kSerialMul = false;
#endif
  static constexpr bool kSerialDiv = !VX_CFG_IDIV_DPI_ENABLED;
  static constexpr uint32_t kSerialMulLatency = (VX_CFG_XLEN + 1) + 1;
  static constexpr uint32_t kSerialDivLatency = VX_CFG_XLEN + 1;

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

  // A multiply/divide that runs on an iterative unit, and which one (0 mul, 1 div).
  static int serial_unit(const instr_trace_t* trace);

  // Per block and iterative unit, the first cycle it takes a new operation.
  std::array<std::array<uint64_t, 2>, VX_CFG_NUM_ALU_BLOCKS> serial_free_{};
};

}
