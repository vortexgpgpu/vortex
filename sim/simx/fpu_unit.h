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

class FpuUnit : public FuncUnit<VX_CFG_NUM_FPU_BLOCKS> {
public:
  FpuUnit(const SimContext& ctx, const char* name, Core*);

  // Partial bandwidth: a lane-gather stage takes results off the response
  // port. Otherwise the commit arbiter reads the response port directly.
  static constexpr bool kGather = (VX_CFG_NUM_FPU_BLOCKS != VX_CFG_ISSUE_WIDTH)
                               || (VX_CFG_NUM_FPU_LANES != VX_CFG_SIMD_WIDTH);

  // FP64 lanes (VX_CFG_NUM_FP64_LANES). Below the FPU lanes (FLEN 64), an
  // operation on FP64 values runs on a narrow FP64 datapath beside an FP32-only
  // one, in kFp64Batches batches of kFp64Lanes, one per cycle.
  static constexpr uint32_t kFp64Lanes = (VX_CFG_NUM_FP64_LANES < VX_CFG_NUM_FPU_LANES)
                                       ? VX_CFG_NUM_FP64_LANES : VX_CFG_NUM_FPU_LANES;
  static constexpr bool kFp64Split = (VX_CFG_FLEN >= 64) && (kFp64Lanes < VX_CFG_NUM_FPU_LANES);
  static constexpr uint32_t kFp64Batches = VX_CFG_NUM_FPU_LANES / kFp64Lanes;
  static_assert(VX_CFG_NUM_FPU_LANES % kFp64Lanes == 0, "FP64 lanes must divide the FPU lanes");

protected:
  void on_reset() override;
  void on_tick() override;

private:
  // Per-unit functional execution. Called only from this unit's tick().
  void execute(instr_trace_t* trace);

  uint32_t latency_of(const instr_trace_t* trace) const;

  // The operation reads or writes an FP64 value.
  static bool is_fp64(const instr_trace_t* trace);

  // Per block: batches of the FP64 operation at the input head still to issue.
  std::array<uint32_t, VX_CFG_NUM_FPU_BLOCKS> fp64_batches_{};
};

}
