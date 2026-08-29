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

#include <vector>

#include "func_unit.h"

namespace vortex {

class FpuUnit : public FuncUnit<VX_CFG_NUM_FPU_BLOCKS> {
public:
  FpuUnit(const SimContext& ctx, const char* name, Core*);

protected:
  void on_tick() override;

private:
  // Per-unit functional execution. Called only from this unit's tick().
  void execute(instr_trace_t* trace);

  uint32_t latency_of(const instr_trace_t* trace) const;

  // Result-exit cycles of operations inside the arithmetic pipelines: an
  // operation holds a tag slot from acceptance until its result leaves the
  // datapath, so at most VX_CFG_FPU_QUEUE_SIZE operations overlap however
  // deep the pipelines are. Results exit out of order across the different
  // pipelines, so slots free by exit time, not acceptance order.
  std::array<std::vector<uint64_t>, VX_CFG_NUM_FPU_BLOCKS> inflight_;
};

}
