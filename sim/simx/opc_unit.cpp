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

#include "opc_unit.h"
#include "core.h"

using namespace vortex;

OpcUnit::OpcUnit(const SimContext &ctx)
  : SimObject<OpcUnit>(ctx, "opc-unit")
  , Input(this)
  , Output(this) {
  this->reset();
}

OpcUnit::~OpcUnit() {}

void OpcUnit::reset() {
  total_stalls_ = 0;
}

void OpcUnit::tick() {
  if (Input.empty())
    return;
  auto trace = Input.front();

  uint32_t stalls = 0;

  // calculate bank conflict stalls
  for (uint32_t i = 0; i < NUM_SRC_REGS; ++i) {
    for (uint32_t j = i + 1; j < NUM_SRC_REGS; ++j) {
      if ((trace->src_regs[i].type == RegType::None)
       || (trace->src_regs[j].type == RegType::None))
        continue;
      if ((trace->src_regs[i].type == RegType::Integer && trace->src_regs[i].id() == 0)
       || (trace->src_regs[j].type == RegType::Integer && trace->src_regs[j].id() == 0))
        continue; // skip x0
      // bank conflict
      uint32_t bank_i = trace->src_regs[i].idx % NUM_GPR_BANKS;
      uint32_t bank_j = trace->src_regs[j].idx % NUM_GPR_BANKS;
      if (bank_i == bank_j)
        ++stalls;
    }
  }

  total_stalls_ += stalls;

  Output.push(trace, 2 + stalls);

  DT(3, "pipeline-operands: " << *trace);

  Input.pop();
}

void OpcUnit::writeback(instr_trace_t* trace) {
  __unused(trace);
}