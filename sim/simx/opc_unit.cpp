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
#include <iostream>
#include <simobject.h>

using namespace vortex;

OpcUnit::OpcUnit(const SimContext &ctx, const char* name)
  : SimObject<OpcUnit>(ctx, name)
  , Input(this)
  , Output(this) {
  this->reset();
}

OpcUnit::~OpcUnit() {}

void OpcUnit::reset() {
  total_stalls_ = 0;
  cur_trace_ = nullptr;
  release_cycle_ = 0;
}

static uint32_t compute_bank_conflicts(const instr_trace_t* trace) {
  uint32_t stalls = 0;
  for (uint32_t i = 0; i < NUM_SRC_REGS; ++i) {
    for (uint32_t j = i + 1; j < NUM_SRC_REGS; ++j) {
      if ((trace->src_regs[i].type == RegType::None)
       || (trace->src_regs[j].type == RegType::None))
        continue;
      if ((trace->src_regs[i].type == RegType::Integer && trace->src_regs[i].id() == 0)
       || (trace->src_regs[j].type == RegType::Integer && trace->src_regs[j].id() == 0))
        continue; // skip x0
      uint32_t bank_i = trace->src_regs[i].idx % NUM_GPR_BANKS;
      uint32_t bank_j = trace->src_regs[j].idx % NUM_GPR_BANKS;
      if (bank_i == bank_j)
        ++stalls;
    }
  }
  return stalls;
}

void OpcUnit::tick() {
  auto cur_cycle = SimPlatform::instance().cycles();

  // forward held uop once its collection phase has elapsed
  if (cur_trace_ != nullptr && cur_cycle >= release_cycle_) {
    if (!Output.try_send(cur_trace_, 1))
      return;
    DT(3, this->name() << "-pipeline operands: " << *cur_trace_);
    cur_trace_ = nullptr;
  }

  // accept next uop into the holding slot
  if (cur_trace_ == nullptr && !Input.empty()) {
    auto trace = Input.peek();
    uint32_t stalls = compute_bank_conflicts(trace);
    total_stalls_ += stalls;
    cur_trace_ = trace;
    release_cycle_ = cur_cycle + 1 + stalls;
    Input.pop();
  }
}

void OpcUnit::writeback(instr_trace_t* trace) {
  __unused(trace);
}