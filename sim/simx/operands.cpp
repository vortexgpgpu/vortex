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

#include "operands.h"
#include "core.h"

using namespace vortex;

Operands::Operands(const SimContext &ctx, Core* core)
    : SimObject<Operands>(ctx, "operands")
    , Input(this)
    , Output(this)
    , opc_units_(NUM_OPCS)
    , gpr_unit_(GPR::Create())
    , out_arb_(ArbiterType::RoundRobin, NUM_OPCS)
    , core_(core) {
  // create OPC units
  for (uint32_t i = 0; i < NUM_OPCS; i++) {
    opc_units_.at(i) = OpcUnit::Create(core);
  }
  // connect OPC to GPR
  for (uint32_t i = 0; i < NUM_OPCS; i++) {
    opc_units_.at(i)->gpr_req_ports.bind(&gpr_unit_->ReqIn.at(i));
    gpr_unit_->RspOut.at(i).bind(&opc_units_.at(i)->gpr_rsp_ports);
  }
  // initialize
  this->reset();
}

Operands::~Operands() {}

void Operands::reset() {
  out_arb_.reset();
  total_stalls_ = 0;
}

void Operands::tick() {
  {
    // process outgoing instructions
    BitVector<> valid_set(NUM_OPCS);
    for (uint32_t i = 0; i < NUM_OPCS; i++) {
      valid_set.set(i, !opc_units_.at(i)->Output.empty());
    }
    if (valid_set.any()) {
      uint32_t g = out_arb_.grant(valid_set);
      auto trace = opc_units_.at(g)->Output.front();
      this->Output.push(trace);
      opc_units_.at(g)->Output.pop();
      DT(3, "pipeline-operands: " << *trace);
    }
  }

  // process incoming instructions
  if (Input.empty())
    return;
  auto trace = this->Input.front();
  for (uint32_t i = 0; i < NUM_OPCS; i++) {
    // skip if busy
    if (opc_units_.at(i)->Input.full())
      continue;
    // assign instruction
    opc_units_.at(i)->Input.push(trace);
    Input.pop();
    break;
  }
}

void Operands::writeback(instr_trace_t* trace) {
  __unused(trace);
}