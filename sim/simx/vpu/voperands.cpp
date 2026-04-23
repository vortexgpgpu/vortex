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

#include "voperands.h"
#include "core.h"

using namespace vortex;

Operands::Operands(const SimContext &ctx, Core* core)
    : SimObject<Operands>(ctx, "operands")
    , Input(this)
    , Output(this)
    , opc_units_(NUM_OPCS) {
  static_assert(NUM_OPCS <= PER_ISSUE_WARPS, "invalid NUM_OPCS value");
  // create OPC units
  for (uint32_t i = 0; i < NUM_OPCS; i++) {
    opc_units_.at(i) = VOpcUnit::Create(core);
  }

  if (NUM_OPCS >= 2) {
    char sname[100];
    snprintf(sname, 100, "%s-rsp_arb", this->name().c_str());
    rsp_arb_ = TraceArbiter::Create(sname, ArbiterType::RoundRobin, NUM_OPCS, 1);
    for (uint32_t i = 0; i < NUM_OPCS; ++i) {
      opc_units_.at(i)->Output.bind(&rsp_arb_->Inputs.at(i));
    }
    rsp_arb_->Outputs.at(0).bind(&this->Output);
  } else {
    // pass-thru
    this->Input.bind(&opc_units_.at(0)->Input);
    opc_units_.at(0)->Output.bind(&this->Output);
  }
}

Operands::~Operands() {
  //--
}

void Operands::reset() {
  //--
}

void Operands::tick() {
  if (NUM_OPCS < 2)
    return; // pass-thru

  // process requests
  if (Input.empty())
    return;
  auto trace = this->Input.front();
  for (uint32_t i = 0; i < NUM_OPCS; i++) {
    uint32_t wis = trace->wid / ISSUE_WIDTH;
    uint32_t index = wis % NUM_OPCS;
    opc_units_.at(index)->Input.push(trace);
    Input.pop();
    break;
  }
}

uint32_t Operands::total_stalls() const {
  uint32_t total = 0;
  for (const auto& opc_unit : opc_units_) {
    total += opc_unit->total_stalls();
  }
  return total;
}

void Operands::writeback(instr_trace_t* trace) {
  uint32_t wis = trace->wid / ISSUE_WIDTH;
  uint32_t index = wis % NUM_OPCS;
  opc_units_.at(index)->writeback(trace);
}