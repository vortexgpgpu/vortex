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
    , sopc_units_(NUM_OPCS)
    , vopc_units_(NUM_VOPCS)
    , sgpr_unit_(GPR::Create())
    , vgpr_unit_(VGPR::Create())
    , out_arb_(ArbiterType::RoundRobin, NUM_OPCS + NUM_VOPCS)
    , core_(core) {
  // create OPC units
  for (uint32_t i = 0; i < NUM_OPCS; i++) {
    sopc_units_.at(i) = OpcUnit::Create();
  }

  // create VOPC units
  for (uint32_t i = 0; i < NUM_VOPCS; i++) {
    vopc_units_.at(i) = VOpcUnit::Create(core);
  }

  // connect OPC to GPR
  for (uint32_t i = 0; i < NUM_OPCS; i++) {
    sopc_units_.at(i)->gpr_req_ports.bind(&sgpr_unit_->ReqIn.at(i));
    sgpr_unit_->RspOut.at(i).bind(&sopc_units_.at(i)->gpr_rsp_ports);
  }

  // connect VOPC to GPR and VGPR
  for (uint32_t i = 0; i < NUM_VOPCS; i++) {
    vopc_units_.at(i)->gpr_req_ports.bind(&sgpr_unit_->ReqIn.at(NUM_OPCS + i));
    sgpr_unit_->RspOut.at(NUM_OPCS + i).bind(&vopc_units_.at(i)->gpr_rsp_ports);
    vopc_units_.at(i)->vgpr_req_ports.bind(&vgpr_unit_->ReqIn.at(i));
    vgpr_unit_->RspOut.at(i).bind(&vopc_units_.at(i)->vgpr_rsp_ports);
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
  // process outgoing instructions
  {
    BitVector<> valid_set(NUM_OPCS + NUM_VOPCS);
    for (uint32_t i = 0; i < NUM_OPCS; i++) {
      valid_set.set(i, !sopc_units_.at(i)->Output.empty());
    }
    for (uint32_t i = 0; i < NUM_VOPCS; i++) {
      valid_set.set(NUM_OPCS + i, !vopc_units_.at(i)->Output.empty());
    }
    if (valid_set.any()) {
      uint32_t g = out_arb_.grant(valid_set);
      instr_trace_t* trace;
      if (g >= NUM_OPCS) {
        g -= NUM_OPCS;
        trace = vopc_units_.at(g)->Output.front();
        vopc_units_.at(g)->Output.pop();
      } else {
        trace = sopc_units_.at(g)->Output.front();
        sopc_units_.at(g)->Output.pop();
      }
      this->Output.push(trace, 1);
      DT(3, "pipeline-operands: " << *trace);
    }
  }

  // process incoming instructions
  if (Input.empty())
    return;
  auto trace = this->Input.front();
  if (trace->fu_type == FUType::VPU
    || (trace->fu_type == FUType::LSU && (trace->lsu_type == LsuType::VLOAD || trace->lsu_type == LsuType::VSTORE))) {
    for (uint32_t i = 0; i < NUM_VOPCS; i++) {
      // skip if busy
      if (vopc_units_.at(i)->Input.full())
        continue;
      // assign instruction
      vopcu_table_[trace] = i;
      vopc_units_.at(i)->Input.push(trace);
      Input.pop();
      break;
    }
  } else {
    for (uint32_t i = 0; i < NUM_OPCS; i++) {
      // skip if busy
      if (sopc_units_.at(i)->Input.full())
        continue;
      // assign instruction
      vopcu_table_.erase(trace);
      sopc_units_.at(i)->Input.push(trace);
      Input.pop();
      break;
    }
  }
}

void Operands::writeback(instr_trace_t* trace) {
  auto it = vopcu_table_.find(trace);
  if (it != vopcu_table_.end()) {
    vopc_units_.at(it->second)->writeback(trace);
  }
}