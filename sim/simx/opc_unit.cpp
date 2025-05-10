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
  , Input(this, 1)
  , Output(this)
  , gpr_req_ports(this)
  , gpr_rsp_ports(this) {
  this->reset();
}

OpcUnit::~OpcUnit() {}

void OpcUnit::reset() {
  pending_rsps_ = 0;
  instr_pending_ = false;
  total_stalls_ = 0;
}

void OpcUnit::tick() {
  // process incoming instructions
  if (Input.empty())
    return;
  auto trace = Input.front();

  if (!instr_pending_) {
    // calculate operands to fetch
    std::bitset<NUM_SRC_REGS> opd_to_fetch;
    for (uint32_t i = 0; i < NUM_SRC_REGS; i++) {
      if (trace->src_regs[i].id() == 0)
        continue; // skip x0 or empty
      // skip duplicates
      bool is_dup = false;
      for (uint32_t j = 0; j < i; j++) {
        if (trace->src_regs[i].id() == trace->src_regs[j].id()) {
          is_dup = true;
          break;
        }
      }
      if (!is_dup) {
        opd_to_fetch.set(i);
        ++pending_rsps_;
      }
    }

    // Send GPR requests
    for (uint32_t i = 0; i < NUM_SRC_REGS; i++) {
      if (opd_to_fetch.test(i)) {
        GprReq gpr_req;
        gpr_req.rid = trace->src_regs[i].id();
        gpr_req.wid = trace->wid;
        gpr_req.opd = i;
        gpr_req_ports.push(gpr_req);
      }
    }

    // mark current instruction as pending
    instr_pending_ = true;
  }

  // process incoming GPR responses
  if (!gpr_rsp_ports.empty()) {
    assert(pending_rsps_ != 0);
    --pending_rsps_;
    auto rsp = gpr_rsp_ports.front();
    __unused(rsp);
    gpr_rsp_ports.pop();
  }

  // process outgoing instructions
  if (0 == pending_rsps_) {
    auto trace = Input.front();
    this->Output.push(trace);
    // release instruction
    Input.pop();
    instr_pending_ = false;
  }
}

void OpcUnit::writeback(instr_trace_t* trace) {
  __unused(trace);
}