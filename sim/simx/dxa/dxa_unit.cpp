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

#include "dxa_unit.h"
#include "core.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;

instr_trace_t* DxaUnit::process(instr_trace_t* trace) {
  if (req_out_.full()) {
    return nullptr;
  }

  // 4-lane wgather encoding:
  //   Lane 0: rs1=smem_addr, rs2=coord2
  //   Lane 1: rs1=meta,      rs2=coord3
  //   Lane 2: rs1=coord0,    rs2=coord4
  //   Lane 3: rs1=coord1,    rs2=cta_mask
  auto& rs1 = trace->src_data[0];
  auto& rs2 = trace->src_data[1];

  uint64_t smem_addr = static_cast<uint64_t>(rs1.at(0).u);
  uint32_t meta      = rs1.at(1).u;
  uint32_t coords[5] = {
    static_cast<uint32_t>(rs1.at(2).u),
    static_cast<uint32_t>(rs1.at(3).u),
    static_cast<uint32_t>(rs2.at(0).u),
    static_cast<uint32_t>(rs2.at(1).u),
    static_cast<uint32_t>(rs2.at(2).u),
  };
  uint32_t cta_mask  = rs2.at(3).u;
  uint32_t desc_slot = meta & 0x0fu;
  uint32_t raw_bar   = (meta >> 4) & 0x07ffffffu;

  DxaReq req;
  req.core      = core_;
  req.uuid      = trace->uuid;
  req.wid       = trace->wid;
  req.desc_slot = desc_slot;
  // Keep raw bar_id; multicast offset arithmetic relies on encoded form
  // (cta_no in low 8 bits → bar_id + cta_idx targets next CTA's same bar).
  // Release call site decodes via bar_decode_id().
  req.bar_id    = raw_bar;
  req.cta_mask  = cta_mask;
  req.smem_addr = smem_addr;
  for (int i = 0; i < 5; ++i) req.coords[i] = coords[i];

  // NOTE: barrier transaction registration is now the kernel's responsibility
  // via vx_barrier_arrive_tx() (see sw/kernel/include/vx_barrier.h::arrive_tx).
  // The DXA pipeline only emits release events on completion; pre-registration
  // happens explicitly per-CTA so multicast destinations correctly wait.

  req_out_.send(req);
  DT(4, "dxa-unit submit: core=" << core_->id() << ", wid=" << trace->wid
     << ", slot=" << desc_slot << ", bar=" << bar_id
     << ", cta_mask=0x" << std::hex << cta_mask << std::dec);
  return trace;
}
