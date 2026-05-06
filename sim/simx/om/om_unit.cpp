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

#include "om_unit.h"
#include "core.h"
#include "constants.h"
#include "debug.h"
#include "types.h"

using namespace vortex;

instr_trace_t* OmUnit::process(instr_trace_t* trace) {
  if (req_out_.full()) {
    return nullptr;
  }

  // vx_om encoding (mirrors VX_om_unit.sv lines 45-50):
  //   rs1 = (y << 16) | (x << 1) | face
  //   rs2 = color
  //   rs3 = depth
  OmReq req;
  req.uuid = trace->uuid;
  req.tag  = uint32_t(trace->uuid);

  uint32_t bits = 0;
  for (uint32_t t = 0; t < NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) continue;
    bits |= (1u << t);
    uint32_t pos_face = trace->src_data[0].at(t).u;
    req.face[t]  = uint8_t(pos_face & 0x1);
    req.pos_x[t] = (pos_face >> 1) & 0x7fff;
    req.pos_y[t] = (pos_face >> 16) & 0xffff;
    req.color[t] = trace->src_data[1].at(t).u;
    req.depth[t] = trace->src_data[2].at(t).u;
  }
  req.tmask_bits = bits;

  req_out_.send(req);
  DT(3, "om-unit submit: core=" << core_->id() << ", wid=" << trace->wid);
  return trace;
}
