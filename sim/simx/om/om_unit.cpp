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

// vx_om_export. Each lane carries its own aperture address plus colour/depth, so
// one packet covers the whole warp -- no sub-pixel loop and no window read.
//
// The address is passed through UNDECODED: turning it back into (x, y, face)
// needs the OM's aperture DCRs, which are cluster state. In hardware that decode
// lives at the OM ingress; here it is OmCore's job, for exactly the same reason.
instr_trace_t* OmUnit::process_export(instr_trace_t* trace, uint32_t export_mask) {
  if (req_out_.full()) {
    return nullptr;
  }

  OmReq req;
  req.uuid = trace->uuid;
  req.tag  = uint32_t(trace->uuid);
  req.from_aperture = true;
  req.export_mask   = export_mask;

  uint32_t mask_bits = 0;
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) continue;
    req.addr[t]  = trace->src_data[0].at(t).u;   // aperture address
    req.color[t] = trace->src_data[1].at(t).u;
    req.depth[t] = trace->src_data[2].at(t).u;
    mask_bits |= (1u << t);
  }
  req.tmask_bits = mask_bits;

  req_out_.send(req);
  DT(3, "om-unit export: core=" << core_->id() << ", wid=" << trace->wid);
  return trace;
}
