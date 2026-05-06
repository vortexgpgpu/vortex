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

#include "tex_unit.h"
#include "core.h"
#include "constants.h"
#include "debug.h"
#include "types.h"

using namespace vortex;

instr_trace_t* TexUnit::process(instr_trace_t* trace, uint32_t block_id) {
  if (req_out_.full()) {
    return nullptr;
  }

  auto& instr = *trace->instr_ptr;
  auto args   = std::get<IntrTexArgs>(instr.get_args());

  // Pre-clear destination data; TexCore will fill on completion.
  trace->dst_data.assign(NUM_THREADS, reg_data_t{});

  TexReq req;
  req.uuid     = trace->uuid;
  req.tag      = uint32_t(trace->uuid);   // arbiter prepends input-index bits
  req.stage    = args.stage;
  req.trace    = trace;
  req.block_id = block_id;

  uint32_t bits = 0;
  for (uint32_t t = 0; t < NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) continue;
    bits |= (1u << t);
    req.u[t]   = static_cast<int32_t>(trace->src_data[0].at(t).u);
    req.v[t]   = static_cast<int32_t>(trace->src_data[1].at(t).u);
    req.lod[t] = trace->src_data[2].at(t).u;
  }
  req.tmask_bits = bits;

  req_out_.send(req);
  DT(3, "tex-unit submit: core=" << core_->id() << ", wid=" << trace->wid
     << ", stage=" << args.stage);
  return trace;
}
