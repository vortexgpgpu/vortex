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

#include "raster_unit.h"
#include "core.h"
#include "constants.h"
#include "debug.h"
#include "types.h"

using namespace vortex;

instr_trace_t* RasterUnit::process(instr_trace_t* trace, uint32_t block_id) {
  if (req_out_.full()) {
    return nullptr;
  }

  // vx_rast has no source operands; build a Req carrying the trace pointer
  // and the active-lane mask. RasterCore returns one stamp per active lane.
  trace->dst_data.assign(NUM_THREADS, reg_data_t{});

  RasterReq req;
  req.uuid     = trace->uuid;
  req.tag      = uint32_t(trace->uuid);
  req.trace    = trace;
  req.block_id = block_id;

  uint32_t bits = 0;
  for (uint32_t t = 0; t < NUM_THREADS; ++t) {
    if (trace->tmask.test(t)) bits |= (1u << t);
  }
  req.tmask_bits = bits;

  req_out_.send(req);
  DT(3, "raster-unit submit: core=" << core_->id() << ", wid=" << trace->wid);
  return trace;
}
