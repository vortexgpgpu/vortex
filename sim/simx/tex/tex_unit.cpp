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
  trace->dst_data.assign(VX_CFG_NUM_THREADS, reg_data_t{});

  TexReq req;
  req.uuid     = trace->uuid;
  req.tag      = uint32_t(trace->uuid);   // arbiter prepends input-index bits
  req.stage    = args.stage;
  req.trace    = trace;
  req.block_id = block_id;

  // vx_tex_cmp carries its reference in rs3, so its layer, level and lane
  // filter ride above the coordinates; they are unpacked into the vx_tex lod
  // operand's fields here, and the coordinates sign-extended from their top bit.
  req.cmp = std::get<TexType>(instr.get_op_type()) == TexType::SAMPLE_CMP;
  constexpr uint32_t kHi = 32 - VX_TEX_CMP_COORD_BITS;
  uint32_t bits = 0;
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) continue;
    bits |= (1u << t);
    const uint32_t u = trace->src_data[0].at(t).u;
    const uint32_t v = trace->src_data[1].at(t).u;
    if (req.cmp) {
      req.u[t]   = static_cast<int32_t>(u << kHi) >> kHi;
      req.v[t]   = static_cast<int32_t>(v << kHi) >> kHi;
      req.ref[t] = trace->src_data[2].at(t).u;
      const uint32_t layer = (u >> VX_TEX_CMP_U_LAYER_SHIFT) & ((1u << VX_TEX_CMP_LAYER_BITS) - 1u);
      req.lod[t] = ((v >> VX_TEX_CMP_V_LEVEL_SHIFT) & (uint32_t)VX_TEX_LOD_MAX)
                 | (((v >> VX_TEX_CMP_V_LANE_FILTER) & 1u) << VX_TEX_ARG_LANE_FILTER)
                 | (((v >> VX_TEX_CMP_V_BILINEAR) & 1u) << VX_TEX_ARG_BILINEAR)
                 | (layer << VX_TEX_ARG_LAYER_SHIFT);
    } else {
      req.u[t]   = static_cast<int32_t>(u);
      req.v[t]   = static_cast<int32_t>(v);
      req.lod[t] = trace->src_data[2].at(t).u;
    }
  }
  req.tmask_bits = bits;

  // Combinational into the shared arbiter; the arbiter's forward is the
  // registered boundary.
  req_out_.send(req, 0);
  DT(3, "tex-unit submit: core=" << core_->id() << ", wid=" << trace->wid
     << ", stage=" << args.stage);
  return trace;
}
