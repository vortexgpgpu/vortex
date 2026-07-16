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

#pragma once

#include <array>
#include <simobject.h>
#include "instr_trace.h"
#include "constants.h"
#include "types.h"

namespace vortex {

class Core;

// OmReq — per-issue output-merger packet on the cluster OM bus.
// vx_om has no destination register; OmCore drives the R-M-W
// asynchronously and SfuUnit commits the trace at issue time.
struct OmReq {
  uint64_t                                   uuid       = 0;
  uint32_t                                   tag        = 0;     // routing tag (set by submitter; arb prepends input idx)
  uint32_t                                   tmask_bits = 0;     // active-thread bitmask (VX_CFG_NUM_THREADS lsbs)
  std::array<uint32_t, VX_CFG_NUM_THREADS>          pos_x      = {};    // VX_OM_DIM_BITS
  std::array<uint32_t, VX_CFG_NUM_THREADS>          pos_y      = {};
  std::array<uint8_t,  VX_CFG_NUM_THREADS>          face       = {};    // 1-bit back-face flag
  std::array<uint32_t, VX_CFG_NUM_THREADS>          color      = {};    // ARGB8888 source
  std::array<uint32_t, VX_CFG_NUM_THREADS>          depth      = {};    // VX_OM_DEPTH_BITS source

  // vx_om_export: the fragment arrives as an APERTURE ADDRESS rather than a
  // decoded position. The decode needs the OM DCRs (xbits/ybits/record_shift),
  // which are cluster state -- in hardware that decode lives at the OM ingress,
  // and here it lives in OmCore for the same reason. The core-side unit does not
  // have, and must not need, the OM's DCRs.
  bool                                             from_aperture = false;
  std::array<uint64_t, VX_CFG_NUM_THREADS>          addr       = {};
  uint32_t                                         export_mask = 0;    // bit0=colour, bit1=depth

  OmReq() = default;
};

// Per-core SFU PE for vx_om. Plain (non-SimObject) helper owned by SfuUnit.
// Decodes operands into an OmReq and posts onto the SFU's outbound channel.
// Returns the trace on success (caller falls through to writeback — vx_om
// has no return value), nullptr on backpressure.
class OmUnit {
public:
  OmUnit(Core* core, SimChannel<OmReq>& req_out)
    : core_(core), req_out_(req_out) {}

  // Submit one vx_om4 sub-pixel request. mask_bits selects the lanes that cover
  // this sub-pixel; the caller pre-packs src_data[0..2] = {pos_face, colour,
  // depth}. Returns the trace if accepted, or nullptr on a full output channel.
  instr_trace_t* process_export(instr_trace_t* trace, uint32_t export_mask);

private:
  Core*               core_;
  SimChannel<OmReq>&  req_out_;
};

} // namespace vortex
