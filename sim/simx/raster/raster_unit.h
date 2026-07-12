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

// RasterReq — per-issue raster pop on the cluster RASTER bus.
// Each request asks RasterCore for one quad-mask word per active lane.
struct RasterReq {
  uint64_t                                   uuid       = 0;
  uint32_t                                   tag        = 0;     // arbiter-routing tag
  uint32_t                                   tmask_bits = 0;     // active lanes (VX_CFG_NUM_THREADS lsbs)
  uint32_t                                   core_id    = 0;     // global issuing-core id (static tile→core ownership)
  instr_trace_t*                             trace      = nullptr;
  uint32_t                                   block_id   = 0;

  RasterReq() = default;

  // Required by TxRxArbiter's DT(4, ... << req) trace at simx/types.h.
  friend std::ostream& operator<<(std::ostream& os, const RasterReq& req) {
    os << "tag=0x" << std::hex << req.tag << std::dec
       << ", tmask=0x" << std::hex << req.tmask_bits << std::dec
       << " (#" << req.uuid << ")";
    return os;
  }
};

// RasterStamp — per-lane raster output the distributor stages into the launched
// warp's frag_payload_t. P2: just {pos_mask, pid} — the FS recomputes per-corner
// edge values from the primitive edges + quad origin. `pos_mask = 0` marks an
// uncovered lane; all-zero across the wave is the producer-drained signal.
struct RasterStamp {
  uint32_t pos_mask = 0;                                  // (pos_y<<18) | (pos_x<<4) | mask
  uint32_t pid      = 0;
};

// RasterRsp — per-lane raster payload returned to SfuUnit, which stages each
// covered lane's stamp into the launched warp's gfx window frag_payload_t.
struct RasterRsp {
  uint64_t                                   uuid     = 0;
  uint32_t                                   tag      = 0;
  std::array<RasterStamp, VX_CFG_NUM_THREADS>       stamps   = {};
  instr_trace_t*                             trace    = nullptr;
  uint32_t                                   block_id = 0;

  RasterRsp() = default;
  RasterRsp(const RasterReq& req)
    : uuid(req.uuid), tag(req.tag), stamps{}, trace(req.trace), block_id(req.block_id) {}

  // Required by TxRxArbiter's DT(4, ... << rsp) trace at simx/types.h.
  friend std::ostream& operator<<(std::ostream& os, const RasterRsp& rsp) {
    os << "tag=0x" << std::hex << rsp.tag << std::dec
       << " (#" << rsp.uuid << ")";
    return os;
  }
};

using RasterBusArbiter = TxRxArbiter<RasterReq, RasterRsp>;

class RasterCore;

} // namespace vortex
