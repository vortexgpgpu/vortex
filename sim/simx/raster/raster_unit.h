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

// RasterStamp — per-lane raster output carrying everything the CSR
// plumbing needs to expose to the kernel (pos_mask + pid + 4-corner
// barycentric coords on each of 3 axes). `pos_mask = 0` is the "drained"
// sentinel (matches raster kernel.s `vx_rast() == 0` check).
struct RasterStamp {
  uint32_t pos_mask = 0;                                  // (pos_y<<18) | (pos_x<<4) | mask
  uint32_t pid      = 0;
  std::array<std::array<uint32_t, 4>, 3> bcoords = {};    // [axis][corner], raw float-bit pattern
};

// RasterRsp — per-lane raster payload returned to SfuUnit.
// stamps[t].pos_mask is the vx_rast result; pid + bcoords are latched
// into CsrUnit storage for VX_CSR_RASTER_PID / VX_CSR_RASTER_BCOORD_*.
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

// Per-core SFU PE for vx_rast. Plain (non-SimObject) helper owned by
// SfuUnit. Builds a RasterReq from the trace metadata and posts onto the
// SFU's outbound channel; returns nullptr on backpressure (caller retries
// next cycle, leaving the trace in the input channel).
class RasterUnit {
public:
  RasterUnit(Core* core, SimChannel<RasterReq>& req_out)
    : core_(core), req_out_(req_out) {}

  instr_trace_t* process(instr_trace_t* trace, uint32_t block_id);

private:
  Core*                  core_;
  SimChannel<RasterReq>& req_out_;
};

} // namespace vortex
