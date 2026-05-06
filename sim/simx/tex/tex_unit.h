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

// TexReq — per-issue texture sample packet on the cluster TEX bus. Mirrors
// VX_tex_bus_if::req_data. The simulator-only fields (`trace`, `block_id`)
// ride alongside and round-trip via TexRsp so the SfuUnit can route the
// completed trace back to its writeback output lane.
struct TexReq {
  uint64_t                                uuid       = 0;
  uint32_t                                tag        = 0;     // {core_local, queue_slot} routing tag
  uint32_t                                stage      = 0;
  uint32_t                                tmask_bits = 0;     // active-thread bitmask (NUM_THREADS lsbs)
  std::array<int32_t,  NUM_THREADS>       u          = {};    // s32 fixed-point coord (VX_TEX_FXD_FRAC bits)
  std::array<int32_t,  NUM_THREADS>       v          = {};
  std::array<uint32_t, NUM_THREADS>       lod        = {};

  // SimX-only: routing back to the per-core SfuUnit writeback.
  instr_trace_t* trace    = nullptr;
  uint32_t       block_id = 0;

  TexReq() = default;
};

// TexRsp — final filtered texels for an in-flight TexReq. Mirrors
// VX_tex_bus_if::rsp_data; carries `trace` + `block_id` back through the
// arbiter so SfuUnit can land the trace on the correct writeback output.
struct TexRsp {
  uint64_t                            uuid     = 0;
  uint32_t                            tag      = 0;
  std::array<uint32_t, NUM_THREADS>   texels   = {};
  instr_trace_t*                      trace    = nullptr;
  uint32_t                            block_id = 0;

  TexRsp() = default;
  // Allow Req → Rsp copy in TxRxArbiter's bypass binding.
  TexRsp(const TexReq& req)
    : uuid(req.uuid), tag(req.tag), texels{}, trace(req.trace), block_id(req.block_id) {}
};

using TexBusArbiter = TxRxArbiter<TexReq, TexRsp>;

class TexCore;

// Per-core SFU PE for vx_tex. Plain (non-SimObject) helper owned by SfuUnit.
// Decodes (u, v, lod) per active thread plus the TEX stage from
// `IntrTexArgs`, allocates a tag-store slot (mirroring VX_tex_unit's
// VX_index_buffer), and sends a TexReq onto the SFU's outbound TexBus
// channel. Returns the trace on success, nullptr on backpressure (SfuUnit
// retries idempotently next cycle).
class TexUnit {
public:
  TexUnit(Core* core, SimChannel<TexReq>& req_out)
    : core_(core), req_out_(req_out) {}

  // Submit. Returns the trace if accepted (caller pops input + does NOT
  // forward to SFU output — the trace is owned by TexCore until
  // completion), or nullptr on full output channel.
  instr_trace_t* process(instr_trace_t* trace, uint32_t block_id);

private:
  Core*               core_;
  SimChannel<TexReq>& req_out_;
};

} // namespace vortex
