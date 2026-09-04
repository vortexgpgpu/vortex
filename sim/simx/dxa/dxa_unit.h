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

#include "types.h"
#include "instr_trace.h"
#if defined(VX_CFG_EXT_DXA_S2G_ENABLE) && !defined(VX_CFG_EXT_DXA_GROUP_ENABLE)
#error "VX_CFG_EXT_DXA_S2G_ENABLE requires VX_CFG_EXT_DXA_GROUP_ENABLE"
#endif
#ifdef VX_CFG_EXT_DXA_GROUP_ENABLE
#include "dxa_group_tracker.h"
#endif

namespace vortex {

class Core;

enum class DxaDirection : uint8_t {
  G2S,
  S2G,
};

// DxaReq — per-SFU dispatch packet. The SFU owns the outbound SimChannel;
// DxaUnit is a plain helper sub-class of SfuUnit that decodes lanes and
// pushes onto that channel.
struct DxaReq {
  Core*    core;          // routes barrier-release back at completion
  uint64_t uuid;
  uint32_t wid;
  DxaDirection direction;
#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
  uint32_t epoch;
  uint8_t  group_seq;
  uint32_t op_id;
#endif
  uint32_t desc_slot;     // descriptor table index
  uint32_t bar_id;        // RAW (pre bar_decode_id) — kept raw so that
                          // multicast offset arithmetic (bar_id + cta_idx)
                          // increments cta_no in the encoded low byte.
                          // Decoded only at the release call site.
  uint32_t cta_mask;      // multicast warp mask (>1 bit ⇒ multicast)
  uint64_t smem_addr;
  uint32_t coords[5];
};

#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
// READ milestone returned by a worker after all source LMEM responses for one
// high-level S2G operation have been copied into worker-owned payload storage.
struct DxaReadCompletion {
  uint32_t wid;
  uint32_t epoch;
  uint8_t group_seq;
  uint32_t op_id;
};
#endif

// DXA sub-unit of the SFU. Plain (non-SimObject) class owned by SfuUnit;
// lane-decodes a DxaType trace and forwards it onto the SFU's outbound
// DxaReq channel.
class DxaUnit {
public:
  // The outbound channel lives on SfuUnit (which is the SimObject).
#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
  DxaUnit(Core* core, SimChannel<DxaReq>& req_out,
          SimChannel<DxaReadCompletion>& completion_in);
#else
  DxaUnit(Core* core, SimChannel<DxaReq>& req_out)
    : core_(core), req_out_(req_out)
#ifdef VX_CFG_EXT_DXA_GROUP_ENABLE
    , tracker_(VX_CFG_NUM_WARPS)
    , parked_waits_(VX_CFG_NUM_WARPS)
#endif
  {}
#endif

  // Apply one returned READ event and retire at most one boundary per warp.
  void tick();

  // Reset the non-SimObject state owned by this helper.  DxaUnit is embedded
  // in SfuUnit rather than registered as a separate SimObject, so the parent
  // must forward the simulator reset here.  In particular, parked WAIT_READ
  // traces contain pointers into the old instruction pool and must not survive
  // a kernel/reset boundary.
  void reset();

#ifdef VX_CFG_EXT_DXA_GROUP_ENABLE
  bool peek_ready_wait(uint32_t* wid, uint32_t* block_id,
                       instr_trace_t** trace) const;
  void pop_ready_wait(uint32_t wid);
  void poison(uint32_t wid);
  bool drained(uint32_t wid) const;
  bool advance_epoch(uint32_t wid);
#endif

  // Decode lanes 0..3 from the trace's source operands and try to push a
  // DxaReq on the SFU's outbound channel. Returns the trace on success
  // (caller falls through to writeback) or nullptr on backpressure
  // (caller retries next cycle without side effects).
  instr_trace_t* process(instr_trace_t* trace, uint32_t block_id,
                         bool* release_warp, bool* parked);

private:
  Core*               core_;
  SimChannel<DxaReq>& req_out_;
#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
  SimChannel<DxaReadCompletion>& completion_in_;
#endif
#ifdef VX_CFG_EXT_DXA_GROUP_ENABLE
  DxaGroupTracker tracker_;
  struct ParkedWait {
    instr_trace_t* trace = nullptr;
    uint32_t block_id = 0;
    uint8_t snapshot = 0;
    uint32_t n = 0;
    bool ready = false;
  };
  std::vector<ParkedWait> parked_waits_;
#endif
};

} // namespace vortex
