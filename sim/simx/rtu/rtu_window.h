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
//
// RtuWindow — the hit window: a per-(warp, lane, slot) 32-bit slot file holding
// traversal RESULTS. The RTU is its only writer, the shader its only reader, and
// the RTU never reads it back. No instruction writes a slot: a ray reaches the
// traversal datapath from the TRACE burst's own operands, and an intersection
// shader's t and hitAttribute ride its CONTINUE. GETW/GETWF read a contiguous
// window into a GP / FP register group.

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <VX_config.h>
#include <VX_types.h>

#include "instr.h"
#include "instr_trace.h"
#include "types.h"

#ifdef VX_RTU_WINDOW_ENABLE

namespace vortex {

class RtuWindow {
public:
  static constexpr uint32_t SLOT_COUNT = VX_RT_SLOT_COUNT;
  using LaneRegs = std::array<uint32_t, SLOT_COUNT>;
  using WarpRegs = std::array<LaneRegs, VX_CFG_NUM_THREADS>;

  RtuWindow() : regfile_(VX_CFG_NUM_WARPS) {
    for (auto& w : regfile_) {
      for (auto& l : w) {
        l.fill(0);
      }
    }
  }

  // Per-warp [lane][slot] view, for the RTU response paths that write a whole
  // record at once.
  WarpRegs&       warp(uint32_t wid)       { return regfile_.at(wid); }
  const WarpRegs& warp(uint32_t wid) const { return regfile_.at(wid); }


  // GETW / GETWF: windowed read — uop reads slot (start + uop) for each active
  // lane into the uop's dst, FP (NaN-boxed) for GETWF, GP (raw) for GETW. The
  // window streams as one fetched macro-op; the slot file is already staged by
  // the RTU's write of the record (a callback yield, or a WAIT terminal).
  instr_trace_t* process_getw_uop(instr_trace_t* trace, uint32_t uop,
                                  bool is_float) {
    auto args = std::get<IntrGfxwArgs>(trace->instr_ptr->get_args());
    uint32_t slot = args.slot + uop;
    if (slot >= SLOT_COUNT)
      return trace;  // out-of-range window — leave dst unwritten
    auto& wregs = regfile_.at(trace->wid);
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      uint32_t bits = wregs.at(t).at(slot);
      if (is_float)
        trace->dst_data[t].u64 = uint64_t(bits) | 0xffffffff00000000ull;  // NaN-box
      else
        trace->dst_data[t].u = bits;
    }
    return trace;
  }


private:
  std::vector<WarpRegs> regfile_;  // [warp_id][lane][slot]
};

} // namespace vortex

#endif // VX_RTU_WINDOW_ENABLE
