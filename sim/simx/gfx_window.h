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
// GfxWindow — the shared graphics register window (per-(warp, lane, slot) 32-bit
// slot file) that the fixed-function blocks stage their operands through. The
// RTU streams a ray's f0..f7 window and its hit attributes here; TEX (vx_tex4)
// reads its u,v payload and writes its texel; OM (vx_om4) reads its quad
// payload. SETW writes one slot, GETW/GETWF read a contiguous window into a GP /
// FP register group. It is owned at the SFU level so it exists whenever ANY of
// OM / TEX / RTU is built — decoupled from the RTU.

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

#ifdef VX_GFX_WINDOW_ENABLE

namespace vortex {

class GfxWindow {
public:
  static constexpr uint32_t SLOT_COUNT = VX_RT_SLOT_COUNT;
  // RASTER dispatch v2 (FWD) payload window: the raster distributor stages the
  // per-lane record {pos_mask, pid} (2 words) into slots [FRAG_SLOT_BASE..]; the
  // FS reads them with GETWS and recomputes per-corner edge values from the
  // primitive edges. The base sits outside the RTU object-ray range [8..13] (at
  // [19..20]) so an FS can hold this record and an in-flight RTU query at once.
  // Single source of truth: VX_types.toml [gfx_window] -> VX_GFX_FRAG_SLOT_BASE
  // (shared with the kernel ABI).
  static constexpr uint32_t FRAG_SLOT_BASE = VX_GFX_FRAG_SLOT_BASE;
  static constexpr uint32_t FRAG_WORDS     = VX_GFX_FRAG_WORDS;
  using LaneRegs = std::array<uint32_t, SLOT_COUNT>;
  using WarpRegs = std::array<LaneRegs, VX_CFG_NUM_THREADS>;

  GfxWindow() : regfile_(VX_CFG_NUM_WARPS) {
    for (auto& w : regfile_) {
      for (auto& l : w) {
        l.fill(0);
        // Vulkan instanceCullMask: a kernel that never touches VX_RT_CULL_MASK
        // should see the "no culling" default (0xff matches every instance
        // mask). Zero would mean "no rays hit any instance" per Vulkan — exactly
        // the opposite of what un-set state should imply.
        l[VX_RT_CULL_MASK] = 0xffu;
      }
    }
  }

  uint32_t get(uint32_t wid, uint32_t lane, uint32_t slot) const {
    return regfile_.at(wid).at(lane).at(slot);
  }
  void set(uint32_t wid, uint32_t lane, uint32_t slot, uint32_t val) {
    regfile_.at(wid).at(lane).at(slot) = val;
  }

  // Per-warp [lane][slot] view, for the RTU trace/wait/terminal paths that
  // stream a whole window at once.
  WarpRegs&       warp(uint32_t wid)       { return regfile_.at(wid); }
  const WarpRegs& warp(uint32_t wid) const { return regfile_.at(wid); }

  // SETW: single-slot write of rs1 -> window[wid][lane][slot] for each active
  // lane. Used by callback dispatchers (RTU) and the FF op setup paths.
  instr_trace_t* process_set(instr_trace_t* trace) {
    auto& instr = *trace->instr_ptr;
    auto args = std::get<IntrRtuArgs>(instr.get_args());
    uint32_t slot = args.slot;
    if (slot >= SLOT_COUNT) {
      return trace;
    }
    auto& wregs = regfile_.at(trace->wid);
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      wregs.at(t)[slot] = static_cast<uint32_t>(trace->src_data[0].at(t).u);
    }
    return trace;
  }

  // GETW / GETWF: windowed read — uop reads slot (start + uop) for each active
  // lane into the uop's dst, FP (NaN-boxed) for GETWF, GP (raw) for GETW. The
  // window streams as one fetched macro-op; the slot file is already staged (by
  // a callback yield / WAIT2 terminal for the RTU, or by the producing FF op).
  instr_trace_t* process_getw_uop(instr_trace_t* trace, uint32_t uop,
                                  bool is_float) {
    auto args = std::get<IntrRtuArgs>(trace->instr_ptr->get_args());
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

  // GETWS: like GETW but the window's warp dimension is indexed by rs1
  // (block_idx/slot), not the executing wid — the FWD-v2 fragment-record read.
  // The raster unit seeded the record at regfile[slot]; the FS reads it back by
  // the slot it recovered from CTA_BLOCK_ID (uniform across the warp's lanes).
  instr_trace_t* process_getws_uop(instr_trace_t* trace, uint32_t uop) {
    auto args = std::get<IntrRtuArgs>(trace->instr_ptr->get_args());
    uint32_t slot = args.slot + uop;
    if (slot >= SLOT_COUNT)
      return trace;  // out-of-range window — leave dst unwritten
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      uint32_t widx = trace->src_data[0].at(t).u;  // block_idx = warp-dim index
      if (widx >= regfile_.size()) continue;       // guard stray index
      trace->dst_data[t].u = regfile_.at(widx).at(t).at(slot);
    }
    return trace;
  }

private:
  std::vector<WarpRegs> regfile_;  // [warp_id][lane][slot]
};

} // namespace vortex

#endif // VX_GFX_WINDOW_ENABLE
