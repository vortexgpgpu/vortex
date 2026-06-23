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

// FF<->SIMT interface-law structural assertion (gfx_v2 master plan §1.3, P0).
//
// Every fixed-function graphics / RTU op (the CUSTOM1 family routed to the SFU)
// crosses the FF<->SIMT boundary, and the recurring graphics bug class is an
// *un-ordered* crossing: a value handed across work items through shared
// side-band state with no scoreboard guarantee (the §8 OM determinism bug, the
// RASTER multi-drawcall dropped-draw-call bug). The binding law (C1-C5) requires
// every such crossing to be either scoreboard-ordered (the op retires under a
// destination handle) or provably side-effect-free.
//
// This file mechanizes that as a single classifier over the decoded op: each FF
// op declares exactly one handoff class, and the assertion checks the decoded
// destination against the declaration. Two failure modes abort unconditionally:
//   - a scoreboarded op decoded with no destination register (a decode bug), and
//   - an *unclassified* FF op (a new op added without declaring its handoff).
// The second is the point of the whole exercise: a new FF op cannot reach the
// pipeline without an author deciding, here, how it crosses the boundary.
//
// Ops that are known to violate the law today (the §3 in-flight defects) are
// classified KnownViolation: they warn once and pass in the default build (so
// the CI parity matrix and the existing suite still run), but abort under
// VX_GFX_STRICT_DOCTRINE=1 — the switch P1/P2 flip on as each defect is retired.
//
// Each op type is gated by the same VX_CFG_EXT_* macro that defines it (and its
// OpType variant member) in types.h, so this stays compilable in every config.

#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <variant>

#include "instr.h"
#include "types.h"

namespace vortex {
namespace gfx_doctrine {

// How an FF op crosses the FF<->SIMT boundary (gfx_v2 §1.3, C1-C5).
enum class Handoff {
  Scoreboarded,    // retires under a destination register / completion handle (C2/C3)
  SideEffectFree,  // no architectural effect the SIMT pipeline does not already order
  KnownViolation,  // a §3 in-flight defect: side effect / shared side-band, no handle
  Unclassified,    // not declared — a build error by construction
};

// Declare the handoff class of one decoded FF op. The per-enum switches are
// deliberately exhaustive (no default) so adding an enumerator forces a decision
// here.
inline Handoff classify(const OpType& op) {
#ifdef VX_CFG_EXT_TEX_ENABLE
  if (std::holds_alternative<TexType>(op)) {
    // vx_tex / vx_tex4: texel (and sync handle) writes back to rd.
    return Handoff::Scoreboarded;
  }
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
  if (std::holds_alternative<RasterType>(op)) {
    switch (std::get<RasterType>(op)) {
    case RasterType::POP:   return Handoff::Scoreboarded;    // pos_mask writes back to rd
    case RasterType::BEGIN: return Handoff::SideEffectFree;  // idempotent per-frame trigger
    case RasterType::FWD_RUN: return Handoff::Scoreboarded;  // blocks until epoch drains (C5 single-owner)
    }
    return Handoff::Unclassified;
  }
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
  if (std::holds_alternative<OmType>(op)) {
    // vx_om4 is fire-and-forget (rd=x0) AND reads the cross-unit shared graphics
    // window mid-sequence — the §3.1 C3/C4 defect. P1 gives it a scoreboard
    // handle (parity with vx_tex4) and a per-unit window.
    return Handoff::KnownViolation;
  }
#endif
#ifdef VX_GFX_WINDOW_ENABLE
  if (std::holds_alternative<RtuType>(op)) {
    switch (std::get<RtuType>(op)) {
    case RtuType::TRACE2:                                    // handle writes back to rd
    case RtuType::WAIT2:                                     // status writes back to rd
    case RtuType::GETWF:                                     // window read -> rd group
    case RtuType::GETW:   return Handoff::Scoreboarded;      // window read -> rd group
    case RtuType::CB_RET: return Handoff::SideEffectFree;    // parked-context release, ordered by mret
    case RtuType::SETW:   return Handoff::KnownViolation;    // C4: write into cross-unit shared window
    }
    return Handoff::Unclassified;
  }
#endif
  (void)op;
  return Handoff::Unclassified;
}

// Structural assertion. Invoked once per decoded EXT2 instruction; only FF ops
// (FUType::SFU) are subject to the law — WGATHER and other EXT2 SIMT ops are not
// FF-unit crossings.
inline void check(const Instr& instr) {
  if (instr.get_fu_type() != FUType::SFU) {
    return;
  }

  const bool has_dest = (instr.get_dest_reg().type != RegType::None);
  const Handoff h = classify(instr.get_op_type());

  switch (h) {
  case Handoff::Scoreboarded:
    // A scoreboarded op MUST carry the destination it retires under; a missing
    // one means the decode dropped the handle — the crossing is now un-ordered.
    if (!has_dest) {
      std::cerr << "[gfx-doctrine] FATAL: FF op (op_type variant #"
                << instr.get_op_type().index()
                << ") is declared scoreboarded but decoded with no destination "
                   "register (un-ordered FF<->SIMT handoff)." << std::endl;
      std::abort();
    }
    return;

  case Handoff::SideEffectFree:
    return;

  case Handoff::KnownViolation: {
    // The §3 debt: warn once, and abort only under the strict switch so the
    // default build (and the CI parity matrix) still runs while P1/P2 land.
    static const bool strict = [] {
      const char* e = std::getenv("VX_GFX_STRICT_DOCTRINE");
      return e && e[0] && std::strcmp(e, "0") != 0;
    }();
    if (strict) {
      std::cerr << "[gfx-doctrine] FATAL (strict): FF op (op_type variant #"
                << instr.get_op_type().index()
                << ") violates the C3/C4 interface law (un-ordered FF<->SIMT "
                   "handoff via shared side-band); see gfx_v2_true_gpu.md §3."
                << std::endl;
      std::abort();
    }
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::cerr << "[gfx-doctrine] WARNING: FF op (op_type variant #"
                << instr.get_op_type().index()
                << ") is a known §3 interface-law violation (un-ordered handoff); "
                   "tracked for P1/P2. Set VX_GFX_STRICT_DOCTRINE=1 to enforce."
                << std::endl;
    }
    return;
  }

  case Handoff::Unclassified:
    std::cerr << "[gfx-doctrine] FATAL: FF op (op_type variant #"
              << instr.get_op_type().index()
              << ") has no declared FF<->SIMT handoff class. Every FF op must "
                 "declare one in sim/simx/gfx_doctrine.h::classify() — a "
                 "scoreboarded destination, provably side-effect-free, or an "
                 "explicit §3 KnownViolation." << std::endl;
    std::abort();
  }
}

} // namespace gfx_doctrine
} // namespace vortex
