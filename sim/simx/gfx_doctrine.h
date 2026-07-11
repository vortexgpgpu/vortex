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

// FF<->SIMT interface-law structural assertion.
//
// Every fixed-function graphics / RTU op (the CUSTOM1 family routed to the SFU)
// crosses the FF<->SIMT boundary, and the recurring graphics bug class is an
// *un-ordered* crossing: a value handed across work items through shared
// side-band state with no scoreboard guarantee. The binding law requires
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
// Ops that are known to violate the law today are classified KnownViolation:
// they warn once and pass in the default build (so the CI parity matrix and the
// existing suite still run), but abort under VX_GFX_STRICT_DOCTRINE=1 — the
// switch flipped on as each defect is retired.
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

// How an FF op crosses the FF<->SIMT boundary.
enum class Handoff {
  Scoreboarded,    // retires under a destination register / completion handle
  SideEffectFree,  // no architectural effect the SIMT pipeline does not already order
  KnownViolation,  // an in-flight defect: side effect / shared side-band, no handle
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
#ifdef VX_CFG_EXT_OM_ENABLE
  if (std::holds_alternative<OmType>(op)) {
    // vx_om4 is fire-and-forget (rd=x0) AND reads the cross-unit shared graphics
    // window mid-sequence. Still a KnownViolation: giving it a scoreboard handle
    // needs a rd-carrying OM completion op wired through the OM unit, SimX, and
    // mesa emission, and OM determinism is itself a sensitive path. The raster
    // frag record no longer aliases the RTU object-ray window, so the window is no
    // longer cross-unit-shared between the fragment record and an in-flight ray
    // query; the residual debt (TEX/OM/RTU time-sharing the remaining slots by
    // convention) is a per-unit scoreboard-retired window that would give vx_om4 a
    // scoreboard handle (parity with vx_tex4) and a per-unit window.
    return Handoff::KnownViolation;
  }
#endif
#ifdef VX_GFX_WINDOW_ENABLE
  if (std::holds_alternative<GfxwType>(op)) {
    switch (std::get<GfxwType>(op)) {
    case GfxwType::TRACE:                                    // handle writes back to rd
    case GfxwType::WAIT:                                     // status writes back to rd
    case GfxwType::GETWF:                                     // window read -> rd group
    case GfxwType::GETW:                                      // window read -> rd group
    case GfxwType::GETWS:  return Handoff::Scoreboarded;      // slot-indexed window read -> rd group
    case GfxwType::CB_RET: return Handoff::SideEffectFree;    // parked-context release, ordered by mret
    case GfxwType::SETW:   return Handoff::KnownViolation;    // write into cross-unit shared window
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
    // The debt: warn once, and abort only under the strict switch so the default
    // build (and the CI parity matrix) still runs while the fixes land.
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
