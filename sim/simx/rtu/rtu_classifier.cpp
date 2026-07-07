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

#include "rtu_classifier.h"
#include "rtu_types.h"  // scene-format constants: kPhase2TriFlag*
#include <VX_types.h>   // VX_RT_FLAG_*, VX_RT_CB_TYPE_*

namespace vortex { namespace rtu {

TriClassify classify_tri_hit(uint32_t ray_flags,
                             uint32_t tri_flags,
                             uint32_t inst_flags,
                             bool     back_facing) {
  TriClassify out{};
  out.action = TriAction::Ignore;
  out.terminate_on_first_hit = false;
  out.yield_sbt_idx = 0;
  out.yield_cb_type = 0;

  // TRIANGLE_FLIP_FACING — the instance inverts the geometry's winding,
  // so front/back swap before culling is evaluated.
  if (inst_flags & kRtuInstanceFlagTriFlip) back_facing = !back_facing;

  // Face culling — applied to any hit, opaque or not. An instance with
  // TRIANGLE_FACING_CULL_DISABLE disables both cull directions for its geometry.
  if (!(inst_flags & kRtuInstanceFlagTriCullDis)) {
    if (back_facing  && (ray_flags & VX_RT_FLAG_CULL_BACK_FACING))  return out;
    if (!back_facing && (ray_flags & VX_RT_FLAG_CULL_FRONT_FACING)) return out;
  }

  // Effective-opacity override. The geometry's OPAQUE bit is overridden
  // first by the instance FORCE_{,NO_}OPAQUE flags, then by the ray's OPAQUE /
  // NO_OPAQUE flags (ray flags take precedence over instance flags per the
  // Vulkan spec). If both members of a pair are set, OPAQUE wins (spec leaves
  // it undefined; we pick OPAQUE for determinism).
  bool tri_opaque = (tri_flags & kPhase2TriFlagOpaque) != 0;
  if (inst_flags & kRtuInstanceFlagForceOpaque)      tri_opaque = true;
  else if (inst_flags & kRtuInstanceFlagForceNoOpq)  tri_opaque = false;
  if (ray_flags & VX_RT_FLAG_OPAQUE)         tri_opaque = true;
  else if (ray_flags & VX_RT_FLAG_NO_OPAQUE) tri_opaque = false;

  // Cull-by-opacity-class.
  if (tri_opaque  && (ray_flags & VX_RT_FLAG_CULL_OPAQUE))    return out;
  if (!tri_opaque && (ray_flags & VX_RT_FLAG_CULL_NO_OPAQUE)) return out;

  if (tri_opaque) {
    out.action = TriAction::Commit;
    // TERMINATE_ON_FIRST_HIT — shadow-ray fast path. Caller commits
    // and then stops scanning further tris.
    if (ray_flags & VX_RT_FLAG_TERMINATE_ON_FIRST_HIT) {
      out.terminate_on_first_hit = true;
    }
  } else {
    out.action = TriAction::Yield;
    out.yield_sbt_idx = (tri_flags >> kPhase2TriSbtIdxShift) & kPhase2TriSbtIdxMask;
    out.yield_cb_type = (tri_flags & kPhase2TriFlagProc)
                          ? VX_RT_CB_TYPE_PROC
                          : VX_RT_CB_TYPE_ANYHIT;
  }
  return out;
}

LaneAction finalise_lane(uint32_t ray_flags,
                         bool     any_hit,
                         bool     yield_pending,
                         uint32_t yield_cb_type) {
  // 1. Yield candidate wins if present — even if there's also a closer
  //    opaque hit, the walker will only mark yield_pending when the
  //    candidate is closer than the best opaque (see walker logic).
  if (yield_pending) {
    return (yield_cb_type == VX_RT_CB_TYPE_PROC) ? LaneAction::YieldIs
                                                  : LaneAction::YieldAhs;
  }

  // 2. Opaque hit with CHS enabled and SKIP_CLOSEST_HIT clear → fire CHS.
  if (any_hit
      && (ray_flags & VX_RT_FLAG_ENABLE_CHS)
      && !(ray_flags & VX_RT_FLAG_SKIP_CLOSEST_HIT)) {
    return LaneAction::YieldChs;
  }

  // 3. No hit with MISS enabled → fire MISS.
  if (!any_hit && (ray_flags & VX_RT_FLAG_ENABLE_MISS)) {
    return LaneAction::YieldMiss;
  }

  // 4. No callback: terminal status only.
  return any_hit ? LaneAction::TerminalHit : LaneAction::TerminalMiss;
}

}}  // namespace vortex::rtu
