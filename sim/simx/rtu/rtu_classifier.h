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
// PRISM RTU — callback classifier (hit policy).
// Layer 3 of the rtu_implementation.md refactor (Option C, 13 files).
//
// Pure-function policy module. Given a triangle hit (or end-of-lane
// state), applies the Vulkan ray-flag + per-tri opacity rules to
// decide what the walker should do: commit, yield AHS/IS, fire CHS,
// fire MISS, or ignore.
//
// In SimX: stateless free functions (no class state). In SystemC: this
// becomes a combinational `SC_MODULE(CbClassifier)` with no clock
// domain — every output is a pure function of its inputs.
//
// Single point of update for §8.8 (more ray flags) and §11 OMM
// (Opacity Micromap policy). Before the refactor this logic was
// inlined separately in the flat-list and BVH4 walkers.

#ifndef _VX_RTU_CLASSIFIER_H_
#define _VX_RTU_CLASSIFIER_H_

#include <cstdint>

namespace vortex { namespace rtu {

// ────────────────────────────────────────────────────────────────────
// Per-tri decision: called by walker after ray_triangle returns a hit.
// ────────────────────────────────────────────────────────────────────
enum class TriAction : uint8_t {
  Ignore,     // ray flags / face culling killed this tri; walker skips
  Commit,     // opaque hit — caller updates best_t / best_* fields
  Yield,      // non-opaque candidate — caller stages yield_* fields
};

struct TriClassify {
  TriAction action;
  bool      terminate_on_first_hit;  // Commit only: walker stops scanning
  uint32_t  yield_sbt_idx;           // Yield only: extracted from tri_flags
  uint32_t  yield_cb_type;           // Yield only: ANYHIT or PROC
};

// Apply ray flags + per-tri OPAQUE/PROC/SBT_IDX bits to decide the
// per-tri action. Walker has already invoked ray_triangle (which
// produced back_facing); walker also runs SKIP_TRIANGLES / SKIP_AABBS
// before calling this (those flags gate the whole leaf, not per-tri).
TriClassify classify_tri_hit(uint32_t ray_flags,
                             uint32_t tri_flags,
                             bool     back_facing);

// ────────────────────────────────────────────────────────────────────
// End-of-lane decision: called by walker after all tris in scene have
// been classified. Combines `any_hit`, `yield_pending`, and ray flags
// to decide whether to fire a callback and which kind.
// ────────────────────────────────────────────────────────────────────
enum class LaneAction : uint8_t {
  TerminalHit,    // emit TERMINAL HIT — no callback
  TerminalMiss,   // emit TERMINAL MISS — no callback
  YieldAhs,       // queue AHS callback (non-PROC yield candidate)
  YieldIs,        // queue IS callback (PROC yield candidate)
  YieldChs,       // queue CHS callback (any_hit + ENABLE_CHS, no SKIP_CHS)
  YieldMiss,      // queue MISS callback (!any_hit + ENABLE_MISS)
};

LaneAction finalise_lane(uint32_t ray_flags,
                         bool     any_hit,
                         bool     yield_pending,
                         uint32_t yield_cb_type);

}}  // namespace vortex::rtu

#endif  // _VX_RTU_CLASSIFIER_H_
