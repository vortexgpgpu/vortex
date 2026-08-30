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
// PRISM RTU — scene walkers.
//
// Two walker classes: FlatWalker traverses a flat triangle list (with optional
// one-deep TLAS instance expansion), Bvh4Walker traverses a CW-BVH scene with
// TLAS→BLAS recursion. Both expose the same one-method interface:
//
//   WalkResult walk_lane(const RtuReq&, lane, SceneView&, LaneState&, PerfStats&)
//
// The walk reads the scene through a SceneView — the set of cache lines the
// calling context has already pulled. A read of a line the context does not
// hold sets SceneView::miss and unwinds the walk immediately: nothing is
// committed, and the caller fetches the reported line and calls again. The walk
// is deterministic, so replaying it against a larger line set reproduces the
// same prefix and gets one step further. That is the demand-fetch model: the
// sequence of misses IS the sequence of node fetches the hardware issues, in
// hardware order, and the difference in the PerfStats test counts between two
// calls is the work the newly-arrived node unlocked.
//
// The caller therefore passes a PerfStats that accumulates across the replays
// of one ray, and reads it as a cumulative total, not a delta.
//
// Walkers are pure mechanics — no policy. Per-tri opacity / culling / flag
// decisions go through rtu_classifier::classify_tri_hit; the end-of-lane
// CHS/MISS/yield decision goes through finalise_lane. Primitive math lives in
// rtu_isect. The walker owns only the traversal FSM + scene reads.

#ifndef _VX_RTU_WALKER_H_
#define _VX_RTU_WALKER_H_

#include <cstdint>

namespace vortex { namespace rtu {

struct RtuReq;
struct SceneView;
struct LaneState;
struct PerfStats;

struct WalkResult {
  bool stalled  = false;   // a needed line is absent; nothing was committed
  bool cb_yield = false;   // the completed walk ended on a callback candidate
};

// ────────────────────────────────────────────────────────────────────
// FlatWalker — TRI_LIST scenes (the whole flat list is the BLAS) and flat TLAS
// scenes (loop over instance records, transform the world ray into each
// instance's object space, walk that instance's BLAS as a flat list).
// ────────────────────────────────────────────────────────────────────
class FlatWalker {
public:
  WalkResult walk_lane(const RtuReq& req, uint32_t lane, SceneView& sv,
                       LaneState& out, PerfStats& perf);
};

// ────────────────────────────────────────────────────────────────────
// Bvh4Walker — depth-first traversal of a compressed wide-BVH scene with
// TLAS→BLAS LeafInst recursion. The internal-node fan-out is width-generic:
// CW-BVH4 (64 B nodes) and CW-BVH6 (96 B nodes) decode into a common
// VxBvhNodeView, so one traversal datapath serves both widths.
// ────────────────────────────────────────────────────────────────────
class Bvh4Walker {
public:
  WalkResult walk_lane(const RtuReq& req, uint32_t lane, SceneView& sv,
                       LaneState& out, PerfStats& perf);
};

}}  // namespace vortex::rtu

#endif  // _VX_RTU_WALKER_H_
