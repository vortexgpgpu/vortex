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
// PRISM RTU — scene walkers (Layer 4 of the rtu_implementation.md
// refactor, Option C / 13 files).
//
// Two walker classes: FlatWalker traverses a flat triangle list (with
// optional one-deep TLAS instance expansion), Bvh4Walker traverses a
// CW-BVH4 scene with TLAS→BLAS recursion. Both expose the same
// one-method interface:
//
//   bool walk_lane(Slot& s, LaneState& l, uint32_t lane, uint32_t slot_idx)
//
// returning true iff the lane queued a CB_YIELD entry into the shared
// AHS queue. The orchestrator (RtuCore::Impl::compute_intersections)
// picks which walker to drive per lane based on l.scene_kind, and ORs
// the per-lane returns to decide whether the slot advances to IN_QUEUE
// (callback pending) or RESP (terminal).
//
// Walkers are pure mechanics — no policy. Per-tri opacity / culling /
// flag decisions go through rtu_classifier::classify_tri_hit; the
// end-of-lane CHS/MISS/yield decision goes through finalise_lane.
// Primitive math (ray-triangle, ray-aabb, affine ray xform) lives in
// rtu_isect. The walker owns only the traversal FSM + memory reads.
//
// SystemC mapping: each walker becomes one SC_MODULE with a traversal
// FSM. Per-walker state (perf counter handle, queue handle) becomes
// constructor-bound module ports.

#ifndef _VX_RTU_WALKER_H_
#define _VX_RTU_WALKER_H_

#include <cstdint>
#include <deque>

namespace vortex { namespace rtu {

struct Slot;
struct LaneState;
struct PerfStats;
struct QueueEntry;

// ────────────────────────────────────────────────────────────────────
// FlatWalker — Phase 1/8 walker. Handles TRI_LIST scenes (the entire
// flat list is the BLAS) and Phase-8 TLAS scenes (loop over instance
// records, transform world ray into each instance's object space,
// walk that instance's BLAS as a flat list).
// ────────────────────────────────────────────────────────────────────
class FlatWalker {
public:
  FlatWalker(PerfStats& perf, std::deque<QueueEntry>& queue)
    : perf_(perf), queue_(queue) {}

  bool walk_lane(Slot& s, LaneState& l, uint32_t lane, uint32_t slot_idx);

private:
  PerfStats& perf_;
  std::deque<QueueEntry>& queue_;
};

// ────────────────────────────────────────────────────────────────────
// Bvh4Walker — Phase 4 walker. Depth-first traversal of a compressed
// wide-BVH scene with TLAS→BLAS LeafInst recursion. The internal-node
// fan-out is width-generic: CW-BVH4 (scene_kind=2, 64 B nodes) and
// CW-BVH6 (scene_kind=3, 96 B nodes) decode into a common VxBvhNodeView,
// so one traversal datapath serves both widths (RTL parametrizes the
// box-PE array by VX_CFG_RTU_BVH_WIDTH). Recursion is bounded by a
// fixed-depth stack (kBvhStackCap inside the .cpp).
// ────────────────────────────────────────────────────────────────────
class Bvh4Walker {
public:
  Bvh4Walker(PerfStats& perf, std::deque<QueueEntry>& queue)
    : perf_(perf), queue_(queue) {}

  bool walk_lane(Slot& s, LaneState& l, uint32_t lane, uint32_t slot_idx);

private:
  PerfStats& perf_;
  std::deque<QueueEntry>& queue_;
};

}}  // namespace vortex::rtu

#endif  // _VX_RTU_WALKER_H_
