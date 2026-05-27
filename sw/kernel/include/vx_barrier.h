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

#include <vx_intrinsics.h>
#include <stdint.h>

namespace vortex {

class barrier {
public:
  barrier(uint32_t id, uint32_t num_warps = get_num_sub_groups()) {
    bar_id_ = (id << 8) + get_local_group_id();
    num_warps_ = num_warps;
  }

  // Notify arrival at barrier (non-blocking)
  // Returns: phase (current generation number)
  uint32_t arrive() {
    return vx_barrier_arrive(bar_id_, num_warps_);
  }

  // Wait for barrier phase to complete
  // Blocks until generation > phase
  void wait(uint32_t phase) {
    vx_barrier_wait(bar_id_, phase);
  }

  // Convenience: arrive and wait in one call
  void arrive_and_wait() {
    vx_barrier(bar_id_, num_warps_);
  }

  // Pre-register `count` pending transaction events on this barrier.
  // Used by async data-movement (e.g. DXA multicast) to declare expected
  // completions BEFORE issuing the operation, so non-issuing CTAs/warps
  // know to wait. Count is cumulative across multiple calls.
  void expect_tx(uint32_t count = 1) {
    vx_barrier_expect_tx(bar_id_, count);
  }

  // Packed barrier id (for passing to DXA issue instructions)
  uint32_t id() const { return bar_id_; }

private:
  uint32_t bar_id_;
  uint32_t num_warps_;
};

class gbarrier {
public:
  gbarrier(uint32_t id, uint32_t num_cores = vx_num_cores()) {
    bar_id_ = (id << 8) | 0x80000000;
    num_cores_ = num_cores;
  }

  // Notify arrival at barrier (non-blocking)
  // Returns: phase (current generation number)
  uint32_t arrive() {
    return vx_barrier_arrive(bar_id_, num_cores_);
  }

  // Wait for barrier phase to complete
  // Blocks until generation > phase
  void wait(uint32_t phase) {
    vx_barrier_wait(bar_id_, phase);
  }

  // Convenience: arrive and wait in one call
  void arrive_and_wait() {
    vx_barrier(bar_id_, num_cores_);
  }

  // Pre-register `count` pending transaction events on this gbar (per-core).
  void expect_tx(uint32_t count = 1) {
    vx_barrier_expect_tx(bar_id_, count);
  }

  // Barrier identifier
  uint32_t id() const { return bar_id_; }

private:
  uint32_t bar_id_;
  uint32_t num_cores_;
};

// Local-group barrier — all members of a local group (i.e., all CTAs the
// dispatcher placed on the same core as one cluster) see the same bar
// slot. Unlike `barrier` (per-CTA, intra-CTA sync), the bar handle does
// NOT embed the caller's CTA id, so multiple CTAs constructing the same
// `group_barrier(id, num_peers)` share one bar_unit slot. Used as a
// rendezvous point across cluster members for intra-core DXA multicast:
// every receiver CTA arrives here to guarantee its per-CTA event-bar
// `expect_tx` is in effect before the issuer fires the multicast.
//
// Naming: `vortex::barrier` is intra-CTA ("local" to one CTA);
// `vortex::group_barrier` is across the local group of CTAs.
class group_barrier {
public:
  group_barrier(uint32_t id, uint32_t num_peers) {
    // bar_addr = {wid=0, bar_id=id} — same slot from every CTA's perspective.
    bar_id_ = (id << 8);
    num_peers_ = num_peers;
  }

  uint32_t arrive() {
    return vx_barrier_arrive(bar_id_, num_peers_);
  }

  void wait(uint32_t phase) {
    vx_barrier_wait(bar_id_, phase);
  }

  void arrive_and_wait() {
    vx_barrier(bar_id_, num_peers_);
  }

  uint32_t id() const { return bar_id_; }

private:
  uint32_t bar_id_;
  uint32_t num_peers_;
};

}