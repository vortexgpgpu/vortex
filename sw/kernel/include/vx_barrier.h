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

#ifndef VX_DXA_SOFT_BAR_BIT
#define VX_DXA_SOFT_BAR_BIT         (1u << 26)
#define VX_DXA_SOFT_BAR_OFFSET_MASK (VX_DXA_SOFT_BAR_BIT - 1u)
#endif

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

  // Pre-register `count` transaction events on this barrier. If a completion
  // arrives first, the signed event balance carries that early release as
  // credit and this call can bring the balance back toward zero.
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

// Cross-CTA barrier for all CTAs in a local group (same core/cluster).
// Unlike `barrier`, the bar slot does NOT embed the caller's CTA id, so
// all CTAs sharing the same id/num_peers use one hardware bar_unit slot.
// Useful for CTA-level rendezvous protocols that intentionally need all peers
// to line up before one CTA proceeds.
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

struct smem_barrier_state {
  int32_t events;
  uint32_t arrived;
  uint32_t phase;
  uint32_t expected_warps;
};

class smem_barrier {
public:
  smem_barrier(smem_barrier_state* state,
               uint32_t num_warps = get_num_sub_groups())
    : state_(state)
    , num_warps_(num_warps) {
  }

  void init() {
    uint32_t active = (uint32_t)vx_active_threads();
    vx_tmc_one();
    state_->events = 0;
    state_->arrived = 0;
    state_->phase = 0;
    state_->expected_warps = num_warps_;
    vx_tmc(active);
  }

  void expect_tx(uint32_t count = 1) {
    atomic_add_events((int32_t)count);
  }

  void complete_tx(uint32_t count = 1) {
    atomic_add_events(-((int32_t)count));
  }

  uint32_t arrive() {
    uint32_t phase = atomic_query_phase();
    uint32_t active = (uint32_t)vx_active_threads();
    vx_tmc_one();
    __atomic_fetch_add(&state_->arrived, 1u, __ATOMIC_SEQ_CST);
    vx_tmc(active);
    return phase;
  }

  uint32_t wait(uint32_t phase) {
    uint32_t spin_iters = 0;
    uint32_t active = (uint32_t)vx_active_threads();
    vx_tmc_one();
    uint32_t target_arrived = state_->expected_warps * (phase + 1);
    for (;;) {
      int32_t events = __atomic_fetch_add(&state_->events, 0, __ATOMIC_SEQ_CST);
      uint32_t arrived = __atomic_fetch_add(&state_->arrived, 0u, __ATOMIC_SEQ_CST);
      if (events == 0 && arrived >= target_arrived) {
        if (state_->phase == phase)
          state_->phase = phase + 1;
        break;
      }
      ++spin_iters;
    }
    vx_tmc(active);
    return spin_iters;
  }

  void arrive_and_wait() {
    wait(arrive());
  }

  smem_barrier_state* state() const {
    return state_;
  }

  uint32_t id() const {
    return VX_DXA_SOFT_BAR_BIT |
           ((uint32_t)(uintptr_t)&state_->events & VX_DXA_SOFT_BAR_OFFSET_MASK);
  }

private:
  void atomic_add_events(int32_t value) {
    uint32_t active = (uint32_t)vx_active_threads();
    vx_tmc_one();
    __atomic_fetch_add(&state_->events, value, __ATOMIC_SEQ_CST);
    vx_tmc(active);
  }

  uint32_t atomic_query_phase() const {
    uint32_t active = (uint32_t)vx_active_threads();
    vx_tmc_one();
    uint32_t phase = __atomic_fetch_add(&state_->phase, 0u, __ATOMIC_SEQ_CST);
    vx_tmc(active);
    return phase;
  }

  smem_barrier_state* state_;
  uint32_t num_warps_;
};

}
