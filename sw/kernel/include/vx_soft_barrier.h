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

#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn2.h>

#ifdef VX_CFG_EXT_SBAR_ENABLE

#ifndef VX_CFG_EXT_A_ENABLE
#error "software barriers require the atomic extension"
#endif

#ifndef VX_CFG_LMEM_ENABLE
#error "software barriers require local memory"
#endif

namespace vortex {

struct alignas(4) soft_barrier_state {
  volatile uint32_t phase;
  volatile uint32_t arrivals;
  volatile uint32_t pending_transactions;
};

static_assert(sizeof(soft_barrier_state) == 3 * sizeof(uint32_t),
              "software barrier state layout changed");

class soft_barrier {
public:
  explicit soft_barrier(soft_barrier_state* state,
                        uint32_t expected_warps = get_num_sub_groups())
    : state_(state)
    , expected_warps_(expected_warps) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(state);
    const uintptr_t base = uintptr_t(VX_MEM_LMEM_BASE_ADDR);
    const uintptr_t offset = address - base;
    if (address < base
     || offset > ((uintptr_t(1) << VX_CFG_LMEM_LOG_SIZE) - sizeof(*state))
     || (address & (alignof(soft_barrier_state) - 1)) != 0
     || expected_warps == 0
     || expected_warps > VX_CFG_NUM_WARPS)
      __builtin_trap();
  }

  void init() {
    if (is_cta_leader()) {
      __atomic_store_n(&state_->arrivals, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&state_->pending_transactions, 0,
                       __ATOMIC_RELAXED);
      __atomic_store_n(&state_->phase, 0, __ATOMIC_RELEASE);
    }
  }

  uint32_t arrive() {
    const uint32_t token =
        __atomic_load_n(&state_->phase, __ATOMIC_ACQUIRE);
    vx_wsync();
    if (is_warp_leader()) {
      const uint32_t old =
          __atomic_fetch_add(&state_->arrivals, 1, __ATOMIC_RELEASE);
      if (old >= expected_warps_)
        __builtin_trap();
    }
    return token;
  }

  void expect_tx(uint32_t count = 1) {
    if (count == 0 || count > VX_CFG_MAX_BAR_EVENTS)
      __builtin_trap();
    if (is_warp_leader()) {
      const uint32_t old = __atomic_fetch_add(
          &state_->pending_transactions, count, __ATOMIC_RELAXED);
      if (old > VX_CFG_MAX_BAR_EVENTS - count)
        __builtin_trap();
    }
  }

  void complete_tx(uint32_t count = 1) {
    if (count == 0 || count > VX_CFG_MAX_BAR_EVENTS)
      __builtin_trap();
    if (is_warp_leader()) {
      const uint32_t old = __atomic_fetch_sub(
          &state_->pending_transactions, count, __ATOMIC_RELEASE);
      if (old < count)
        __builtin_trap();
    }
  }

  void wait(uint32_t token) {
    if (token > 1)
      __builtin_trap();

    if (is_cta_leader()) {
      while (__atomic_load_n(&state_->arrivals, __ATOMIC_ACQUIRE)
               != expected_warps_
          || __atomic_load_n(&state_->pending_transactions,
                             __ATOMIC_ACQUIRE) != 0)
        ;

      __atomic_store_n(&state_->arrivals, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&state_->phase, token ^ 1u, __ATOMIC_RELEASE);
    }

    while (__atomic_load_n(&state_->phase, __ATOMIC_ACQUIRE) == token)
      ;
  }

  void arrive_and_wait() {
    wait(arrive());
  }

  soft_barrier_state* state() const {
    return state_;
  }

  uint32_t dxa_completion_ref() const {
    const uintptr_t address =
        reinterpret_cast<uintptr_t>(&state_->pending_transactions);
    return uint32_t(address - uintptr_t(VX_MEM_LMEM_BASE_ADDR));
  }

private:
  static bool is_warp_leader() {
    return vx_thread_id() == 0;
  }

  static bool is_cta_leader() {
    return get_sub_group_id() == 0 && is_warp_leader();
  }

  soft_barrier_state* state_;
  uint32_t expected_warps_;
};

} // namespace vortex

#endif
