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

#ifdef VX_CFG_EXT_MBAR_ENABLE

namespace vortex {

struct alignas(4) mbarrier_state {
  volatile uint32_t value;
};

static_assert(sizeof(mbarrier_state) == sizeof(uint32_t),
              "mbarrier state must occupy one word");

namespace detail {

constexpr uint32_t bits_for_limit(uint32_t limit) {
  return limit <= 1 ? 1 : 1 + bits_for_limit(limit >> 1);
}

constexpr uint32_t kArrivalBits = bits_for_limit(VX_CFG_NUM_WARPS);
constexpr uint32_t kTransactionBits =
    bits_for_limit(VX_CFG_MAX_BAR_EVENTS);
constexpr uint32_t kPendingArrivalShift = 1;
constexpr uint32_t kExpectedArrivalShift =
    kPendingArrivalShift + kArrivalBits;
constexpr uint32_t kTransactionShift =
    kExpectedArrivalShift + kArrivalBits;
constexpr uint32_t kStateBits =
    kTransactionShift + kTransactionBits;

constexpr uint32_t bit_mask(uint32_t width) {
  return (uint32_t(1) << width) - 1;
}

static_assert(kStateBits <= 32,
              "mbarrier state must fit in one word");

} // namespace detail

inline uint32_t mbarrier_state_phase(
    const volatile mbarrier_state* state) {
  return state->value & 1u;
}

inline uint32_t mbarrier_state_pending_arrivals(
    const volatile mbarrier_state* state) {
  return (state->value >> detail::kPendingArrivalShift)
       & detail::bit_mask(detail::kArrivalBits);
}

inline uint32_t mbarrier_state_expected_arrivals(
    const volatile mbarrier_state* state) {
  return (state->value >> detail::kExpectedArrivalShift)
       & detail::bit_mask(detail::kArrivalBits);
}

inline uint32_t mbarrier_state_pending_transactions(
    const volatile mbarrier_state* state) {
  return (state->value >> detail::kTransactionShift)
       & detail::bit_mask(detail::kTransactionBits);
}

class mbarrier {
public:
  explicit mbarrier(mbarrier_state* state) : state_(state) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(state);
    const uintptr_t base = uintptr_t(VX_MEM_LMEM_BASE_ADDR);
    const uintptr_t offset = address - base;
    if (address < base
     || offset > ((uintptr_t(1) << VX_CFG_LMEM_LOG_SIZE) - sizeof(*state))
     || (address & (alignof(mbarrier_state) - 1)) != 0)
      __builtin_trap();
  }

  void init(uint32_t expected_arrivals) {
    if (expected_arrivals == 0
     || expected_arrivals > VX_CFG_NUM_WARPS)
      __builtin_trap();
    vx_mbarrier_init(state_, expected_arrivals);
  }

  uint32_t arrive(uint32_t count = 1) {
    if (count == 0 || count > VX_CFG_NUM_WARPS)
      __builtin_trap();
    return vx_mbarrier_arrive(state_, count);
  }

  void expect_tx(uint32_t count = 1) {
    if (count == 0 || count > VX_CFG_MAX_BAR_EVENTS)
      __builtin_trap();
    vx_mbarrier_expect_tx(state_, count);
  }

  void wait(uint32_t phase) {
    if (phase > 1)
      __builtin_trap();
    vx_mbarrier_wait(state_, phase);
  }

  void arrive_and_wait() {
    wait(arrive());
  }

  mbarrier_state* state() const {
    return state_;
  }

  uint32_t dxa_completion_ref() const {
    return uint32_t(reinterpret_cast<uintptr_t>(state_)
                  - uintptr_t(VX_MEM_LMEM_BASE_ADDR));
  }

private:
  mbarrier_state* state_;
};

} // namespace vortex

#endif
