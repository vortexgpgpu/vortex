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
#include <vector>
#include "types.h"
#include "amo_ops.h"

namespace vortex {

// Per-LLC-bank AMO helper: the RVA RMW kernel + a small reservation
// table. Not a SimObject — owned synchronously by CacheBank, exercised
// from processRequests() in the same cycle as a write-hit. A separate
// ticked object would add a phantom cycle with no RTL counterpart.
class AmoUnit {
public:
  struct Reservation {
    uint32_t hart_id   = 0;
    uint64_t line_addr = 0;
    bool     valid     = false;
    uint32_t lru       = 0;
  };

  // RVA permits spurious SC failure but not spurious success, so any
  // capacity ≥ 2 is conformant. `reservation_size` is a per-bank knob.
  explicit AmoUnit(uint32_t reservation_size);

  // Pure RMW kernel — no reservation touch, no mutation.
  AmoComputeResult compute(AmoType op, uint8_t width,
                           uint64_t old_word, uint64_t rhs) const;

  // LR installs a reservation for (hart_id, line_addr). LRU eviction
  // when the table is full.
  void reserve(uint32_t hart_id, uint64_t line_addr);

  // SC's success criterion: an entry exists for (hart_id, line_addr).
  bool check(uint32_t hart_id, uint64_t line_addr) const;

  // Break reservations on `line_addr` belonging to harts other than
  // `except_hart_id`. Triggered on every committed write reaching the
  // LLC bank's tag array (writethroughs from above + AMO commits +
  // LLC's own write-back write-hit commits).
  void invalidate(uint64_t line_addr, uint32_t except_hart_id);

  // Drop the reservation matching this entry. Called on a successful
  // SC to consume the lock.
  void clear(uint32_t hart_id, uint64_t line_addr);

  void reset();

  // Stats / diagnostics.
  uint32_t capacity() const { return (uint32_t)reservations_.size(); }

private:
  std::vector<Reservation> reservations_;
  uint32_t lru_clock_ = 0;
};

} // namespace vortex
