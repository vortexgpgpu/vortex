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
#include <unordered_map>
#include "types.h"
#include "amo_ops.h"

namespace vortex {

// Per-LLC-bank AMO helper: the RVA RMW kernel + a reservation set.
// Reservations are tracked per hart — each hart owns at most one
// reservation, which only another hart's committed write to the same
// line can break. A hart's reservation is never displaced by another
// hart's LR, so a contended LR/SC retry loop always has a winner each
// round (forward progress); the prior fixed-size CAM could evict a
// hart's reservation between its own LR and SC and livelock when the
// number of contending harts exceeded the table size. Owned
// synchronously by CacheBank and exercised from processRequests() in
// the same cycle as a write-hit.
class AmoUnit {
public:
  // `reservation_size` seeds the table; capacity grows per contending
  // hart so forward progress does not depend on the knob.
  explicit AmoUnit(uint32_t reservation_size);

  // Pure RMW kernel — no reservation touch, no mutation. `unsigned_minmax`
  // selects the unsigned variant of MIN/MAX (signed by default).
  AmoComputeResult compute(MemOp op, uint8_t width,
                           uint64_t old_word, uint64_t rhs,
                           bool unsigned_minmax = false) const;

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

  // Stats / diagnostics: number of live reservations.
  uint32_t capacity() const { return (uint32_t)reservations_.size(); }

private:
  // hart_id -> reserved line address. One entry per hart at most.
  std::unordered_map<uint32_t, uint64_t> reservations_;
};

} // namespace vortex
