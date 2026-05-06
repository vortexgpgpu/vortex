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

#include "amo_unit.h"
#include <cassert>

using namespace vortex;

AmoUnit::AmoUnit(uint32_t reservation_size)
  : reservations_(reservation_size)
{
  // RVA conformance floor: ≥ 2 entries to allow forward progress
  // for at least two harts under contention.
  assert(reservation_size >= 2 && "AMO_RS_SIZE must be >= 2");
}

AmoComputeResult AmoUnit::compute(AmoType op, uint8_t width,
                                  uint64_t old_word, uint64_t rhs) const {
  return amo_compute(op, width, old_word, rhs);
}

void AmoUnit::reserve(uint32_t hart_id, uint64_t line_addr) {
  ++lru_clock_;
  // Same-hart re-reserve overwrites the existing entry — only one
  // reservation per hart per RVA semantics.
  for (auto& r : reservations_) {
    if (r.valid && r.hart_id == hart_id) {
      r.line_addr = line_addr;
      r.lru       = lru_clock_;
      return;
    }
  }
  // Find an invalid slot, else evict the LRU one. Both policies are
  // conformant because RVA permits spurious SC failure.
  Reservation* victim = nullptr;
  uint32_t oldest = UINT32_MAX;
  for (auto& r : reservations_) {
    if (!r.valid) { victim = &r; break; }
    if (r.lru < oldest) { oldest = r.lru; victim = &r; }
  }
  victim->hart_id   = hart_id;
  victim->line_addr = line_addr;
  victim->valid     = true;
  victim->lru       = lru_clock_;
}

bool AmoUnit::check(uint32_t hart_id, uint64_t line_addr) const {
  for (const auto& r : reservations_) {
    if (r.valid && r.hart_id == hart_id && r.line_addr == line_addr) {
      return true;
    }
  }
  return false;
}

void AmoUnit::invalidate(uint64_t line_addr, uint32_t except_hart_id) {
  for (auto& r : reservations_) {
    if (r.valid && r.line_addr == line_addr && r.hart_id != except_hart_id) {
      r.valid = false;
    }
  }
}

void AmoUnit::clear(uint32_t hart_id, uint64_t line_addr) {
  for (auto& r : reservations_) {
    if (r.valid && r.hart_id == hart_id && r.line_addr == line_addr) {
      r.valid = false;
    }
  }
}

void AmoUnit::reset() {
  for (auto& r : reservations_) {
    r = Reservation{};
  }
  lru_clock_ = 0;
}
