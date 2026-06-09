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

AmoUnit::AmoUnit(uint32_t reservation_size) {
  assert(reservation_size >= 2 && "VX_CFG_AMO_RS_SIZE must be >= 2");
  reservations_.reserve(reservation_size);
}

AmoComputeResult AmoUnit::compute(MemOp op, uint8_t width,
                                  uint64_t old_word, uint64_t rhs,
                                  bool unsigned_minmax) const {
  return amo_compute(op, width, old_word, rhs, unsigned_minmax);
}

void AmoUnit::reserve(uint32_t hart_id, uint64_t line_addr) {
  // One reservation per hart: a re-reserve overwrites the hart's own
  // entry and never touches another hart's.
  reservations_[hart_id] = line_addr;
}

bool AmoUnit::check(uint32_t hart_id, uint64_t line_addr) const {
  auto it = reservations_.find(hart_id);
  return it != reservations_.end() && it->second == line_addr;
}

void AmoUnit::invalidate(uint64_t line_addr, uint32_t except_hart_id) {
  // Break every other hart's reservation on this line.
  for (auto it = reservations_.begin(); it != reservations_.end(); ) {
    if (it->second == line_addr && it->first != except_hart_id) {
      it = reservations_.erase(it);
    } else {
      ++it;
    }
  }
}

void AmoUnit::clear(uint32_t hart_id, uint64_t line_addr) {
  auto it = reservations_.find(hart_id);
  if (it != reservations_.end() && it->second == line_addr) {
    reservations_.erase(it);
  }
}

void AmoUnit::reset() {
  reservations_.clear();
}
