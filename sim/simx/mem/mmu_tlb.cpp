// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <VX_config.h>

#ifdef VM_ENABLE

#include "mmu_tlb.h"
#include <cstddef>

namespace vortex {

Tlb::Tlb(uint32_t size)
    : entries_(size)
{}

std::pair<bool, uint64_t> Tlb::lookup(uint64_t vpn) {
  ++reads_;
  for (auto& e : entries_) {
    if (e.valid && e.vpn == vpn) {
      e.mru = true;
      ++hits_;
      return {true, e.ppn};
    }
  }
  ++misses_;
  return {false, 0};
}

void Tlb::fill(uint64_t vpn, uint64_t ppn, uint8_t flags) {
  // Prefer an invalid slot; fall back to a non-MRU victim. If all slots
  // are valid AND every slot has mru=true, clear MRU on every entry but
  // the soon-to-be-filled one (mirrors the source VX_mmu_tlb behavior).
  int victim = -1;
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (!entries_[i].valid) { victim = (int)i; break; }
  }
  if (victim < 0) {
    for (size_t i = 0; i < entries_.size(); ++i) {
      if (!entries_[i].mru) { victim = (int)i; break; }
    }
  }
  if (victim < 0) {
    // All entries are valid + MRU. Clear MRU bits and pick slot 0.
    for (auto& e : entries_) e.mru = false;
    victim = 0;
  }

  if (entries_[victim].valid)
    ++evictions_;

  entries_[victim].valid = true;
  entries_[victim].mru   = true;
  entries_[victim].vpn   = vpn;
  entries_[victim].ppn   = ppn;
  entries_[victim].flags = flags;
}

void Tlb::flush() {
  for (auto& e : entries_) {
    e.valid = false;
    e.mru = false;
  }
}

} // namespace vortex

#endif // VM_ENABLE
