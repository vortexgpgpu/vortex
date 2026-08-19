// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "mmu_tlb.h"

#ifdef VX_CFG_VM_ENABLE

#include <util.h>

namespace vortex {

static constexpr uint32_t VPN_BITS_PER_LEVEL = log2ceil(VX_VM_PT_SIZE / VX_VM_PTE_SIZE);

Tlb::Tlb(uint32_t size, uint32_t num_banks)
    : entries_(size)
    , num_banks_(num_banks)
    , bank_size_(size / num_banks)
{}

uint64_t Tlb::level_mask(uint8_t level) {
  // VPN bits below a superpage's level belong to its page offset.
  return ~((1ULL << (level * VPN_BITS_PER_LEVEL)) - 1);
}

Tlb::LookupResult Tlb::lookup(uint64_t vpn) {
  ++reads_;
  uint32_t base = this->bank_of(vpn) * bank_size_;
  for (uint32_t i = base; i < base + bank_size_; ++i) {
    auto& e = entries_[i];
    auto mask = level_mask(e.level);
    if (e.valid && (e.vpn & mask) == (vpn & mask)) {
      e.mru = true;
      ++hits_;
      // Resolve the (super)page to the specific 4 KB frame.
      return {true, (e.ppn & mask) | (vpn & ~mask), e.level};
    }
  }
  ++misses_;
  return {};
}

void Tlb::fill(uint64_t vpn, uint64_t ppn, uint8_t level, uint8_t flags) {
  // Prefer an invalid slot of the bank; fall back to a non-MRU victim. If
  // every slot is valid AND MRU, clear the MRU bits and evict the first.
  uint32_t base = this->bank_of(vpn) * bank_size_;
  int victim = -1;
  for (uint32_t i = base; i < base + bank_size_; ++i) {
    if (!entries_[i].valid) { victim = (int)i; break; }
  }
  if (victim < 0) {
    for (uint32_t i = base; i < base + bank_size_; ++i) {
      if (!entries_[i].mru) { victim = (int)i; break; }
    }
  }
  if (victim < 0) {
    for (uint32_t i = base; i < base + bank_size_; ++i) {
      entries_[i].mru = false;
    }
    victim = (int)base;
  }

  if (entries_[victim].valid)
    ++evictions_;

  entries_[victim] = Entry{true, true, level, vpn, ppn, flags};
}

void Tlb::flush() {
  for (auto& e : entries_) {
    e.valid = false;
    e.mru = false;
  }
}

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
