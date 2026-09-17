// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <VX_config.h>

#ifdef VX_CFG_VM_ENABLE

#include "mmu_tlb.h"
#include "tlb_types.h"
#include <VX_types.h>
#include <cassert>
#include <cstddef>

namespace vortex {

Tlb::Tlb(uint32_t size, uint32_t num_banks)
    : entries_(size)
    , num_banks_(num_banks)
    , bank_size_(size / num_banks)
{
  assert(num_banks != 0 && (num_banks & (num_banks - 1)) == 0);
  assert(size % num_banks == 0);
}

static constexpr uint32_t VPN_LEVEL_BITS = TLB_VPN_LEVEL_BITS;
static_assert((VX_VM_PT_SIZE / VX_VM_PTE_SIZE) == (1u << VPN_LEVEL_BITS),
              "TLB_VPN_LEVEL_BITS must match the page-table fan-out");

Tlb::Result Tlb::lookup(uint64_t vpn) {
  ++reads_;
  // Only the VPN's own bank is searched: a superpage translation serves
  // lookups from other banks only after each re-walks and installs its own
  // copy, exactly as the banked CAMs behave.
  uint32_t base = bank_of(vpn) * bank_size_;
  for (uint32_t i = base; i < base + bank_size_; ++i) {
    auto& e = entries_[i];
    if (!e.valid) {
      continue;
    }
    uint32_t shift = e.level * VPN_LEVEL_BITS;
    if ((e.vpn >> shift) == (vpn >> shift)) {
      e.mru = true;
      ++hits_;
      uint64_t low_mask = (uint64_t(1) << shift) - 1;
      return {true, e.ppn | (vpn & low_mask), e.flags, e.level};
    }
  }
  ++misses_;
  return {};
}

void Tlb::fill(uint64_t vpn, uint64_t ppn, uint8_t flags, uint8_t level) {
  // Victim selection stays within the VPN's bank: prefer an invalid slot,
  // fall back to a non-MRU victim, and if the whole bank is valid + MRU,
  // clear the bank's MRU bits and evict its slot 0.
  uint32_t base = bank_of(vpn) * bank_size_;
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

  if (entries_[victim].valid) {
    ++evictions_;
  }

  uint64_t low_mask = (uint64_t(1) << (level * VPN_LEVEL_BITS)) - 1;
  entries_[victim].valid = true;
  entries_[victim].mru   = true;
  entries_[victim].vpn   = vpn;
  entries_[victim].ppn   = ppn & ~low_mask;
  entries_[victim].flags = flags;
  entries_[victim].level = level;
}

void Tlb::flush() {
  for (auto& e : entries_) {
    e.valid = false;
    e.mru = false;
  }
}

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
