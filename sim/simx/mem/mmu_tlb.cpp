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
#include <cstddef>

namespace vortex {

Tlb::Tlb(uint32_t size)
    : entries_(size)
{}

static constexpr uint32_t VPN_LEVEL_BITS = TLB_VPN_LEVEL_BITS;
static_assert((VX_VM_PT_SIZE / VX_VM_PTE_SIZE) == (1u << VPN_LEVEL_BITS),
              "TLB_VPN_LEVEL_BITS must match the page-table fan-out");

Tlb::Result Tlb::lookup(uint64_t vpn) {
  ++reads_;
  for (auto& e : entries_) {
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
  // Prefer an invalid slot; fall back to a non-MRU victim. If all slots
  // are valid AND every slot has mru=true, clear all MRU bits and evict slot 0.
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
