// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <VX_config.h>

#ifdef VX_CFG_VM_ENABLE

#include <VX_types.h>
#include <cstdint>
#include <vector>

namespace vortex {

// Per-core TLB, split into banks selected by the low VPN bits. Each bank is
// a small fully-associative CAM of {vpn -> ppn} translations with MRU-style
// eviction; entries carry their page level so superpage leaves match on the
// VPN bits above that level. Mirrors hw/rtl/mem/VX_mmu_tlb.sv.
class Tlb {
public:
  explicit Tlb(uint32_t size = VX_CFG_TLB_SIZE, uint32_t num_banks = 1);

  struct LookupResult {
    bool     hit = false;
    uint64_t ppn = 0;      // 4 KB-resolved
    uint8_t  level = 0;
  };

  uint32_t bank_of(uint64_t vpn) const { return vpn & (num_banks_ - 1); }

  // Returns the translation for the given vpn. Increments `reads_` on
  // every call and `hits_` on a successful lookup.
  LookupResult lookup(uint64_t vpn);

  // Install a new translation at the given page level. Evicts a non-MRU
  // entry of the bank when full; updates `evictions_` if the chosen slot
  // was previously valid.
  void fill(uint64_t vpn, uint64_t ppn, uint8_t level, uint8_t flags);

  // Invalidate every entry (sfence.vma equivalent).
  void flush();

  uint64_t reads()     const { return reads_; }
  uint64_t hits()      const { return hits_; }
  uint64_t misses()    const { return misses_; }
  uint64_t evictions() const { return evictions_; }

private:
  struct Entry {
    bool     valid = false;
    bool     mru   = false;
    uint8_t  level = 0;
    uint64_t vpn   = 0;
    uint64_t ppn   = 0;
    uint8_t  flags = 0;
  };

  static uint64_t level_mask(uint8_t level);

  // Flat array, bank-major; small enough for a linear scan per bank to
  // model the CAM lookup.
  std::vector<Entry> entries_;
  uint32_t num_banks_;
  uint32_t bank_size_;

  uint64_t reads_     = 0;
  uint64_t hits_      = 0;
  uint64_t misses_    = 0;
  uint64_t evictions_ = 0;
};

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
