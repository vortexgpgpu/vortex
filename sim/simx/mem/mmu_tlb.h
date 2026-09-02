// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <VX_config.h>

#ifdef VX_CFG_VM_ENABLE

#include <cstdint>
#include <vector>

namespace vortex {

// Per-core TLB, banked: the entry array splits into num_banks single-ported
// partitions selected by the low VPN bits (mirrors hw/rtl/vm/VX_tlb_l1.sv).
// Each partition is a small fully-associative CAM with MRU-style eviction;
// the per-cycle one-lookup-per-bank discipline is enforced by the Mmu stage.
// Tracks MMU perf counters (VX_DCR_MPM_CLASS_MEM).
class Tlb {
public:
  explicit Tlb(uint32_t size = VX_CFG_TLB_SIZE,
               uint32_t num_banks = VX_CFG_L1_TLB_NUM_BANKS);

  // Which bank a VPN's lookup (and fill) must use.
  uint32_t bank_of(uint64_t vpn) const { return vpn & (num_banks_ - 1); }
  uint32_t num_banks() const { return num_banks_; }

  struct Result {
    bool     hit = false;
    uint64_t ppn = 0;     // 4KB-granule translation (superpage entries
                          // splice the vpn low bits back in)
    uint8_t  flags = 0;
    uint8_t  level = 0;
  };

  // Increments `reads_` on every call and `hits_` on a successful lookup.
  Result lookup(uint64_t vpn);

  // Install a new translation at the given page level (0 = base page,
  // 1 = mega, 2 = giga); one superpage entry covers its whole range.
  // Evicts a non-MRU entry when the TLB is full; updates `evictions_`
  // if the chosen slot was previously valid.
  void fill(uint64_t vpn, uint64_t ppn, uint8_t flags, uint8_t level);

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
    uint64_t vpn   = 0;
    uint64_t ppn   = 0;   // aligned down to the entry's level
    uint8_t  flags = 0;
    uint8_t  level = 0;
  };

  // Flat array, partitioned by bank: bank b owns the contiguous slice
  // [b*bank_size_, (b+1)*bank_size_). Small enough (typ. 32 entries) for a
  // per-cycle linear scan to model CAM lookup behavior.
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
