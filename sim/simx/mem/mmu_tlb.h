// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <VX_config.h>

#ifdef VM_ENABLE

#include <cstdint>
#include <vector>

namespace vortex {

// Per-core TLB. Mirrors VX_mmu_tlb.sv: a small fully-associative CAM
// of {vpn → ppn} translations with MRU-style eviction. Tracks the
// MMU perf counters surfaced by VX_DCR_MPM_CLASS_VM.
class Tlb {
public:
  explicit Tlb(uint32_t size = TLB_SIZE);

  // Returns {hit, ppn} for the given vpn. Increments `reads_` on every
  // call and `hits_` on a successful lookup.
  std::pair<bool, uint64_t> lookup(uint64_t vpn);

  // Install a new translation. Evicts a non-MRU entry when the TLB is
  // full; updates `evictions_` if the chosen slot was previously valid.
  void fill(uint64_t vpn, uint64_t ppn, uint8_t flags);

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
    uint64_t ppn   = 0;
    uint8_t  flags = 0;
  };

  // Linear flat array; size is small enough (typ. 32) that a per-cycle
  // linear scan models hardware CAM behavior cleanly.
  std::vector<Entry> entries_;

  uint64_t reads_     = 0;
  uint64_t hits_      = 0;
  uint64_t misses_    = 0;
  uint64_t evictions_ = 0;
};

} // namespace vortex

#endif // VM_ENABLE
