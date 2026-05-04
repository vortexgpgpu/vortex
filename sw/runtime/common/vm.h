// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

// VX_config.h before the guard — auto-defines VM_ENABLE from TOML.
#include <VX_config.h>

#ifdef VM_ENABLE

#include <cstdint>
#include <memory>
#include <random>
#include <unordered_map>
#include <mem.h>
#include <mem_alloc.h>

namespace vortex {

// Sv32/Sv39 PTE permission bits (RISC-V privileged spec, table 4.4).
// Bit positions are architectural; named here so the VMM does not
// embed magic numbers when composing PTE entries.
constexpr uint32_t PTE_V = 1u << 0;
constexpr uint32_t PTE_R = 1u << 1;
constexpr uint32_t PTE_W = 1u << 2;
constexpr uint32_t PTE_X = 1u << 3;
constexpr uint32_t PTE_U = 1u << 4;

// Runtime-side virtual-memory manager. Owns the page-table allocator
// (a region of simulated DRAM at PAGE_TABLE_BASE_ADDR) and a parallel
// virtual-address allocator. On every vx_mem_alloc, allocates a PA
// from the global pool, then mints a VA + installs PTEs so the kernel
// can use the VA. Maintains its own SATP record — the simulator's
// hardware MMU receives SATP via the kernel's csrw at boot, so the
// VMManager does not need to push it into the simulator.
class VMManager {
public:
  explicit VMManager(RAM* ram);
  ~VMManager();

  int init();
  int phy_to_virt_map(uint64_t size, uint64_t* dev_pAddr, uint32_t flags);
  bool need_trans(uint64_t dev_pAddr);
  uint64_t page_table_walk(uint64_t vAddr_bits);
  uint64_t map_p2v(uint64_t ppn, uint32_t flags);
  int virtual_mem_reserve(uint64_t dev_addr, uint64_t size, int flags);

  // Install an identity (VA == PA) mapping covering [addr, addr + size).
  // Uses megapage PTEs where alignment + size permit, leaf PTEs otherwise.
  // Reserves the same range in the VA allocator so phy_to_virt_map will
  // not later mint colliding VAs.
  int install_identity_map(uint64_t addr, uint64_t size);

private:
  int init_page_table(uint64_t addr, uint64_t size);
  uint8_t alloc_page_table(uint64_t* pt_addr);
  int16_t update_page_table(uint64_t ppn, uint64_t vpn, uint32_t flag, uint8_t leaf_level = 0);

  uint64_t read_pte(uint64_t addr);
  void write_pte(uint64_t addr, uint64_t value = 0xbaadf00d);
  bool is_satp_unset() const { return satp_ == nullptr; }
  uint64_t get_base_ppn() const { return satp_ ? satp_->get_base_ppn() : 0; }
  uint8_t get_mode() const { return satp_ ? satp_->get_mode() : 0; }

  RAM* ram_;
  std::unique_ptr<SATP_t> satp_;

  MemoryAllocator* page_table_mem_;
  MemoryAllocator* virtual_mem_;
  std::unordered_map<uint64_t, uint64_t> addr_mapping;

  // Randomized VA allocation (VORTEX_RANDOMIZE_VA / VORTEX_VA_SEED).
  // Phase 4 will surface this through the regression scripts; the
  // mechanism itself is ported now to keep parity with source.
  std::mt19937_64 rng_;
  bool randomize_va_;
};

} // namespace vortex

#endif // VM_ENABLE
