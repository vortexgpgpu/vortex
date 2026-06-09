// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

// VMManager is unconditionally compiled into the generic libvortex.so —
// VM is discovered at runtime (CP DEV_CAPS.VM_ENABLED), not via #ifdef.
// It is inert (never constructed) on a device without an MMU. No
// HW-private VX_config.h dependency: VM config comes from VX_types.h.
#include <VX_types.h>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mem_alloc.h>
#include <vm_types.h>

namespace vortex {

// Sv32/Sv39 PTE permission bits (RISC-V privileged spec, table 4.4).
constexpr uint32_t PTE_V = 1u << 0;
constexpr uint32_t PTE_R = 1u << 1;
constexpr uint32_t PTE_W = 1u << 2;
constexpr uint32_t PTE_X = 1u << 3;
constexpr uint32_t PTE_U = 1u << 4;

// Driver-supplied raw device-memory I/O. PA-addressed bulk read/write
// with no ACL semantics — VMManager batches all PT updates at PT-page
// granularity (one transfer per dirty PT page) before reaching this
// interface, so the per-call cost amortizes naturally across many
// PTEs even on FPGA backends.
//
// Driver adapters:
//   simx, rtlsim : memcpy into the sim's RAM backing store (with the
//                  per-RAM ACL temporarily bypassed inside the adapter).
//   xrt, opae    : DMA transfer to/from the FPGA's PT region (the same
//                  low-level path used by mem_write/mem_read).
class DeviceMemIO {
public:
  virtual ~DeviceMemIO() = default;
  virtual void read (void* dst, uint64_t addr, size_t size) = 0;
  virtual void write(const void* src, uint64_t addr, size_t size) = 0;
};

// Host-resident virtual-memory manager. Owns the page-table allocator
// (a PA region at VX_MEM_PAGE_TABLE_BASE_ADDR) and a parallel virtual-address
// allocator. On every vx_mem_alloc, allocates a PA from the global
// pool, then mints a VA + installs PTEs so the kernel can use the VA.
//
// Page tables are SHADOWED in host memory. All read_pte / write_pte
// hit the shadow (host memcpy, no device round-trip). Modifications
// mark the touched PT page dirty; the runtime calls flush() (or lets
// it run implicitly at the end of each public mutator) to push the
// dirty pages to device memory in one bulk transfer per page. This
// matches the host-shadow / batched-update pattern used by mainstream
// GPU drivers (CUDA, ROCm, Level Zero) and keeps the FPGA DMA path
// efficient — a 1 MB allocation costs 1 DMA per ~512 PTEs instead of
// 256 individual PTE writes.
//
// SATP is maintained locally so need_trans / page_table_walk can use
// it; the hardware MMU separately receives SATP via the kernel's
// csrw at boot (no push from host needed).
class VMManager {
public:
  explicit VMManager(DeviceMemIO* dev_io);
  ~VMManager();

  int init();
  int phy_to_virt_map(uint64_t size, uint64_t* dev_pAddr, uint32_t flags);
  bool need_trans(uint64_t dev_pAddr);
  uint64_t page_table_walk(uint64_t vAddr_bits);
  uint64_t map_p2v(uint64_t ppn, uint32_t flags);
  int virtual_mem_reserve(uint64_t dev_addr, uint64_t size, int flags);

  // Install an identity (VA == PA) mapping covering [addr, addr + size).
  // Uses megapage PTEs where alignment + size permit, leaf PTEs otherwise.
  int install_identity_map(uint64_t addr, uint64_t size);

  // Push every dirty shadow PT page to device memory (one device write
  // per dirty page). Implicitly invoked at the end of init() /
  // phy_to_virt_map() / install_identity_map() / map_p2v() so callers
  // can ignore it; exposed publicly for explicit pre-launch sync.
  int flush();

  // Packed SATP value (page-table root + mode) — programmed into the CP so
  // its DMA's MMU walker can find the page table. 0 before init().
  uint64_t satp() const { return satp_ ? satp_->get_satp() : 0; }

private:
  uint8_t alloc_page_table(uint64_t* pt_addr);
  int16_t update_page_table(uint64_t ppn, uint64_t vpn, uint32_t flag, uint8_t leaf_level = 0);

  uint64_t read_pte(uint64_t addr);
  void write_pte(uint64_t addr, uint64_t value = 0xbaadf00d);

  // Shadow-PT helpers — page = PT-sized aligned chunk covering addr.
  std::vector<uint8_t>& touch_pt_page(uint64_t addr);
  const std::vector<uint8_t>* peek_pt_page(uint64_t addr) const;

  bool is_satp_unset() const { return satp_ == nullptr; }
  uint64_t get_base_ppn() const { return satp_ ? satp_->get_base_ppn() : 0; }
  uint8_t  get_mode()     const { return satp_ ? satp_->get_mode() : 0; }

  DeviceMemIO* dev_io_;
  std::unique_ptr<SATP_t> satp_;

  MemoryAllocator* page_table_mem_;
  MemoryAllocator* virtual_mem_;
  std::unordered_map<uint64_t, uint64_t> addr_mapping;

  // Host shadow: key = PT-page-aligned PA, value = VX_VM_PT_SIZE bytes.
  std::unordered_map<uint64_t, std::vector<uint8_t>> shadow_pt_;
  // Dirty set: PT-page-aligned PAs needing flush to device.
  std::unordered_set<uint64_t> dirty_pt_pages_;

  // Randomized VA allocation (VORTEX_RANDOMIZE_VA / VORTEX_VA_SEED).
  std::mt19937_64 rng_;
  bool randomize_va_;
};

} // namespace vortex
