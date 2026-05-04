// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// Include VX_config.h first — it auto-defines VM_ENABLE when the TOML has
// VM_ENABLE = true (via `#ifndef VM_DISABLE / #define VM_ENABLE`), so the
// guard below must follow this include or the file compiles to nothing.
#include <VX_config.h>

#ifdef VM_ENABLE
#include "vm.h"

#include <vortex.h>
#include <common.h>
#include <util.h>
#include <cassert>
#include <cstdlib>
#include <iostream>

using namespace vortex;

namespace {
// Translate the runtime's VX_MEM_{READ,WRITE} access-flag bitmask into PTE
// permission bits. The runtime's flag has READ=1, WRITE=2 (no EXEC); the
// historical formula (flag<<1)|0x3 effectively asserted V|R always plus W
// if WRITE. Spelled out symbolically here so the bit shuffle stays auditable.
uint32_t pte_flags_from_access(uint32_t access_flags) {
  uint32_t pte = PTE_V | PTE_R;
  if (access_flags & VX_MEM_WRITE)
    pte |= PTE_W;
  return pte;
}
} // namespace

VMManager::VMManager(RAM* ram)
    : ram_(ram),
      satp_(nullptr),
      page_table_mem_(nullptr),
      virtual_mem_(nullptr) {
  const char* randomize_env = std::getenv("VORTEX_RANDOMIZE_VA");
  randomize_va_ = (randomize_env != nullptr && std::atoi(randomize_env) != 0);

  const char* seed_env = std::getenv("VORTEX_VA_SEED");
  uint64_t seed = seed_env ? std::atoll(seed_env) : 0x12345678ULL;
  rng_.seed(seed);

  if (randomize_va_) {
    std::cout << "[VM] Virtual address randomization ENABLED (seed=0x"
              << std::hex << seed << std::dec << ")" << std::endl;
  }
}

VMManager::~VMManager() {
  delete virtual_mem_;
  delete page_table_mem_;
}

int VMManager::virtual_mem_reserve(uint64_t dev_addr, uint64_t size, int /*flags*/) {
  CHECK_ERR(virtual_mem_->reserve(dev_addr, size), {
    return err;
  });
  DBGPRINT("[RT:mem_reserve] addr: 0x%lx, size:0x%lx\n", dev_addr, size);
  return 0;
}

int VMManager::init() {
  uint64_t pt_addr = 0;
  std::cout << "VMManager Initialization..." << std::endl;
  page_table_mem_ = new MemoryAllocator(PAGE_TABLE_BASE_ADDR, PT_SIZE_LIMIT, MEM_PAGE_SIZE, CACHE_BLOCK_SIZE);
  if (page_table_mem_ == nullptr)
    return 1;

  uint64_t virtual_mem_size = (GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR);
#ifdef XLEN_32
  // Keep VAs within the 32-bit address space.
  uint64_t max_va_end = 0x100000000ULL;
  if (ALLOC_BASE_ADDR + virtual_mem_size > max_va_end) {
    virtual_mem_size = max_va_end - ALLOC_BASE_ADDR;
  }
#endif
  virtual_mem_ = new MemoryAllocator(ALLOC_BASE_ADDR, virtual_mem_size, MEM_PAGE_SIZE, CACHE_BLOCK_SIZE);

  if (virtual_mem_ == nullptr)
    return 1;

  if (VM_ADDR_MODE == BARE) {
    DBGPRINT("[RT:init_VM] VA_MODE = BARE MODE(addr= 0x0)\n");
  } else {
    CHECK_ERR(alloc_page_table(&pt_addr), { return err; });
  }

  // Stash a local SATP record so need_trans / page_table_walk see the
  // PT base. The simulator's hardware MMU sees SATP via the kernel's
  // csrw at boot — VMManager does not push it into the simulator.
  satp_ = std::make_unique<SATP_t>(pt_addr, /*asid=*/0);

  if (VM_ADDR_MODE != BARE) {
    // Identity-map the system regions that are PA-addressed by the kernel
    // and runtime: the IO MMIO range and the high region containing the
    // page table itself + per-warp stacks. The kernel image range is not
    // mapped here — it is installed by mem_reserve() when the loader
    // reserves it (see runtime simx/rtlsim drivers). The high region's
    // upper bound is GLOBAL_MEM_SIZE — the simulator's RAM extent is the
    // natural ceiling, and STACK_BASE_ADDR sits within it by construction.
    CHECK_ERR(install_identity_map(0, USER_BASE_ADDR), { return err; });
    CHECK_ERR(install_identity_map(PAGE_TABLE_BASE_ADDR,
                                   GLOBAL_MEM_SIZE - PAGE_TABLE_BASE_ADDR), {
      return err;
    });
  }
  return 0;
}

bool VMManager::need_trans(uint64_t dev_pAddr) {
  (void)dev_pAddr;
  // System PA regions (IO, kernel image, page table, stack) are
  // identity-mapped at boot via install_identity_map(), so every
  // address that needs to round-trip through the page table walks the
  // PTEs. The only address that bypasses translation is one issued
  // before SATP is set or in BARE mode.
  if (this->is_satp_unset() || get_mode() == BARE)
    return false;
  return true;
}

uint64_t VMManager::map_p2v(uint64_t ppn, uint32_t flags) {
  if (addr_mapping.find(ppn) != addr_mapping.end())
    return addr_mapping[ppn];

  uint64_t vpn;
  if (randomize_va_) {
    const int MAX_ATTEMPTS = 1000;
    bool allocated = false;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
      uint64_t va_range_start = ALLOC_BASE_ADDR;
      uint64_t va_range_end = PAGE_TABLE_BASE_ADDR;
#ifdef XLEN_32
      uint64_t max_va = 0xFFFFFFFFULL;
      if (va_range_end > max_va)
        va_range_end = max_va;
#endif
      uint64_t va_range_size = va_range_end - va_range_start;
      uint64_t max_pages = va_range_size >> MEM_PAGE_LOG2_SIZE;
      std::uniform_int_distribution<uint64_t> dist(0, max_pages - 1);
      uint64_t random_page_offset = dist(rng_);
      uint64_t candidate_va = va_range_start + (random_page_offset << MEM_PAGE_LOG2_SIZE);

      bool in_use = false;
      for (const auto& mapping : addr_mapping) {
        if (mapping.second == (candidate_va >> MEM_PAGE_LOG2_SIZE)) {
          in_use = true;
          break;
        }
      }
      if (!in_use && virtual_mem_->reserve(candidate_va, MEM_PAGE_SIZE) == 0) {
        vpn = candidate_va >> MEM_PAGE_LOG2_SIZE;
        allocated = true;
        break;
      }
    }
    if (!allocated) {
      virtual_mem_->allocate(MEM_PAGE_SIZE, &vpn);
      vpn >>= MEM_PAGE_LOG2_SIZE;
    }
  } else {
    virtual_mem_->allocate(MEM_PAGE_SIZE, &vpn);
    vpn >>= MEM_PAGE_LOG2_SIZE;
  }

  CHECK_ERR(update_page_table(ppn, vpn, pte_flags_from_access(flags)), );
  addr_mapping[ppn] = vpn;
  return vpn;
}

int VMManager::phy_to_virt_map(uint64_t size, uint64_t* dev_pAddr, uint32_t flags) {
  if (!need_trans(*dev_pAddr))
    return 0;

  uint64_t init_pAddr = *dev_pAddr;
  uint64_t num_pages = size >> MEM_PAGE_LOG2_SIZE;
  uint64_t base_ppn = init_pAddr >> MEM_PAGE_LOG2_SIZE;

  uint64_t base_vpn;
  if (addr_mapping.find(base_ppn) != addr_mapping.end()) {
    base_vpn = addr_mapping[base_ppn];
  } else {
    uint64_t base_va = 0;
    if (randomize_va_) {
      const int MAX_ATTEMPTS = 1000;
      bool allocated = false;
      for (int attempt = 0; attempt < MAX_ATTEMPTS && !allocated; ++attempt) {
        uint64_t va_range_start = ALLOC_BASE_ADDR;
        uint64_t va_range_end = PAGE_TABLE_BASE_ADDR;
#ifdef XLEN_32
        uint64_t max_va = 0xFFFFFFFFULL;
        if (va_range_end > max_va)
          va_range_end = max_va;
#endif
        uint64_t va_range_size = va_range_end - va_range_start;
        uint64_t max_pages = (va_range_size >> MEM_PAGE_LOG2_SIZE) - num_pages;
        std::uniform_int_distribution<uint64_t> dist(0, max_pages - 1);
        uint64_t random_page_offset = dist(rng_);
        uint64_t candidate_va = va_range_start + (random_page_offset << MEM_PAGE_LOG2_SIZE);

        bool range_available = true;
        for (uint64_t i = 0; i < num_pages && range_available; ++i) {
          uint64_t test_vpn = (candidate_va >> MEM_PAGE_LOG2_SIZE) + i;
          for (const auto& mapping : addr_mapping) {
            if (mapping.second == test_vpn) {
              range_available = false;
              break;
            }
          }
        }
        if (range_available && virtual_mem_->reserve(candidate_va, size) == 0) {
          base_va = candidate_va;
          allocated = true;
        }
      }
      if (!allocated) {
        base_va = 0;
        CHECK_ERR(virtual_mem_->allocate(size, &base_va), );
      }
    } else {
      base_va = 0;
      CHECK_ERR(virtual_mem_->allocate(size, &base_va), );
    }
    base_vpn = base_va >> MEM_PAGE_LOG2_SIZE;
  }

  uint64_t init_vAddr = (base_vpn << MEM_PAGE_LOG2_SIZE) | (init_pAddr & ((1 << MEM_PAGE_LOG2_SIZE) - 1));

  for (uint64_t i = 0; i < num_pages; i++) {
    uint64_t ppn = base_ppn + i;
    uint64_t vpn = base_vpn + i;
    if (addr_mapping.find(ppn) == addr_mapping.end()) {
      CHECK_ERR(update_page_table(ppn, vpn, pte_flags_from_access(flags)), );
      addr_mapping[ppn] = vpn;
    }
  }

  assert(page_table_walk(init_vAddr) == init_pAddr && "VA->PA round-trip mismatch");

  *dev_pAddr = init_vAddr;
  return 0;
}

uint8_t VMManager::alloc_page_table(uint64_t* pt_addr) {
  CHECK_ERR(page_table_mem_->allocate(PT_SIZE, pt_addr), { return err; });
  CHECK_ERR(init_page_table(*pt_addr, PT_SIZE), { return err; });
  return 0;
}

int VMManager::init_page_table(uint64_t addr, uint64_t size) {
  uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
  uint8_t* src = new uint8_t[asize]();
  if (src == nullptr)
    return 1;
  ram_->enable_acl(false);
  ram_->write(src, addr, asize);
  ram_->enable_acl(true);
  delete[] src;
  return 0;
}

int16_t VMManager::update_page_table(uint64_t ppn, uint64_t vpn, uint32_t pte_flag, uint8_t leaf_level) {
#if VM_ADDR_MODE == SV39
  assert((((ppn >> 44) == 0) && ((vpn >> 27) == 0)) && "Upper bits are not zero!");
#else
  assert((((ppn >> 20) == 0) && ((vpn >> 20) == 0)) && "Upper 12 bits are not zero!");
#endif
  assert(leaf_level < PT_LEVEL && "leaf_level out of range");
  int i = PT_LEVEL - 1;
  vAddr_t vaddr(vpn << MEM_PAGE_LOG2_SIZE);
  uint64_t pte_addr = 0, pte_bytes = 0;
  uint64_t pt_addr = 0;
  uint64_t cur_base_ppn = get_base_ppn();

  while (i >= 0) {
    pte_addr = (cur_base_ppn * PT_SIZE) + (vaddr.vpn[i] * PTE_SIZE);
    pte_bytes = read_pte(pte_addr);
    PTE_t pte_chk(pte_bytes);
    if (pte_chk.v == 1 && ((pte_bytes & 0xFFFFFFFF) != 0xbaadf00d)) {
      cur_base_ppn = pte_chk.ppn;
    } else {
      if (i == (int)leaf_level) {
        // Leaf: caller supplies the raw PTE permission bits (PTE_V|PTE_R
        // already required by the SV32/SV39 spec for any leaf).
        PTE_t new_pte(ppn << MEM_PAGE_LOG2_SIZE, pte_flag);
        write_pte(pte_addr, new_pte.pte_bytes);
        break;
      } else {
        // Interior: allocate next-level table; PTE_V only (RWX cleared marks
        // it as a pointer to the next-level table per the spec).
        alloc_page_table(&pt_addr);
        PTE_t new_pte(pt_addr, PTE_V);
        write_pte(pte_addr, new_pte.pte_bytes);
        cur_base_ppn = new_pte.ppn;
      }
    }
    i--;
  }
  return 0;
}


int VMManager::install_identity_map(uint64_t addr, uint64_t size) {
  // Per-level page coverage: L0 = MEM_PAGE_SIZE, each higher level
  // multiplies by PT entries-per-table. SV32: L1=4MB. SV39: L1=2MB, L2=1GB.
  constexpr uint64_t PT_FANOUT = PT_SIZE / PTE_SIZE;
  uint64_t level_size[PT_LEVEL];
  level_size[0] = MEM_PAGE_SIZE;
  for (uint8_t l = 1; l < PT_LEVEL; ++l) {
    level_size[l] = level_size[l - 1] * PT_FANOUT;
  }

  // Identity-mapped system regions get full read/write/execute (the
  // kernel image needs X, stack needs R+W, IO MMIO needs R+W). The
  // host-side ACL on RAM, set independently via mem_access(), is the
  // actual permission boundary.
  constexpr uint32_t IDENTITY_PTE_FLAGS = PTE_V | PTE_R | PTE_W | PTE_X;

  // Reserve the VA range so future phy_to_virt_map cannot mint a VA here.
  // Best-effort: overlap with prior identity regions is harmless.
  (void)virtual_mem_->reserve(addr, size);

  uint64_t cur = addr;
  uint64_t end = addr + size;
  while (cur < end) {
    uint64_t remaining = end - cur;
    // Pick the largest level whose granule both fits the remaining range
    // and aligns with cur. Falls back to L0 (4 KB) in the worst case.
    uint8_t leaf_level = 0;
    for (int l = PT_LEVEL - 1; l > 0; --l) {
      if ((cur % level_size[l]) == 0 && remaining >= level_size[l]) {
        leaf_level = (uint8_t)l;
        break;
      }
    }
    uint64_t vpn = cur >> MEM_PAGE_LOG2_SIZE;
    CHECK_ERR(update_page_table(vpn, vpn, IDENTITY_PTE_FLAGS, leaf_level), {
      return err;
    });
    cur += level_size[leaf_level];
  }
  return 0;
}

uint64_t VMManager::page_table_walk(uint64_t vAddr_bits) {
  if (!need_trans(vAddr_bits))
    return vAddr_bits;

  uint8_t level = PT_LEVEL;
  int i = level - 1;
  vAddr_t vaddr(vAddr_bits);
  uint64_t pte_addr = 0, pte_bytes = 0;
  uint64_t cur_base_ppn = get_base_ppn();

  while (true) {
    pte_addr = (cur_base_ppn * PT_SIZE) + (vaddr.vpn[i] * PTE_SIZE);
    pte_bytes = read_pte(pte_addr);
    PTE_t pte(pte_bytes);
    assert(((pte.pte_bytes & 0xFFFFFFFF) != 0xbaadf00d) && "uninitialized PTE");

    if ((pte.v == 0) | ((pte.r == 0) & (pte.w == 1))) {
      throw Page_Fault_Exception("[RT:PTW] invalid entry");
    }
    if ((pte.r == 0) & (pte.w == 0) & (pte.x == 0)) {
      i--;
      if (i < 0)
        throw Page_Fault_Exception("[RT:PTW] no leaf node");
      cur_base_ppn = pte.ppn;
      continue;
    }
    if (pte.r == 0)
      throw Page_Fault_Exception("[RT:PTW] permission");
    cur_base_ppn = pte.ppn;
    break;
  }
  return (cur_base_ppn << MEM_PAGE_LOG2_SIZE) + vaddr.pgoff;
}

void VMManager::write_pte(uint64_t addr, uint64_t value) {
  uint8_t* src = new uint8_t[PTE_SIZE];
  for (uint64_t i = 0; i < PTE_SIZE; ++i) {
    src[i] = (value >> (i << 3)) & 0xff;
  }
  ram_->enable_acl(false);
  ram_->write(src, addr, PTE_SIZE);
  ram_->enable_acl(true);
  delete[] src;
}

uint64_t VMManager::read_pte(uint64_t addr) {
  uint8_t* dest = new uint8_t[PTE_SIZE];
#ifdef XLEN_32
  uint64_t mask = 0x00000000FFFFFFFFULL;
#else
  uint64_t mask = 0xFFFFFFFFFFFFFFFFULL;
#endif
  ram_->read(dest, addr, PTE_SIZE);
  uint64_t ret = (*(uint64_t*)dest) & mask;
  delete[] dest;
  return ret;
}

// get_base_ppn() and get_mode() are inline accessors in the header.

#endif // VM_ENABLE
