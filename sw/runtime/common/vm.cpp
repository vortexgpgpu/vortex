// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// VMManager is always compiled into libvortex.so — VM is a runtime device
// property, not a compile-time #ifdef. The Sv32/Sv39 split comes from
// VX_VM_ADDR_MODE (VX_types.h); HW-private VX_config.h is not included.
#include <VX_types.h>
#include "vm.h"

#include <vortex.h>
#include <common.h>
#include <util.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace vortex;

namespace {
// Translate the runtime's VX_MEM_{READ,WRITE} access-flag bitmask into PTE
// permission bits (V|R always; W if WRITE).
uint32_t pte_flags_from_access(uint32_t access_flags) {
  uint32_t pte = PTE_V | PTE_R;
  if (access_flags & VX_MEM_WRITE)
    pte |= PTE_W;
  return pte;
}

// Ceil-log2 — used for the per-level VPN field width.
constexpr unsigned vm_clog2(uint64_t n) {
  unsigned r = 0;
  while ((uint64_t(1) << r) < n) ++r;
  return r;
}
// VPN bits per page-table level = log2(PTEs per table). SV39 -> 9, SV32 -> 10.
constexpr unsigned VM_VPN_BITS = vm_clog2(VX_VM_PT_SIZE / VX_VM_PTE_SIZE);
} // namespace

VMManager::VMManager(DeviceMemIO* dev_io)
    : dev_io_(dev_io),
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
  page_table_mem_ = new MemoryAllocator(VX_MEM_PAGE_TABLE_BASE_ADDR, VX_VM_PT_SIZE_LIMIT, VX_VM_PAGE_SIZE, CACHE_BLOCK_SIZE);
  if (page_table_mem_ == nullptr)
    return 1;

  uint64_t virtual_mem_size = (GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR);
#if VX_VM_ADDR_MODE == SV32
  // Keep VAs within the 32-bit address space.
  uint64_t max_va_end = 0x100000000ULL;
  if (ALLOC_BASE_ADDR + virtual_mem_size > max_va_end) {
    virtual_mem_size = max_va_end - ALLOC_BASE_ADDR;
  }
#endif
  virtual_mem_ = new MemoryAllocator(ALLOC_BASE_ADDR, virtual_mem_size, VX_VM_PAGE_SIZE, CACHE_BLOCK_SIZE);
  if (virtual_mem_ == nullptr)
    return 1;

  if (VX_VM_ADDR_MODE == BARE) {
    DBGPRINT("[RT:init_VM] VA_MODE = BARE MODE(addr= 0x0)\n");
  } else {
    CHECK_ERR(alloc_page_table(&pt_addr), { return err; });
  }

  // Stash a local SATP record so need_trans / page_table_walk see the
  // PT base. The simulator's hardware MMU sees SATP via the kernel's
  // csrw at boot — VMManager does not push it into the simulator.
  satp_ = std::make_unique<SATP_t>(pt_addr, /*asid=*/0);

  if (VX_VM_ADDR_MODE != BARE) {
    // Identity-map system regions that are PA-addressed by the kernel
    // and runtime: IO MMIO range and the high region containing the
    // page table + per-warp stacks. The kernel image is mapped later
    // via mem_reserve() once the loader knows its extents.
    CHECK_ERR(install_identity_map(0, VX_MEM_USER_BASE_ADDR), { return err; });
    CHECK_ERR(install_identity_map(VX_MEM_PAGE_TABLE_BASE_ADDR,
                                   GLOBAL_MEM_SIZE - VX_MEM_PAGE_TABLE_BASE_ADDR), {
      return err;
    });
  }
  // install_identity_map() already flushed; an extra flush here is a
  // cheap no-op if nothing else dirtied the shadow.
  return flush();
}

bool VMManager::need_trans(uint64_t dev_pAddr) {
  (void)dev_pAddr;
  // System PA regions (IO, kernel image, page table, stack) are
  // identity-mapped at boot, so every address goes through PTW
  // except those issued before SATP is set or in BARE mode.
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
      uint64_t va_range_end = VX_MEM_PAGE_TABLE_BASE_ADDR;
#if VX_VM_ADDR_MODE == SV32
      uint64_t max_va = 0xFFFFFFFFULL;
      if (va_range_end > max_va)
        va_range_end = max_va;
#endif
      uint64_t va_range_size = va_range_end - va_range_start;
      uint64_t max_pages = va_range_size >> VX_VM_PAGE_LOG2_SIZE;
      std::uniform_int_distribution<uint64_t> dist(0, max_pages - 1);
      uint64_t random_page_offset = dist(rng_);
      uint64_t candidate_va = va_range_start + (random_page_offset << VX_VM_PAGE_LOG2_SIZE);

      bool in_use = false;
      for (const auto& mapping : addr_mapping) {
        if (mapping.second == (candidate_va >> VX_VM_PAGE_LOG2_SIZE)) {
          in_use = true;
          break;
        }
      }
      if (!in_use && virtual_mem_->reserve(candidate_va, VX_VM_PAGE_SIZE) == 0) {
        vpn = candidate_va >> VX_VM_PAGE_LOG2_SIZE;
        allocated = true;
        break;
      }
    }
    if (!allocated) {
      virtual_mem_->allocate(VX_VM_PAGE_SIZE, &vpn);
      vpn >>= VX_VM_PAGE_LOG2_SIZE;
    }
  } else {
    virtual_mem_->allocate(VX_VM_PAGE_SIZE, &vpn);
    vpn >>= VX_VM_PAGE_LOG2_SIZE;
  }

  CHECK_ERR(update_page_table(ppn, vpn, pte_flags_from_access(flags)), );
  addr_mapping[ppn] = vpn;
  flush();
  return vpn;
}

int VMManager::phy_to_virt_map(uint64_t size, uint64_t* dev_pAddr, uint32_t flags) {
  if (!need_trans(*dev_pAddr))
    return 0;

  uint64_t init_pAddr = *dev_pAddr;
  // Round up: a sub-page allocation still needs one PTE. A plain
  // `size >> PAGE_LOG2` truncates to 0 for any buffer < 4 KB, leaving
  // it unmapped.
  uint64_t num_pages = (size + VX_VM_PAGE_SIZE - 1) >> VX_VM_PAGE_LOG2_SIZE;
  uint64_t base_ppn = init_pAddr >> VX_VM_PAGE_LOG2_SIZE;

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
        uint64_t va_range_end = VX_MEM_PAGE_TABLE_BASE_ADDR;
#if VX_VM_ADDR_MODE == SV32
        uint64_t max_va = 0xFFFFFFFFULL;
        if (va_range_end > max_va)
          va_range_end = max_va;
#endif
        uint64_t va_range_size = va_range_end - va_range_start;
        uint64_t max_pages = (va_range_size >> VX_VM_PAGE_LOG2_SIZE) - num_pages;
        std::uniform_int_distribution<uint64_t> dist(0, max_pages - 1);
        uint64_t random_page_offset = dist(rng_);
        uint64_t candidate_va = va_range_start + (random_page_offset << VX_VM_PAGE_LOG2_SIZE);

        bool range_available = true;
        for (uint64_t i = 0; i < num_pages && range_available; ++i) {
          uint64_t test_vpn = (candidate_va >> VX_VM_PAGE_LOG2_SIZE) + i;
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
    base_vpn = base_va >> VX_VM_PAGE_LOG2_SIZE;
  }

  uint64_t init_vAddr = (base_vpn << VX_VM_PAGE_LOG2_SIZE) | (init_pAddr & ((1 << VX_VM_PAGE_LOG2_SIZE) - 1));

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
  return flush();
}

uint8_t VMManager::alloc_page_table(uint64_t* pt_addr) {
  CHECK_ERR(page_table_mem_->allocate(VX_VM_PT_SIZE, pt_addr), { return err; });
  // Lazily materialize the shadow page (zero-initialized) and mark
  // dirty so flush() pushes the zeros to device memory at least once.
  auto& page = touch_pt_page(*pt_addr);
  std::memset(page.data(), 0, page.size());
  return 0;
}

int16_t VMManager::update_page_table(uint64_t ppn, uint64_t vpn, uint32_t pte_flag, uint8_t leaf_level) {
#if VX_VM_ADDR_MODE == SV39
  assert((((ppn >> 44) == 0) && ((vpn >> 27) == 0)) && "Upper bits are not zero!");
#else
  assert((((ppn >> 20) == 0) && ((vpn >> 20) == 0)) && "Upper 12 bits are not zero!");
#endif
  assert(leaf_level < VX_VM_PT_LEVEL && "leaf_level out of range");
  int i = VX_VM_PT_LEVEL - 1;
  vAddr_t vaddr(vpn << VX_VM_PAGE_LOG2_SIZE);
  uint64_t pte_addr = 0, pte_bytes = 0;
  uint64_t pt_addr = 0;
  uint64_t cur_base_ppn = get_base_ppn();

  while (i >= 0) {
    pte_addr = (cur_base_ppn * VX_VM_PT_SIZE) + (vaddr.vpn[i] * VX_VM_PTE_SIZE);
    pte_bytes = read_pte(pte_addr);
    PTE_t pte_chk(pte_bytes);
    bool valid = (pte_chk.v == 1) && ((pte_bytes & 0xFFFFFFFF) != 0xbaadf00d);
    if (valid && (pte_chk.r || pte_chk.w || pte_chk.x)) {
      // An existing leaf (super)page already maps this VA — a PTE with any of
      // R/W/X set is a leaf, not a pointer to the next level (RISC-V priv
      // spec). Descending into it would walk mapped data as a page table.
      // Re-mapping is idempotent when the existing leaf already yields the
      // requested translation — install_identity_map legitimately re-covers a
      // sub-range of a coarser identity superpage installed by
      // VMManager::init() — while a different target is a genuine conflict.
      uint64_t span = uint64_t(1) << (i * VM_VPN_BITS);   // 4 KB pages per level-i page
      uint64_t mapped_ppn = (pte_chk.ppn & ~(span - 1)) | (vpn & (span - 1));
      return (mapped_ppn == ppn) ? 0 : -1;
    }
    if (valid) {
      cur_base_ppn = pte_chk.ppn;   // interior node — descend
    } else {
      if (i == (int)leaf_level) {
        // Leaf: caller supplies the raw PTE permission bits.
        PTE_t new_pte(ppn << VX_VM_PAGE_LOG2_SIZE, pte_flag);
        write_pte(pte_addr, new_pte.pte_bytes);
        break;
      } else {
        // Interior: allocate next-level table; PTE_V only (RWX cleared
        // marks it as a pointer to the next-level table per the spec).
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
  // Per-level page coverage: L0 = VX_VM_PAGE_SIZE, each higher level
  // multiplies by PT entries-per-table. SV32: L1=4MB. SV39: L1=2MB, L2=1GB.
  constexpr uint64_t PT_FANOUT = VX_VM_PT_SIZE / VX_VM_PTE_SIZE;
  uint64_t level_size[VX_VM_PT_LEVEL];
  level_size[0] = VX_VM_PAGE_SIZE;
  for (uint8_t l = 1; l < VX_VM_PT_LEVEL; ++l) {
    level_size[l] = level_size[l - 1] * PT_FANOUT;
  }

  // Identity-mapped system regions get full R/W/X; the host-side ACL
  // (set via mem_access) is the actual permission boundary.
  constexpr uint32_t IDENTITY_PTE_FLAGS = PTE_V | PTE_R | PTE_W | PTE_X;

  (void)virtual_mem_->reserve(addr, size);

  uint64_t cur = addr;
  uint64_t end = addr + size;
  while (cur < end) {
    uint64_t remaining = end - cur;
    uint8_t leaf_level = 0;
    for (int l = VX_VM_PT_LEVEL - 1; l > 0; --l) {
      if ((cur % level_size[l]) == 0 && remaining >= level_size[l]) {
        leaf_level = (uint8_t)l;
        break;
      }
    }
    uint64_t vpn = cur >> VX_VM_PAGE_LOG2_SIZE;
    CHECK_ERR(update_page_table(vpn, vpn, IDENTITY_PTE_FLAGS, leaf_level), {
      return err;
    });
    cur += level_size[leaf_level];
  }
  return flush();
}

uint64_t VMManager::page_table_walk(uint64_t vAddr_bits) {
  if (!need_trans(vAddr_bits))
    return vAddr_bits;

  uint8_t level = VX_VM_PT_LEVEL;
  int i = level - 1;
  vAddr_t vaddr(vAddr_bits);
  uint64_t pte_addr = 0, pte_bytes = 0;
  uint64_t cur_base_ppn = get_base_ppn();

  while (true) {
    pte_addr = (cur_base_ppn * VX_VM_PT_SIZE) + (vaddr.vpn[i] * VX_VM_PTE_SIZE);
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
    cur_base_ppn = pte.ppn;   // leaf found at level i
    break;
  }
  // Reconstruct the physical address. For a leaf found at level i > 0 (a
  // mega/gigapage) the low VX_VM_PAGE_LOG2_SIZE + i*VM_VPN_BITS address bits
  // are the offset *within* the superpage and must come from the VA, not
  // from the (superpage-aligned) leaf PPN. For a 4 KB leaf (i == 0) this
  // reduces to the ordinary ppn<<12 | page-offset.
  const uint64_t off_mask =
      (uint64_t(1) << (VX_VM_PAGE_LOG2_SIZE + i * VM_VPN_BITS)) - 1;
  return ((cur_base_ppn << VX_VM_PAGE_LOG2_SIZE) & ~off_mask)
       | (vAddr_bits & off_mask);
}

// -- shadow PT helpers --------------------------------------------------

std::vector<uint8_t>& VMManager::touch_pt_page(uint64_t addr) {
  uint64_t page_pa = addr & ~(uint64_t)(VX_VM_PT_SIZE - 1);
  auto& page = shadow_pt_[page_pa];
  if (page.empty())
    page.resize(VX_VM_PT_SIZE, 0);
  dirty_pt_pages_.insert(page_pa);
  return page;
}

const std::vector<uint8_t>* VMManager::peek_pt_page(uint64_t addr) const {
  uint64_t page_pa = addr & ~(uint64_t)(VX_VM_PT_SIZE - 1);
  auto it = shadow_pt_.find(page_pa);
  if (it == shadow_pt_.end())
    return nullptr;
  return &it->second;
}

void VMManager::write_pte(uint64_t addr, uint64_t value) {
  auto& page = touch_pt_page(addr);
  uint64_t off = addr - (addr & ~(uint64_t)(VX_VM_PT_SIZE - 1));
  // Little-endian byte serialization, matching the device-side PTW
  // which fetches VX_VM_PTE_SIZE bytes from this address.
  for (uint64_t i = 0; i < VX_VM_PTE_SIZE; ++i) {
    page[off + i] = (value >> (i << 3)) & 0xff;
  }
}

uint64_t VMManager::read_pte(uint64_t addr) {
  const auto* page = peek_pt_page(addr);
#if VX_VM_ADDR_MODE == SV32
  uint64_t mask = 0x00000000FFFFFFFFULL;
#else
  uint64_t mask = 0xFFFFFFFFFFFFFFFFULL;
#endif
  if (page == nullptr) {
    // Unallocated PT page reads as zero (matches device DRAM init
    // pattern from cache_init / fresh allocator state).
    return 0;
  }
  uint64_t off = addr - (addr & ~(uint64_t)(VX_VM_PT_SIZE - 1));
  uint64_t v = 0;
  for (uint64_t i = 0; i < VX_VM_PTE_SIZE; ++i) {
    v |= (uint64_t)(*page)[off + i] << (i << 3);
  }
  return v & mask;
}

int VMManager::flush() {
  // One bulk device write per dirty PT page, minimizing DMA transactions.
  if (dirty_pt_pages_.empty())
    return 0;
  for (uint64_t page_pa : dirty_pt_pages_) {
    const auto& page = shadow_pt_[page_pa];
    dev_io_->write(page.data(), page_pa, page.size());
  }
  dirty_pt_pages_.clear();
  return 0;
}
