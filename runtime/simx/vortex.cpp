// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <common.h>

#include <util.h>
#include <processor.h>
#include <arch.h>
#include <mem.h>
#include <constants.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <future>
#include <chrono>

#include <VX_config.h>
#ifdef VM_ENABLE
#include <malloc.h>

#include <VX_types.h>

#include <util.h>

#include <processor.h>
#include <arch.h>
#include <mem.h>
#include <constants.h>
#include <unordered_map>
#include <array>
#include <cmath>
#endif

using namespace vortex;

class vx_device {
public:
    vx_device()
        : arch_(NUM_THREADS, NUM_WARPS, NUM_CORES)
        , ram_(0, MEM_PAGE_SIZE)
        , processor_(arch_)
        , global_mem_(ALLOC_BASE_ADDR, GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR, MEM_PAGE_SIZE, CACHE_BLOCK_SIZE)
    {
        // attach memory module
        processor_.attach_ram(&ram_);
#ifdef VM_ENABLE
	std::cout << "*** VM ENABLED!! ***"<< std::endl;
        CHECK_ERR(init_VM(), );
#endif
    }

  ~vx_device() {
#ifdef VM_ENABLE
  global_mem_.release(PAGE_TABLE_BASE_ADDR);
  // for (auto i = addr_mapping.begin(); i != addr_mapping.end(); i++)
  //   page_table_mem_->release(i->second << MEM_PAGE_SIZE);
  delete virtual_mem_;
  delete page_table_mem_;
#endif
    if (future_.valid()) {
      future_.wait();
    }
  }

  int init() {
    return 0;
  }

  int get_caps(uint32_t caps_id, uint64_t *value) {
    uint64_t _value;
    switch (caps_id) {
    case VX_CAPS_VERSION:
      _value = IMPLEMENTATION_ID;
      break;
    case VX_CAPS_NUM_THREADS:
      _value = NUM_THREADS;
      break;
    case VX_CAPS_NUM_WARPS:
      _value = NUM_WARPS;
      break;
    case VX_CAPS_NUM_CORES:
      _value = NUM_CORES * NUM_CLUSTERS;
      break;
    case VX_CAPS_TC_SIZE:
      _value = TC_SIZE;
      break;
    case VX_CAPS_TC_NUM:
      _value = TC_NUM;
      break;
    case VX_CAPS_CACHE_LINE_SIZE:
      _value = CACHE_BLOCK_SIZE;
      break;
    case VX_CAPS_GLOBAL_MEM_SIZE:
      _value = GLOBAL_MEM_SIZE;
      break;
    case VX_CAPS_LOCAL_MEM_SIZE:
      _value = (1 << LMEM_LOG_SIZE);
      break;
    case VX_CAPS_ISA_FLAGS:
      _value = ((uint64_t(MISA_EXT))<<32) | ((log2floor(XLEN)-4) << 30) | MISA_STD;
      break;
    case VX_CAPS_NUM_MEM_BANKS:
      _value = PLATFORM_MEMORY_NUM_BANKS;
      break;
    case VX_CAPS_MEM_BANK_SIZE:
      _value = 1ull << (MEM_ADDR_WIDTH / PLATFORM_MEMORY_NUM_BANKS);
      break;
    default:
      std::cout << "invalid caps id: " << caps_id << std::endl;
      std::abort();
      return -1;
    }
    *value = _value;
    return 0;
  }

#ifdef VM_ENABLE

  // physical (ppn) to virtual (vpn) mapping
  uint64_t map_p2v(uint64_t ppn, uint32_t flags)
  {
    DBGPRINT(" [RT:MAP_P2V] ppn: %lx\n", ppn);
    if (addr_mapping.find(ppn) != addr_mapping.end()) return addr_mapping[ppn];

    // If ppn to vpn mapping doesnt exist, create mapping
    DBGPRINT(" [RT:MAP_P2V] Not found. Allocate new page table or update a PTE.\n");
    uint64_t vpn;
    virtual_mem_->allocate(MEM_PAGE_SIZE, &vpn);
    vpn = vpn >> MEM_PAGE_LOG2_SIZE;
    CHECK_ERR(update_page_table(ppn, vpn, flags),);
    addr_mapping[ppn] = vpn;
    return vpn;
  }

  bool need_trans(uint64_t dev_pAddr)
  {

    // Check if the satp is set and BARE mode
    if (processor_.is_satp_unset() || get_mode() == BARE)
      return 0;

    // Check if the address is reserved for system usage
    // bool isReserved = (PAGE_TABLE_BASE_ADDR <= dev_pAddr && dev_pAddr < PAGE_TABLE_BASE_ADDR + PT_SIZE_LIMIT);
    if (PAGE_TABLE_BASE_ADDR <= dev_pAddr)
      return 0;

    // Check if the address is reserved for IO usage
    if (dev_pAddr < USER_BASE_ADDR)
      return 0;
    // Check if the address falls within the startup address range
    if ((STARTUP_ADDR <= dev_pAddr) && (dev_pAddr <= (STARTUP_ADDR + 0x40000)))
      return 0;

    // Now all conditions are not met. Return true because the address needs translation
    return 1;
  }

  uint64_t phy_to_virt_map(uint64_t size, uint64_t *dev_pAddr, uint32_t flags)
  {
    DBGPRINT(" [RT:PTV_MAP] size = 0x%lx, dev_pAddr= 0x%lx, flags = 0x%x\n", size, *dev_pAddr, flags);
    DBGPRINT(" [RT:PTV_MAP] bit mode: %d\n", XLEN);

    if (!need_trans(*dev_pAddr))
    {
      DBGPRINT(" [RT:PTV_MAP] Translation is not needed.\n");
      return 0;
    }

    uint64_t init_pAddr = *dev_pAddr;
    uint64_t init_vAddr = (map_p2v(init_pAddr >> MEM_PAGE_LOG2_SIZE, flags) << MEM_PAGE_LOG2_SIZE) | (init_pAddr & ((1 << MEM_PAGE_LOG2_SIZE) - 1));
    uint64_t ppn = 0, vpn = 0;

    // dev_pAddr can be of size greater than a page, but we have to map and update
    // page tables on a page table granularity. So divide the allocation into pages.
    // FUTURE Work: Super Page
    for (ppn = (*dev_pAddr >> MEM_PAGE_LOG2_SIZE); ppn < ((*dev_pAddr) >> MEM_PAGE_LOG2_SIZE) + (size >> MEM_PAGE_LOG2_SIZE) ; ppn++)
    {
      vpn = map_p2v(ppn, flags) >> MEM_PAGE_LOG2_SIZE;
      DBGPRINT(" [RT:PTV_MAP] Search vpn in page table:0x%lx\n", vpn);
      // Currently a 1-1 mapping is used, this can be changed here to support different
      // mapping schemes
    }
    DBGPRINT(" [RT:PTV_MAP] Mapped virtual addr: 0x%lx to physical addr: 0x%lx\n", init_vAddr, init_pAddr);
    // Sanity check
    assert(page_table_walk(init_vAddr) == init_pAddr && "ERROR: translated virtual Addresses are not the same with physical Address\n");

    *dev_pAddr = init_vAddr; // commit vpn to be returned to host
    DBGPRINT(" [RT:PTV_MAP] Translated device virtual addr: 0x%lx\n", *dev_pAddr);

    return 0;
  }
#endif

  int mem_alloc(uint64_t size, int flags, uint64_t *dev_addr)
  {
    uint64_t asize = aligned_size(size, MEM_PAGE_SIZE);
    uint64_t addr = 0;

    DBGPRINT("[RT:mem_alloc] size: 0x%lx, asize, 0x%lx,flag : 0x%d\n", size, asize, flags);
    // HW: when vm is supported this global_mem_ should be virtual memory allocator
    CHECK_ERR(global_mem_.allocate(asize, &addr), {
      return err;
    });
    CHECK_ERR(this->mem_access(addr, asize, flags), {
      global_mem_.release(addr);
      return err;
    });
    *dev_addr = addr;
#ifdef VM_ENABLE
    // VM address translation
    phy_to_virt_map(asize, dev_addr, flags);
#endif
    return 0;
  }

  int mem_reserve(uint64_t dev_addr, uint64_t size, int flags)
  {
    uint64_t asize = aligned_size(size, MEM_PAGE_SIZE);
    CHECK_ERR(global_mem_.reserve(dev_addr, asize), {
      return err;
    });
    DBGPRINT("[RT:mem_reserve] addr: 0x%lx, asize:0x%lx, size: 0x%lx\n", dev_addr, asize, size);
    CHECK_ERR(this->mem_access(dev_addr, asize, flags), {
      global_mem_.release(dev_addr);
      return err;
    });
    return 0;
  }

  int mem_free(uint64_t dev_addr)
  {
#ifdef VM_ENABLE
    uint64_t paddr = page_table_walk(dev_addr);
    return global_mem_.release(paddr);
#else
    return global_mem_.release(dev_addr);
#endif
  }

  int mem_access(uint64_t dev_addr, uint64_t size, int flags)
  {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (dev_addr + asize > GLOBAL_MEM_SIZE)
      return -1;

    ram_.set_acl(dev_addr, size, flags);
    return 0;
  }

  int mem_info(uint64_t *mem_free, uint64_t *mem_used) const
  {
    if (mem_free)
      *mem_free = global_mem_.free();
    if (mem_used)
      *mem_used = global_mem_.allocated();
    return 0;
  }

  int upload(uint64_t dest_addr, const void *src, uint64_t size)
  {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (dest_addr + asize > GLOBAL_MEM_SIZE)
      return -1;
#ifdef VM_ENABLE
    uint64_t pAddr = page_table_walk(dest_addr);
    // uint64_t pAddr;
    // try {
    //   pAddr = page_table_walk(dest_addr);
    // } catch ( Page_Fault_Exception ) {
    //   // HW: place holder
    //   // should be virt_to_phy_map here
    //   phy_to_virt_map(0, dest_addr, 0);
    // }
    DBGPRINT("  [RT:upload] Upload data to vAddr = 0x%lx (pAddr=0x%lx)\n", dest_addr, pAddr);
    dest_addr = pAddr; //Overwirte
#endif

    ram_.enable_acl(false);
    ram_.write((const uint8_t *)src, dest_addr, size);
    ram_.enable_acl(true);

    /*
    DBGPRINT("upload %ld bytes to 0x%lx\n", size, dest_addr);
    for (uint64_t i = 0; i < size && i < 1024; i += 4) {
        DBGPRINT("  0x%lx <- 0x%x\n", dest_addr + i, *(uint32_t*)((uint8_t*)src + i));
    }*/

    return 0;
  }

  int download(void *dest, uint64_t src_addr, uint64_t size)
  {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (src_addr + asize > GLOBAL_MEM_SIZE)
      return -1;
#ifdef VM_ENABLE
    uint64_t pAddr = page_table_walk(src_addr);
    DBGPRINT("  [RT:download] Download data to vAddr = 0x%lx (pAddr=0x%lx)\n", src_addr, pAddr);
    src_addr = pAddr; //Overwirte
#endif

    ram_.enable_acl(false);
    ram_.read((uint8_t *)dest, src_addr, size);
    ram_.enable_acl(true);

    /*DBGPRINT("download %ld bytes from 0x%lx\n", size, src_addr);
    for (uint64_t i = 0; i < size && i < 1024; i += 4) {
        DBGPRINT("  0x%lx -> 0x%x\n", src_addr + i, *(uint32_t*)((uint8_t*)dest + i));
    }*/

    return 0;
  }

  int start(uint64_t krnl_addr, uint64_t args_addr)
  {
    // ensure prior run completed
    if (future_.valid())
    {
      future_.wait();
    }

    // set kernel info
    this->dcr_write(VX_DCR_BASE_STARTUP_ADDR0, krnl_addr & 0xffffffff);
    this->dcr_write(VX_DCR_BASE_STARTUP_ADDR1, krnl_addr >> 32);
    this->dcr_write(VX_DCR_BASE_STARTUP_ARG0, args_addr & 0xffffffff);
    this->dcr_write(VX_DCR_BASE_STARTUP_ARG1, args_addr >> 32);

    // start new run
    future_ = std::async(std::launch::async, [&]
                         { processor_.run(); });

    // clear mpm cache
    mpm_cache_.clear();

    return 0;
  }

  int ready_wait(uint64_t timeout)
  {
    if (!future_.valid())
      return 0;
    uint64_t timeout_sec = timeout / 1000;
    std::chrono::seconds wait_time(1);
    for (;;)
    {
      // wait for 1 sec and check status
      auto status = future_.wait_for(wait_time);
      if (status == std::future_status::ready)
        break;
      if (0 == timeout_sec--)
        return -1;
    }
    return 0;
  }

  int dcr_write(uint32_t addr, uint32_t value)
  {
    if (future_.valid())
    {
      future_.wait(); // ensure prior run completed
    }
    processor_.dcr_write(addr, value);
    dcrs_.write(addr, value);
    return 0;
  }

  int dcr_read(uint32_t addr, uint32_t *value) const
  {
    return dcrs_.read(addr, value);
  }

  int mpm_query(uint32_t addr, uint32_t core_id, uint64_t *value)
  {
    uint32_t offset = addr - VX_CSR_MPM_BASE;
    if (offset > 31)
      return -1;
    if (mpm_cache_.count(core_id) == 0)
    {
      uint64_t mpm_mem_addr = IO_MPM_ADDR + core_id * 32 * sizeof(uint64_t);
      CHECK_ERR(this->download(mpm_cache_[core_id].data(), mpm_mem_addr, 32 * sizeof(uint64_t)), {
        return err;
      });
    }
    *value = mpm_cache_.at(core_id).at(offset);
    return 0;
  }
#ifdef VM_ENABLE
  /* VM Management */

  // Initialize to zero the target page table area. 32bit 4K, 64bit 8K
  uint16_t init_page_table(uint64_t addr, uint64_t size)
  {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    DBGPRINT("  [RT:init_page_table] (addr=0x%lx, size=0x%lx)\n", addr, asize);
    uint8_t *src = new uint8_t[asize];
    if (src == NULL)
      return 1;

    for (uint64_t i = 0; i < asize; ++i)
    {
      src[i] = 0;
    }
    ram_.enable_acl(false);
    ram_.write((const uint8_t *)src, addr, asize);
    ram_.enable_acl(true);
    return 0;
  }

  uint8_t alloc_page_table (uint64_t * pt_addr)
  {
      CHECK_ERR(page_table_mem_->allocate(PT_SIZE, pt_addr), { return err; });
      CHECK_ERR(init_page_table(*pt_addr, PT_SIZE), { return err; });
      DBGPRINT("   [RT:alloc_page_table] addr= 0x%lx\n", *pt_addr);
      return 0;
  }

  // reserve IO space, startup space, and local mem area
  int virtual_mem_reserve(uint64_t dev_addr, uint64_t size, int flags)
  {
    CHECK_ERR(virtual_mem_->reserve(dev_addr, size), {
      return err;
    });
    DBGPRINT("[RT:mem_reserve] addr: 0x%lx, size:0x%lx, size: 0x%lx\n", dev_addr, size, size);
    return 0;
  }

  int16_t init_VM()
  {
    uint64_t pt_addr = 0;
    // Reserve space for PT
    DBGPRINT("[RT:init_VM] Initialize VM\n");
    DBGPRINT("* VM_ADDR_MODE=0x%lx", VM_ADDR_MODE);
    DBGPRINT("* PAGE_TABLE_BASE_ADDR=0x%lx", PAGE_TABLE_BASE_ADDR);
    DBGPRINT("* PT_LEVEL=0x%lx", PT_LEVEL);
    DBGPRINT("* PT_SIZE=0x%lx", PT_SIZE);
    DBGPRINT("* PTE_SIZE=0x%lx", PTE_SIZE);
    DBGPRINT("* TLB_SIZE=0x%lx", TLB_SIZE);
    CHECK_ERR(mem_reserve(PAGE_TABLE_BASE_ADDR, PT_SIZE_LIMIT, VX_MEM_READ_WRITE), {
      return err;
    });
    page_table_mem_ = new MemoryAllocator (PAGE_TABLE_BASE_ADDR, PT_SIZE_LIMIT, MEM_PAGE_SIZE, CACHE_BLOCK_SIZE);
    if (page_table_mem_ == NULL)
    {
      CHECK_ERR(this->mem_free(PAGE_TABLE_BASE_ADDR),);
      return 1;
    }

    // HW: virtual mem allocator has the same address range as global_mem. next step is to adjust it
    virtual_mem_ = new MemoryAllocator(ALLOC_BASE_ADDR, (GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR), MEM_PAGE_SIZE, CACHE_BLOCK_SIZE);
    CHECK_ERR(virtual_mem_reserve(PAGE_TABLE_BASE_ADDR, (GLOBAL_MEM_SIZE - PAGE_TABLE_BASE_ADDR), VX_MEM_READ_WRITE), {
      return err;
    });
    CHECK_ERR(virtual_mem_reserve(STARTUP_ADDR, 0x40000, VX_MEM_READ_WRITE), {
      return err;
    });

    if (virtual_mem_ == nullptr) {
      // virtual_mem_ does not intefere with physical mem, so no need to free space

      return 1;
    }

    if (VM_ADDR_MODE == BARE)
      DBGPRINT("[RT:init_VM] VA_MODE = BARE MODE(addr= 0x0)");
    else
      CHECK_ERR(alloc_page_table(&pt_addr),{return err;});

    CHECK_ERR(processor_.set_satp_by_addr(pt_addr),{return err;});
    return 0;
  }

  // Return value in in ptbr
  uint64_t get_base_ppn()
  {
    return processor_.get_base_ppn();
  }
  uint64_t get_pte_address(uint64_t base_ppn, uint64_t vpn)
  {
    return (base_ppn * PT_SIZE) + (vpn * PTE_SIZE);
  }

  uint8_t get_mode()
  {
    return processor_.get_satp_mode();
  }

  int16_t update_page_table(uint64_t ppn, uint64_t vpn, uint32_t flag)
  {
    DBGPRINT("  [RT:Update PT] Mapping vpn 0x%05lx to ppn 0x%05lx(flags = %u)\n", vpn, ppn, flag);
    // sanity check
#if VM_ADDR_MODE == SV39
    assert((((ppn >> 44) == 0) && ((vpn >> 27) == 0)) && "Upper bits are not zero!");
    uint8_t level = 3;
#else // Default is SV32, BARE will not reach this point.
    assert((((ppn >> 20) == 0) && ((vpn >> 20) == 0)) && "Upper 12 bits are not zero!");
    uint8_t level = 2;
#endif
    int i = level - 1;
    vAddr_t vaddr(vpn << MEM_PAGE_LOG2_SIZE);
    uint64_t pte_addr = 0, pte_bytes = 0;
    uint64_t pt_addr = 0;
    uint64_t cur_base_ppn = get_base_ppn();

    while (i >= 0)
    {
      DBGPRINT("  [RT:Update PT]Start %u-level page table\n", i);
      pte_addr = get_pte_address(cur_base_ppn, vaddr.vpn[i]);
      pte_bytes = read_pte(pte_addr);
      PTE_t pte_chk(pte_bytes);
      DBGPRINT("  [RT:Update PT] PTE addr 0x%lx, PTE bytes 0x%lx\n", pte_addr, pte_bytes);
      if (pte_chk.v == 1 && ((pte_bytes & 0xFFFFFFFF) != 0xbaadf00d))
      {
        DBGPRINT("  [RT:Update PT] PTE valid (ppn 0x%lx), continuing the walk...\n", pte_chk.ppn);
        cur_base_ppn = pte_chk.ppn;
      }
      else
      {
        // If valid bit not set, allocate a next level page table
        DBGPRINT("  [RT:Update PT] PTE Invalid (ppn 0x%lx) ...\n", pte_chk.ppn);
        if (i == 0)
        {
          // Reach to leaf
          DBGPRINT("  [RT:Update PT] Reached to level 0. This should be a leaf node(flag = %x) \n",flag);
          uint32_t pte_flag = (flag << 1) | 0x3;
          PTE_t new_pte(ppn <<MEM_PAGE_LOG2_SIZE, pte_flag);
          write_pte(pte_addr, new_pte.pte_bytes);
          break;
        }
        else
        {
          //  in device memory and store ppn in PTE. Set rwx = 000 in PTE
          // to indicate this is a pointer to the next level of the page table.
          // flag would READ: 0x1, Write 0x2, RW:0x3, which is matched with PTE flags if it is lsh by one.
          alloc_page_table(&pt_addr);
          uint32_t pte_flag = 0x1;
          PTE_t new_pte(pt_addr, pte_flag);
          write_pte(pte_addr, new_pte.pte_bytes);
          cur_base_ppn = new_pte.ppn;
        }
      }
      i--;
    }
    return 0;
  }

  uint64_t page_table_walk(uint64_t vAddr_bits)
  {
    DBGPRINT("  [RT:PTW] start vAddr: 0x%lx\n", vAddr_bits);
    if (!need_trans(vAddr_bits))
    {
      DBGPRINT("  [RT:PTW] Translation is not needed.\n");
      return vAddr_bits;
    }
    uint8_t level = PT_LEVEL;
    int i = level-1;
    vAddr_t vaddr(vAddr_bits);
    uint64_t pte_addr = 0, pte_bytes = 0;
    uint64_t cur_base_ppn = get_base_ppn();
    while (true)
    {
      DBGPRINT("  [RT:PTW]Start %u-level page table walk\n",i);
      // Read PTE.
      pte_addr = get_pte_address(cur_base_ppn, vaddr.vpn[i]);
      pte_bytes = read_pte(pte_addr);
      PTE_t pte(pte_bytes);
      DBGPRINT("  [RT:PTW] PTE addr 0x%lx, PTE bytes 0x%lx\n", pte_addr, pte_bytes);

      assert(((pte.pte_bytes & 0xFFFFFFFF) != 0xbaadf00d) && "ERROR: uninitialzed PTE\n" );
      // Check if it has invalid flag bits.
      if ((pte.v == 0) | ((pte.r == 0) & (pte.w == 1)))
      {
        std::string msg = "  [RT:PTW] Page Fault : Attempted to access invalid entry.";
        throw Page_Fault_Exception(msg);
      }

      if ((pte.r == 0) & (pte.w == 0) & (pte.x == 0))
      {
        i--;
        // Not a leaf node as rwx == 000
        if (i < 0)
        {
          throw Page_Fault_Exception("  [RT:PTW] Page Fault : No leaf node found.");
        }
        else
        {
          // Continue on to next level.
          cur_base_ppn= pte.ppn ;
          DBGPRINT("  [RT:PTW] next base_ppn: 0x%lx\n", cur_base_ppn);
          continue;
        }
      }
      else
      {
        // Leaf node found.
        // Check RWX permissions according to access type.
        if (pte.r == 0)
        {
          throw Page_Fault_Exception("  [RT:PTW] Page Fault : TYPE LOAD, Incorrect permissions.");
        }
        cur_base_ppn= pte.ppn ;
        DBGPRINT("  [RT:PTW] Found PT_Base_Address(0x%lx) on Level %d.\n", pte.ppn,i);
        break;
      }
    }
    uint64_t paddr = (cur_base_ppn << MEM_PAGE_LOG2_SIZE) + vaddr.pgoff;
    return paddr;
  }

  // void read_page_table(uint64_t addr) {
  //     uint8_t *dest = new uint8_t[MEM_PAGE_SIZE];
  //     download(dest,  addr,  MEM_PAGE_SIZE);
  //     DBGPRINT("VXDRV: download %d bytes from 0x%x\n", MEM_PAGE_SIZE, addr);
  //     for (int i = 0; i < MEM_PAGE_SIZE; i += 4) {
  //         DBGPRINT("mem-read: 0x%x -> 0x%x\n", addr + i, *(uint64_t*)((uint8_t*)dest + i));
  //     }
  // }

  void write_pte(uint64_t addr, uint64_t value = 0xbaadf00d)
  {
    DBGPRINT("  [RT:Write_pte] writing pte 0x%lx to pAddr: 0x%lx\n", value, addr);
    uint8_t *src = new uint8_t[PTE_SIZE];
    for (uint64_t i = 0; i < PTE_SIZE; ++i)
    {
      src[i] = (value >> (i << 3)) & 0xff;
    }
    // std::cout << "writing PTE to RAM addr 0x" << std::hex << addr << std::endl;
    ram_.enable_acl(false);
    ram_.write((const uint8_t *)src, addr, PTE_SIZE);
    ram_.enable_acl(true);
  }

  uint64_t read_pte(uint64_t addr)
  {
    uint8_t *dest = new uint8_t[PTE_SIZE];
#ifdef XLEN_32
    uint64_t mask = 0x00000000FFFFFFFF;
#else // 64bit
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
#endif

    ram_.read((uint8_t *)dest, addr, PTE_SIZE);
    uint64_t ret = (*(uint64_t *)((uint8_t *)dest)) & mask;
    DBGPRINT("  [RT:read_pte] reading PTE 0x%lx from RAM addr 0x%lx\n", ret, addr);

    return ret;
  }
#endif // VM_ENABLE

private:
  Arch arch_;
  RAM ram_;
  Processor processor_;
  MemoryAllocator global_mem_;
  DeviceConfig dcrs_;
  std::future<void> future_;
  std::unordered_map<uint32_t, std::array<uint64_t, 32>> mpm_cache_;
#ifdef VM_ENABLE
  std::unordered_map<uint64_t, uint64_t> addr_mapping; // HW: key: ppn; value: vpn
  MemoryAllocator* page_table_mem_;
  MemoryAllocator* virtual_mem_;
#endif
};

#include <callbacks.inc>
