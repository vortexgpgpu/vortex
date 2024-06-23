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

#ifdef VM_ENABLE
#include <VX_config.h>
// #include <vortex.h>
//#include <utils.h>
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
#include <cassert>
#endif

using namespace vortex;

#ifdef VM_ENABLE  
#ifndef NDEBUG
#define DBGPRINT(format, ...) do { printf("[VXDRV] " format "", ##__VA_ARGS__); } while (0)
#else
#define DBGPRINT(format, ...) ((void)0)
#endif

#define CHECK_ERR(_expr, _cleanup)              \
    do {                                        \
        auto err = _expr;                       \
        if (err == 0)                           \
            break;                              \
        printf("[VXDRV] Error: '%s' returned %d!\n", #_expr, (int)err); \
        _cleanup                                \
    } while (false)

///////////////////////////////////////////////////////////////////////////////
//
#include <bitset>
#include <unistd.h>

uint64_t bits(uint64_t addr, uint8_t s_idx, uint8_t e_idx)
{
    return (addr >> s_idx) & ((1 << (e_idx - s_idx + 1)) - 1);
}
bool bit(uint64_t addr, uint8_t idx)
{
    return (addr) & (1 << idx);
}
#endif 

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
        //Set
        set_processor_satp(VM_ADDR_MODE);
#endif
  }

  ~vx_device() {
#ifdef VM_ENABLE
    this->mem_free(PAGE_TABLE_BASE_ADDR); // Right position?
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
    default:
      std::cout << "invalid caps id: " << caps_id << std::endl;
      std::abort();
      return -1;
    }
    *value = _value;
    return 0;
  }

#ifdef VM_ENABLE
  // virtual to phycial mapping
  uint64_t map_p2v(uint64_t pAddr)
  {
    return pAddr + 0xf000000;
  }
  bool need_trans(uint64_t dev_pAddr)
  {
    // Check if the this is the BARE mode
    bool isBAREMode = (get_mode() == VA_MODE::BARE);
    // Check if the address is reserved for system usage
    bool isReserved = (dev_pAddr >= PAGE_TABLE_BASE_ADDR);
    // Check if the address is reserved for IO usage
    bool isIO = (dev_pAddr < USER_BASE_ADDR);
    // Check if the address falls within the startup address range
    bool isStartAddress = (STARTUP_ADDR <= dev_pAddr) && (dev_pAddr <= (STARTUP_ADDR + 0x40000));

    // Print the boolean results for debugging purposes
    // printf("%p, %u, %u\n", (void *)dev_pAddr, isReserved, isStartAddress);

    // Return true if the address needs translation (i.e., it's not reserved and not a start address)
    return (!isBAREMode && !isReserved && !isIO && !isStartAddress);
  }

  uint64_t phy_to_virt_map(uint64_t size, uint64_t *dev_pAddr, uint32_t flags)
  {
    // DBGPRINT("====%s====\n", __PRETTY_FUNCTION__);
    DBGPRINT("  [RT:PTV_MAP] size = 0x%lx, dev_pAddr= 0x%lx, flags = 0x%x\n", size, *dev_pAddr, flags);
    DBGPRINT("  [RT:PTV_MAP] bit mode: %d\n", XLEN);

    // if (*dev_pAddr == STARTUP_ADDR || *dev_pAddr == 0x7FFFF000) {

    if (!need_trans(*dev_pAddr))
    {
      DBGPRINT("  [RT:PTV_MAP] Translation is not needed.\n");
      return 0;
    }

    uint64_t init_pAddr = *dev_pAddr;
    uint64_t init_vAddr = map_p2v(init_pAddr);
    uint64_t ppn = 0, vpn = 0;

    // dev_pAddr can be of size greater than a page, but we have to map and update
    // page tables on a page table granularity. So divide the allocation into pages.
    bool is_start = false;
    for (ppn = (*dev_pAddr) >> 12; ppn < ((*dev_pAddr) >> 12) + (size / MEM_PAGE_SIZE) + 1; ppn++)
    {
      vpn = map_p2v(ppn << 12) >> 12;
      if (is_start == false)
      {
        DBGPRINT("  [RT:PTV_MAP] Search vpn in page table:0x%lx\n", vpn);
        is_start = true;
      }
      else
      {
        DBGPRINT("  [RT:PTV_MAP] Next vpn: 0x%lx\n", vpn);
      }

      // Currently a 1-1 mapping is used, this can be changed here to support different
      // mapping schemes

      // If ppn to vpn mapping doesnt exist.
      if (addr_mapping.find(vpn) == addr_mapping.end())
      {
        // Create mapping.
        update_page_table(ppn, vpn, flags);
        addr_mapping[vpn] = ppn;
      }
    }
    DBGPRINT("  [RT:PTV_MAP] Mapped virtual addr: 0x%lx to physical addr: %lx\n", init_vAddr, init_pAddr);

    // Sanity check
    uint64_t pAddr = page_table_walk(init_vAddr);
    if (pAddr != init_pAddr)
    {
      assert(pAddr == init_pAddr && "ERROR: translated virtual Addresses are not the same with physical Address");
    }

    *dev_pAddr = init_vAddr; // commit vpn to be returned to host
    DBGPRINT("  [RT:PTV_MAP] Translated device virtual addr: 0x%lx\n", *dev_pAddr);

    return 0;
  }
#endif

  int mem_alloc(uint64_t size, int flags, uint64_t *dev_addr)
  {

    uint64_t addr;
    DBGPRINT("  [RT:mem_alloc] mem_alloc size: 0x%lx\n", size);
    CHECK_ERR(global_mem_.allocate(size, &addr), {
      return err;
    });
    CHECK_ERR(this->mem_access(addr, size, flags), {
      global_mem_.release(addr);
      return err;
    });
    *dev_addr = addr;
#ifdef VM_ENABLE
    // VM address translation
    phy_to_virt_map(size, dev_addr, flags);
#endif
    return 0;
  }

  int mem_reserve(uint64_t dev_addr, uint64_t size, int flags)
  {
    CHECK_ERR(global_mem_.reserve(dev_addr, size), {
      return err;
    });
    DBGPRINT("  [RT:mem_reserve] mem_reserve: addr: 0x%lx, size: 0x%lx\n", dev_addr, size);
    CHECK_ERR(this->mem_access(dev_addr, size, flags), {
      global_mem_.release(dev_addr);
      return err;
    });
#ifdef VM_ENABLE
    uint64_t paddr = dev_addr;
    phy_to_virt_map(size, &paddr, flags);
#endif
    return 0;
  }

  int mem_free(uint64_t dev_addr)
  {
#ifdef VM_ENABLE
    uint64_t pAddr = page_table_walk(dev_addr);
    // VM address translation
    return global_mem_.release(pAddr);
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
    // DBGPRINT("====%s====\n", __PRETTY_FUNCTION__);
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (dest_addr + asize > GLOBAL_MEM_SIZE)
      return -1;
#ifdef VM_ENABLE
    uint64_t pAddr = page_table_walk(dest_addr);
    DBGPRINT("  [RT:upload] Upload data to vAddr = 0x%lx (pAddr=0x%lx)\n", dest_addr, pAddr);
    dest_addr = pAddr; //Overwirte
#endif

    ram_.enable_acl(false);
    ram_.write((const uint8_t *)src, dest_addr, size);
    ram_.enable_acl(true);


    /*DBGPRINT("upload %ld bytes to 0x%lx\n", size, dest_addr);
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
  void set_processor_satp(VA_MODE mode)
  {
    // DBGPRINT("====%s====\n", __PRETTY_FUNCTION__);
    uint64_t satp = 0;
    if (mode == VA_MODE::BARE)
    {
      DBGPRINT("  [RT:set_satp] VA_MODE = BARE MODE");
    }
    else
    {
      satp = (alloc_2nd_level_page_table() / MEM_PAGE_SIZE) | (1 << SATP_MODE_IDX);
      DBGPRINT("  [RT:set_satp] VA_MODE = SV mode (satp = 0x%lx)\n", satp);
    }
    processor_.set_satp(satp);
  }

  uint64_t get_ptbr()
  {
    // return processor_.get_satp();
    return processor_.get_satp() & ((1 << SATP_PPN_WIDTH) - 1);
  }
  uint64_t get_pte_address(uint64_t base_page, uint64_t vpn)
  {
    return (base_page * MEM_PAGE_SIZE) + (vpn * PTE_SIZE);
  }

  VA_MODE get_mode()
  {
#ifdef XLEN_32
    return processor_.get_satp() & (1 << SATP_MODE_IDX) ? VA_MODE::SV32 : VA_MODE::BARE;
#else // 64 bit
    return processor_.get_satp() & (1 << SATP_MODE_IDX) ? VA_MODE::SV64 : VA_MODE::BARE;
#endif
  }

  void update_page_table(uint64_t ppn, uint64_t vpn, uint32_t flag)
  {
    // DBGPRINT("====%s====\n", __PRETTY_FUNCTION__);
    DBGPRINT("  [RT:Update PT] Mapping vpn 0x%05lx to ppn 0x%05lx(flags = %u)\n", vpn, ppn, flag);
    assert((((ppn >> 20) == 0) && ((vpn >> 20) == 0)) && "Upper 12 bits are not zero!");
    // Updating page table with the following mapping of (vAddr) to (pAddr).
    //  uint32_t page_bit_shift = log2ceil(PTE_SIZE*NUM_PTE_ENTRY);
    uint64_t ppn_1 = 0, pte_addr = 0, pte_bytes = 0;
    uint64_t vpn_1 = bits(vpn, 10, 19);
    uint64_t vpn_0 = bits(vpn, 0, 9);

    // Read first level PTE.
    DBGPRINT("  [RT:Update PT]Start second-level page table\n");
    pte_addr = get_pte_address(get_ptbr(), vpn_1);
    pte_bytes = read_pte(pte_addr);
    DBGPRINT("  [RT:Update PT] PTE addr 0x%lx, PTE bytes 0x%lx\n", pte_addr, pte_bytes);
    ppn_1 = (pte_bytes >> 10);

    if (bit(pte_bytes, 0) && ((pte_bytes & 0xFFFFFFFF) != 0xbaadf00d))
    {
      // If valid bit set, proceed to next level using new ppn form PTE.
      DBGPRINT("  [RT:Update PT] PTE valid (ppn 0x%lx), continuing the walk...\n", ppn_1);
    }
    else
    {
      // If valid bit not set, allocate a second level page table
      //  in device memory and store ppn in PTE. Set rwx = 000 in PTE
      // to indicate this is a pointer to the next level of the page table.
      DBGPRINT("  [RT:Update PT] PTE Invalid (ppn 0x%lx), continuing the walk...\n", ppn_1);
      ppn_1 = (alloc_1st_level_page_table(vpn_1) >> 12);
      pte_bytes = ((ppn_1 << 10) | 0b0000000001);
      assert((pte_addr >> 32) == 0 && "Upper 32 bits are not zero!");
      write_pte(pte_addr, pte_bytes);
      // if (pte_bytes != read_pte(pte_addr))
      //     DBGPRINT("Read/write values are different!\n");
    }

    DBGPRINT("  [RT:Update PT] Move to first-level page table\n");
    // Read second level PTE.
    pte_addr = get_pte_address(ppn_1, vpn_0);
    pte_bytes = read_pte(pte_addr);

    if (bit(pte_bytes, 0) && ((pte_bytes & 0xFFFFFFFF) != 0xbaadf00d))
    {
      DBGPRINT("  [RT:Update PT] ERROR, shouldn't be here\n");
      exit(1);
      // If valid bit is set, then the page is already allocated.
      // Should not reach this point, a sanity check.
    }
    else
    {
      // If valid bit not set, write ppn of pAddr in PTE. Set rwx = 111 in PTE
      // to indicate this is a leaf PTE and has the stated permissions.
      pte_bytes = ((ppn << 10) | 0b0000001111);
      write_pte(pte_addr, pte_bytes);
      if (pte_bytes != read_pte(pte_addr))
        DBGPRINT("  [RT:Update PT] PTE write value and read value are not matched!\n");
    }
  }

  uint64_t page_table_walk(uint64_t vAddr_bits)
  {
    // DBGPRINT("====%s====\n", __PRETTY_FUNCTION__);
    DBGPRINT("  [RT:PTW] start vAddr: 0x%lx\n", vAddr_bits);
    if (!need_trans(vAddr_bits))
    {
      DBGPRINT("  [RT:PTW] Translation is not needed.\n");
      return vAddr_bits;
    }
    uint64_t LEVELS = 2;
    vAddr_SV32_t vAddr(vAddr_bits);
    uint64_t pte_addr, pte_bytes;
    uint64_t pt_ba = get_ptbr() << 12;

    // Get base page table.

    for (int i = LEVELS - 1; i >= 0; i--)
    {
      // Read PTE.
      pte_addr = pt_ba + vAddr.vpn[i] * PTE_SIZE;
      pte_bytes = read_pte(pte_addr);
      PTE_SV32_t pte(pte_bytes);
      DBGPRINT("  [RT:PTW] Level[%u] pte_bytes = 0x%lx, pte flags = %u)\n", i, pte.ppn, pte.flags);

      // Check if it has invalid flag bits.
      if ((pte.v == 0) | ((pte.r == 0) & (pte.w == 1)))
      {
        std::string msg = "  [RT:PTW] Page Fault : Attempted to access invalid entry. Entry: 0x";
        throw Page_Fault_Exception(msg);
      }

      if ((pte.r == 0) & (pte.w == 0) & (pte.x == 0))
      {
        // Not a leaf node as rwx == 000
        if (i == 0)
        {
          throw Page_Fault_Exception("  [RT:PTW] Page Fault : No leaf node found.");
        }
        else
        {
          // Continue on to next level.
          pt_ba = pte.ppn << 12;
          DBGPRINT("  [RT:PTW] next pt_ba: %p\n", (void *)pt_ba);
        }
      }
      else
      {
        // Leaf node found, finished walking.
        pt_ba = pte.ppn << 12;
        DBGPRINT("  [RT:PTW] Found PT_Base_Address [%d] = %lx\n", i, pt_ba);
        break;
      }
    }

    // pte_bytes is final leaf
    PTE_SV32_t pte(pte_bytes);
    // Check RWX permissions according to access type.
    if (pte.r == 0)
    {
      throw Page_Fault_Exception("  [RT:PTW] Page Fault : TYPE LOAD, Incorrect permissions.");
    }

    uint64_t paddr = pt_ba + vAddr.pgoff;
    return paddr;
  }

  uint64_t alloc_2nd_level_page_table()
  {
    uint64_t addr = PAGE_TABLE_BASE_ADDR;
    uint64_t size = PT_TOTAL_SIZE;
    CHECK_ERR(this->mem_reserve(addr, size, VX_MEM_READ_WRITE), {
      return err;
    });
    init_page_table(addr);
    return addr;
  }
  uint64_t alloc_1st_level_page_table(uint64_t vpn_1)
  {
    uint64_t addr = PAGE_TABLE_BASE_ADDR + PT_SIZE * (1 + vpn_1);
    init_page_table(addr);
    return addr;
  }

  // Initialize to zero the target page table area. 32bit 4K, 64bit 8K
  void init_page_table(uint64_t addr)
  {
    uint64_t asize = aligned_size(PT_SIZE, CACHE_BLOCK_SIZE);
    DBGPRINT("  [RT:init_page_table] (addr=0x%lx, size=0x%lx)\n", addr, asize);
    uint8_t *src = new uint8_t[asize];
    for (uint64_t i = 0; i < PT_SIZE; ++i)
    {
      src[i] = 0;
    }
    ram_.enable_acl(false);
    ram_.write((const uint8_t *)src, addr, asize);
    ram_.enable_acl(true);
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
#endif // JAEWON

private:
  Arch arch_;
  RAM ram_;
  Processor processor_;
  MemoryAllocator global_mem_;
  DeviceConfig dcrs_;
  std::future<void> future_;
  std::unordered_map<uint32_t, std::array<uint64_t, 32>> mpm_cache_;
#ifdef VM_ENABLE
  std::unordered_map<uint64_t, uint64_t> addr_mapping;
#endif
};

#include <callbacks.inc>
