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

#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <unordered_set>
#include <stdexcept>
#include "VX_config.h"
#ifdef VM_ENABLE
#include <unordered_set>
#include <stdexcept>
#include <cassert>
#endif


namespace vortex {


#ifdef VM_ENABLE

// VA MODE
#define BARE 0x0
#define SV32 0x1
#define SV39 0x8

enum ACCESS_TYPE {
  LOAD,
  STORE,
  FETCH
};
class SATP_t
{
  private:
    uint64_t address;
    uint16_t asid;
    uint8_t  mode;
    uint64_t ppn;
    uint64_t satp;

    uint64_t bits(uint64_t input, uint8_t s_idx, uint8_t e_idx)
    {
        return (input>> s_idx) & (((uint64_t)1 << (e_idx - s_idx + 1)) - 1);
    }
    bool bit(uint64_t input , uint8_t idx)
    {
        return (input ) & ((uint64_t)1 << idx);
    }

  public:
    SATP_t(uint64_t satp) : satp(satp)
    {
#ifdef XLEN_32 
      mode = bit(satp, 31);
      asid = bits(satp, 22, 30);
      ppn  = bits(satp, 0,21);
#else
      mode = bits(satp, 60,63);
      asid = bits(satp, 44, 59);
      ppn  = bits(satp, 0,43);
#endif 
      address = ppn << MEM_PAGE_LOG2_SIZE;
    }

    SATP_t(uint64_t address, uint16_t asid) : address(address), asid(asid)
    { 
#ifdef XLEN_32 
      assert((address >> 32) == 0 && "Upper 32 bits are not zero!");
#endif
      mode= VM_ADDR_MODE;
      // asid = 0 ; 
      ppn = address >> MEM_PAGE_LOG2_SIZE;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
#ifdef XLEN_32 
      satp = (((uint64_t)mode << 31) | ((uint64_t)asid << 22) | ppn);
#else
      satp = (((uint64_t)mode << 60) | ((uint64_t)asid << 44) | ppn);
#endif
#pragma GCC diagnostic pop
    }
    uint8_t get_mode()
    {
      return mode;
    } 
    uint16_t get_asid()
    {
      return asid;
    } 
    uint64_t get_base_ppn()
    {
      return ppn;
    } 
    uint64_t get_satp()
    {
      return satp;
    } 
};


class Page_Fault_Exception : public std::runtime_error /* or logic_error */
{
public:
    Page_Fault_Exception(const std::string& what = "") : std::runtime_error(what) {}
    uint64_t addr;
    ACCESS_TYPE type;
};
#endif
struct BadAddress {};
struct OutOfRange {};

class MemDevice {
public:
  virtual ~MemDevice() {}
  virtual uint64_t size() const = 0;
  virtual void read(void* data, uint64_t addr, uint64_t size) = 0;
  virtual void write(const void* data, uint64_t addr, uint64_t size) = 0;
};

///////////////////////////////////////////////////////////////////////////////

class RamMemDevice : public MemDevice {
public:
  RamMemDevice(uint64_t size, uint32_t wordSize);
  RamMemDevice(const char* filename, uint32_t wordSize);
  ~RamMemDevice() {}

  void read(void* data, uint64_t addr, uint64_t size) override;
  void write(const void* data, uint64_t addr, uint64_t size) override;

  virtual uint64_t size() const {
    return contents_.size();
  };

protected:
  std::vector<uint8_t> contents_;
  uint32_t wordSize_;
};

///////////////////////////////////////////////////////////////////////////////

class RomMemDevice : public RamMemDevice {
public:
  RomMemDevice(const char *filename, uint32_t wordSize)
    : RamMemDevice(filename, wordSize)
  {}

  RomMemDevice(uint64_t size, uint32_t wordSize)
    : RamMemDevice(size, wordSize)
  {}

  ~RomMemDevice();

  void write(const void* data, uint64_t addr, uint64_t size) override;
};

///////////////////////////////////////////////////////////////////////////////

class MemoryUnit {
public:

// HW: Expand PageFault struct to contain access_type info for debug purposes
  struct PageFault {
    PageFault(uint64_t a, bool nf)
      : faultAddr(a)
      , notFound(nf)
      // , access_type(ACCESS_TYPE::LOAD)
    {}
    uint64_t    faultAddr;
    bool        notFound;
    // ACCESS_TYPE access_type;
  };

#ifdef VM_ENABLE
  MemoryUnit(uint64_t pageSize = MEM_PAGE_SIZE);
  ~MemoryUnit(){
    if ( this->satp_ != NULL) 
      delete this->satp_;
  };
#else
  MemoryUnit(uint64_t pageSize = 0);
#endif

  void attach(MemDevice &m, uint64_t start, uint64_t end);


#ifdef VM_ENABLE
  void read(void* data, uint64_t addr, uint32_t size, ACCESS_TYPE type = ACCESS_TYPE::LOAD);
  void write(const void* data, uint64_t addr, uint32_t size, ACCESS_TYPE type = ACCESS_TYPE::STORE);
#else
  void read(void* data, uint64_t addr, uint32_t size, bool sup);
  void write(const void* data, uint64_t addr, uint32_t size, bool sup);
#endif

  void amo_reserve(uint64_t addr);
  bool amo_check(uint64_t addr);

#ifdef VM_ENABLE
  void tlbAdd(uint64_t virt, uint64_t phys, uint32_t flags, uint64_t size_bits);
  uint8_t is_satp_unset();
  uint64_t get_satp();
  uint8_t get_mode();
  uint64_t get_base_ppn();
  void set_satp(uint64_t satp);
#else
  void tlbAdd(uint64_t virt, uint64_t phys, uint32_t flags);
#endif

  void tlbRm(uint64_t vaddr);
  void tlbFlush() {
    tlb_.clear();
  }

private:

  struct amo_reservation_t {
    uint64_t addr;
    bool     valid;
  };

  class ADecoder {
  public:
    ADecoder() {}

    void read(void* data, uint64_t addr, uint64_t size);
    void write(const void* data, uint64_t addr, uint64_t size);

    void map(uint64_t start, uint64_t end, MemDevice &md);

  private:

    struct mem_accessor_t {
      MemDevice*  md;
      uint64_t    addr;
    };

    struct entry_t {
      MemDevice*  md;
      uint64_t    start;
      uint64_t    end;
    };

    bool lookup(uint64_t addr, uint32_t wordSize, mem_accessor_t*);

    std::vector<entry_t> entries_;
  };

  struct TLBEntry {
    TLBEntry() {}
  #ifdef VM_ENABLE
    TLBEntry(uint32_t pfn, uint32_t flags, uint64_t size_bits)
      : pfn(pfn)
      , flags(flags)
      , mru_bit(true)
      , size_bits (size_bits)
    {
      d = bit(7);
      a = bit(6);
      g = bit(5);
      u = bit(4);
      x = bit(3);
      w = bit(2);
      r = bit(1);
      v = bit(0);
    }
    bool bit(uint8_t idx)
    {
        return (flags) & (1 << idx);
    }

    uint32_t pfn;
    uint32_t flags;
    bool mru_bit;
    uint64_t size_bits;
    bool d, a, g, u, x, w, r, v;
  #else
    TLBEntry(uint32_t pfn, uint32_t flags)
      : pfn(pfn)
      , flags(flags) 
    {}
    uint32_t pfn;
    uint32_t flags;
  #endif
  };

#ifdef VM_ENABLE
  std::pair<bool, uint64_t> tlbLookup(uint64_t vAddr, ACCESS_TYPE type, uint64_t* size_bits);

  bool need_trans(uint64_t dev_pAddr);
  uint64_t vAddr_to_pAddr(uint64_t vAddr, ACCESS_TYPE type);

  uint64_t get_pte_address(uint64_t base_ppn, uint64_t vpn);
  std::pair<uint64_t, uint8_t> page_table_walk(uint64_t vAddr_bits, ACCESS_TYPE type, uint64_t* size_bits);
#else 
  uint64_t toPhyAddr(uint64_t vAddr, uint32_t flagMask);
  TLBEntry tlbLookup(uint64_t vAddr, uint32_t flagMask);
#endif



  std::unordered_map<uint64_t, TLBEntry> tlb_;
  uint64_t  pageSize_;
  ADecoder  decoder_;
#ifndef VM_ENABLE
  bool      enableVM_;
#endif

  amo_reservation_t amo_reservation_;
#ifdef VM_ENABLE
  std::unordered_set<uint64_t> unique_translations;
  uint64_t TLB_HIT, TLB_MISS, TLB_EVICT, PTW, PERF_UNIQUE_PTW;
  SATP_t *satp_;
#endif

};

///////////////////////////////////////////////////////////////////////////////

class ACLManager {
public:

    void set(uint64_t addr, uint64_t size, int flags);

    bool check(uint64_t addr, uint64_t size, int flags) const;

private:

  struct acl_entry_t {
    uint64_t end;
    int32_t flags;
  };

  std::map<uint64_t, acl_entry_t> acl_map_;
};

///////////////////////////////////////////////////////////////////////////////

class RAM : public MemDevice {
public:

  RAM(uint64_t capacity, uint32_t page_size);
  RAM(uint64_t capacity) : RAM(capacity, capacity) {}
  ~RAM();

  void clear();

  uint64_t size() const override;

  void read(void* data, uint64_t addr, uint64_t size) override;
  void write(const void* data, uint64_t addr, uint64_t size) override;

  void loadBinImage(const char* filename, uint64_t destination);
  void loadHexImage(const char* filename);

  uint8_t& operator[](uint64_t address) {
    return *this->get(address);
  }

  const uint8_t& operator[](uint64_t address) const {
    return *this->get(address);
  }

  void set_acl(uint64_t addr, uint64_t size, int flags);

  void enable_acl(bool enable) {
    check_acl_ = enable;
  }

private:

  uint8_t *get(uint64_t address) const;

  uint64_t capacity_;
  uint32_t page_bits_;
  mutable std::unordered_map<uint64_t, uint8_t*> pages_;
  mutable uint8_t* last_page_;
  mutable uint64_t last_page_index_;
  ACLManager acl_mngr_;
  bool check_acl_;
};

#ifdef VM_ENABLE
class PTE_t 
{

  private:
    uint64_t address;
    uint64_t bits(uint64_t input, uint8_t s_idx, uint8_t e_idx)
    {
        return (input>> s_idx) & (((uint64_t)1 << (e_idx - s_idx + 1)) - 1);
    }
    bool bit(uint64_t input, uint8_t idx)
    {
        return (input) & ((uint64_t)1 << idx);
    }

  public:
#if VM_ADDR_MODE == SV39
    bool N;
    uint8_t PBMT;
#endif
    uint64_t ppn;
    uint32_t rsw;
    uint32_t flags;
    uint8_t level;
    bool d, a, g, u, x, w, r, v;
    uint64_t pte_bytes;

    void set_flags (uint32_t flag)
    {
      this->flags = flag;
      d = bit(flags,7);
      a = bit(flags,6);
      g = bit(flags,5);
      u = bit(flags,4);
      x = bit(flags,3);
      w = bit(flags,2);
      r = bit(flags,1);
      v = bit(flags,0);
    }

    PTE_t(uint64_t address, uint32_t flags) : address(address)
    {
#if VM_ADDR_MODE == SV39
      N = 0;
      PBMT = 0;
      level = 3;
      ppn = address >> MEM_PAGE_LOG2_SIZE;
      // Reserve for Super page support
      // ppn = new uint32_t [level];
      // ppn[2]=bits(address,28,53);
      // ppn[1]=bits(address,19,27);
      // ppn[0]=bits(address,10,18);
      set_flags(flags);
      // pte_bytes = (N  << 63) | (PBMT  << 61) | (ppn <<10) | flags ;
      pte_bytes = (ppn <<10) | flags ;
#else // if VM_ADDR_MODE == SV32
      assert((address>> 32) == 0 && "Upper 32 bits are not zero!");
      level = 2;
      ppn = address >> MEM_PAGE_LOG2_SIZE;
      // Reserve for Super page support
      // ppn = new uint32_t[level];
      // ppn[1]=bits(address,20,31);
      // ppn[0]=bits(address,10,19);
      set_flags(flags);
      pte_bytes = ppn <<10 | flags ;
#endif
    }

    PTE_t(uint64_t pte_bytes) : pte_bytes(pte_bytes)
    { 
#if VM_ADDR_MODE == SV39
      N = bit(pte_bytes,63);
      PBMT = bits(pte_bytes,61,62);
      level = 3;
      ppn=bits(pte_bytes,10,53);
      address = ppn << MEM_PAGE_LOG2_SIZE; 
      // Reserve for Super page support
      // ppn = new uint32_t [level];
      // ppn[2]=bits(pte_bytes,28,53);
      // ppn[1]=bits(pte_bytes,19,27);
      // ppn[0]=bits(pte_bytes,10,18);
#else //#if VM_ADDR_MODE == SV32
      assert((pte_bytes >> 32) == 0 && "Upper 32 bits are not zero!");
      level = 2;
      ppn=bits(pte_bytes,10, 31);
      address = ppn << MEM_PAGE_LOG2_SIZE; 
      // Reserve for Super page support
      // ppn = new uint32_t[level];
      // ppn[1]=bits(address, 20,31);
      // ppn[0]=bits(address, 10,19);
#endif
      rsw = bits(pte_bytes,8,9);
      set_flags((uint32_t)(bits(pte_bytes,0,7)));
    }
    ~PTE_t()
    {
      // Reserve for Super page support
      // delete ppn;
    }
};

class vAddr_t 
{

  private:
    uint64_t address;
    uint64_t bits(uint8_t s_idx, uint8_t e_idx)
    {
        return (address>> s_idx) & (((uint64_t)1 << (e_idx - s_idx + 1)) - 1);
    }
    bool bit( uint8_t idx)
    {
        return (address) & ((uint64_t)1 << idx);
    }

  public:
    uint64_t *vpn;
    uint64_t pgoff;
    uint8_t level;
    vAddr_t(uint64_t address) : address(address)
    {
#if VM_ADDR_MODE == SV39
      level = 3;
      vpn = new uint64_t [level];
      vpn[2] = bits(30,38);
      vpn[1] = bits(21,29);
      vpn[0] = bits(12,20);
      pgoff = bits(0,11);
#else //#if VM_ADDR_MODE == SV32
      assert((address>> 32) == 0 && "Upper 32 bits are not zero!");
      level = 2;
      vpn = new uint64_t [level];
      vpn[1] = bits(22,31);
      vpn[0] = bits(12,21);
      pgoff = bits(0,11);
#endif
    }

    ~vAddr_t()
    {
      delete vpn;
    }
};
#endif

} // namespace vortex
