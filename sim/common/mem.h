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

// Pure SV32/SV39 value classes (SATP_t / PTE_t / vAddr_t /
// Page_Fault_Exception / ACCESS_TYPE / BARE-SV32-SV39 constants) live
// in sw/common — shared by the host runtime's VMManager and the
// simulator's MMU model. Pulled in here so existing sim consumers
// keep working without changes.
#include <vm_types.h>


namespace vortex {


class BadAddress : public std::runtime_error {
public:
  BadAddress() : std::runtime_error("invalid memory address") {}
};

class OutOfRange : public std::runtime_error {
public:
  OutOfRange() : std::runtime_error("out of range memory address") {}
};

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

#ifdef VX_CFG_VM_ENABLE
  MemoryUnit(uint64_t pageSize = VX_CFG_MEM_PAGE_SIZE);
  ~MemoryUnit(){
    if ( this->satp_ != NULL)
      delete this->satp_;
  };
#else
  MemoryUnit(uint64_t pageSize = 0);
#endif

  void attach(MemDevice &m, uint64_t start, uint64_t end);


#ifdef VX_CFG_VM_ENABLE
  void read(void* data, uint64_t addr, uint32_t size, ACCESS_TYPE type = ACCESS_TYPE::LOAD);
  void write(const void* data, uint64_t addr, uint32_t size, ACCESS_TYPE type = ACCESS_TYPE::STORE);
#else
  void read(void* data, uint64_t addr, uint32_t size, bool sup);
  void write(const void* data, uint64_t addr, uint32_t size, bool sup);
#endif

  void amo_reserve(uint64_t addr);
  bool amo_check(uint64_t addr);

#ifdef VX_CFG_VM_ENABLE
  void tlbAdd(uint64_t virt, uint64_t phys, uint32_t flags, uint64_t size_bits);
  uint8_t is_satp_unset();
  uint64_t get_satp();
  uint8_t get_mode();
  uint64_t get_base_ppn();
  void set_satp(uint64_t satp);
  // Public translator entry point used by simx Core::translate to feed
  // translated PAs into LsuReq / icache MemReq.
  uint64_t vAddr_to_pAddr(uint64_t vAddr, ACCESS_TYPE type);
#else
  void tlbAdd(uint64_t virt, uint64_t phys, uint32_t flags);
#endif

  void tlbRm(uint64_t vaddr);
  void tlbFlush() {
    for (auto& entry : tlb_) {
      entry.valid = false;
    }
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
    TLBEntry() : valid(false) {}
  #ifdef VX_CFG_VM_ENABLE
    TLBEntry(uint64_t vpn, uint32_t pfn, uint32_t flags, uint64_t size_bits)
      : vpn(vpn)
      , pfn(pfn)
      , flags(flags)
      , mru_bit(true)
      , size_bits (size_bits)
      , valid(true)
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

    uint64_t vpn;
    uint32_t pfn;
    uint32_t flags;
    bool mru_bit;
    uint64_t size_bits;
    bool valid;
    bool d, a, g, u, x, w, r, v;
  #else
    TLBEntry(uint64_t vpn, uint32_t pfn, uint32_t flags)
      : vpn(vpn)
      , pfn(pfn)
      , flags(flags)
      , valid(true)
    {}
    uint64_t vpn;
    uint32_t pfn;
    uint32_t flags;
    bool valid;
  #endif
  };

#ifdef VX_CFG_VM_ENABLE
  std::pair<bool, uint64_t> tlbLookup(uint64_t vAddr, ACCESS_TYPE type, uint64_t* size_bits);

  bool need_trans(uint64_t dev_pAddr);

  uint64_t get_pte_address(uint64_t base_ppn, uint64_t vpn);
  std::pair<uint64_t, uint8_t> page_table_walk(uint64_t vAddr_bits, ACCESS_TYPE type, uint64_t* size_bits);
#else
  uint64_t toPhyAddr(uint64_t vAddr, uint32_t flagMask);
  TLBEntry tlbLookup(uint64_t vAddr, uint32_t flagMask);
#endif



  // Flat array replacing the unordered_map to model a hardware TLB. Linear
  // scan over a small fixed-size buffer is cheaper than hash lookup at this
  // size, and avoids per-insert allocation churn.
  std::vector<TLBEntry> tlb_;
  uint64_t  pageSize_;
  ADecoder  decoder_;
#ifndef VX_CFG_VM_ENABLE
  bool      enableVM_;
#endif

  amo_reservation_t amo_reservation_;
#ifdef VX_CFG_VM_ENABLE
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
  void copy (uint64_t dest_addr, uint64_t src_addr, uint64_t size);

  void loadBinImage(const char* filename, uint64_t destination);
  void loadHexImage(const char* filename);
  void loadVxImage(const char* filename);

  uint8_t& operator[](uint64_t address) {
    return *this->get(address, true); // mutable access forces allocation
  }

  const uint8_t& operator[](uint64_t address) const {
    return *this->get(address, false); // returns shared zero_page_ if unallocated
  }

  void set_acl(uint64_t addr, uint64_t size, int flags);

  void enable_acl(bool enable) {
    check_acl_ = enable;
  }

private:

  // `allocate=false` returns a pointer into `zero_page_` for unmapped
  // addresses without growing the page set — read paths use this to avoid
  // allocator churn on never-written memory. `allocate=true` mints a real
  // page on demand for write paths.
  uint8_t *get(uint64_t address, bool allocate) const;

  uint64_t capacity_;
  uint32_t page_bits_;

  // 2-level sparse chunk directory: chunk_index → CHUNK_SIZE-entry array of
  // page pointers. Adjacent pages share a contiguous chunk array for better
  // locality than a flat page-indexed map; `last_chunk_` caches the most
  // recently touched chunk for sequential access patterns.
  static constexpr int CHUNK_BITS = 10;
  static constexpr int CHUNK_SIZE = 1 << CHUNK_BITS;
  mutable std::unordered_map<uint64_t, uint8_t**> chunks_;
  mutable uint8_t** last_chunk_;
  mutable uint64_t last_chunk_index_;

  // Shared "0xbaadf00d" sentinel page returned for read-without-allocate.
  // Mutable access through operator[] or `get(addr, true)` will mint a real
  // page instead.
  uint8_t* zero_page_;

  ACLManager acl_mngr_;
  bool check_acl_;
};

// PTE_t / vAddr_t now provided by sw/common/vm_types.h (see include at top).

} // namespace vortex
