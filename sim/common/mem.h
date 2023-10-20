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
#include <unordered_map>
#include <cstdint>

namespace vortex {
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
  
  struct PageFault {
    PageFault(uint64_t a, bool nf)
      : faultAddr(a)
      , notFound(nf) 
    {}
    uint64_t  faultAddr;
    bool      notFound;
  };

  MemoryUnit(uint64_t pageSize = 0);

  void attach(MemDevice &m, uint64_t start, uint64_t end);

  void read(void* data, uint64_t addr, uint64_t size, bool sup);
  void write(const void* data, uint64_t addr, uint64_t size, bool sup);

  void amo_reserve(uint64_t addr);
  bool amo_check(uint64_t addr);

  void tlbAdd(uint64_t virt, uint64_t phys, uint32_t flags);
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
    TLBEntry(uint32_t pfn, uint32_t flags)
      : pfn(pfn)
      , flags(flags) 
    {}
    uint32_t pfn;
    uint32_t flags;
  };

  TLBEntry tlbLookup(uint64_t vAddr, uint32_t flagMask);

  uint64_t toPhyAddr(uint64_t vAddr, uint32_t flagMask);

  std::unordered_map<uint64_t, TLBEntry> tlb_;
  uint64_t  pageSize_;
  ADecoder  decoder_;  
  bool      enableVM_;

  amo_reservation_t amo_reservation_;
};

///////////////////////////////////////////////////////////////////////////////

class RAM : public MemDevice {
public:
  
   RAM(uint32_t page_size, uint64_t capacity = 0);
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

private:

  uint8_t *get(uint64_t address) const;

  uint64_t capacity_;
  uint32_t page_bits_;  
  mutable std::unordered_map<uint64_t, uint8_t*> pages_;
  mutable uint8_t* last_page_;
  mutable uint64_t last_page_index_;
};

} // namespace vortex
