#pragma once

#include <ostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include "types.h"

namespace vortex {
struct BadAddress {};

class MemDevice {
public:
  virtual ~MemDevice() {}
  virtual Size size() const = 0;
  virtual void read(Addr addr, void *data, Size size) = 0;
  virtual void write(Addr addr, const void *data, Size size) = 0;
};

///////////////////////////////////////////////////////////////////////////////

class RamMemDevice : public MemDevice {
public:
  RamMemDevice(Size size, Size wordSize);
  RamMemDevice(const char *filename, Size wordSize);
  ~RamMemDevice() {}

  void read(Addr addr, void *data, Size size) override;  
  void write(Addr addr, const void *data, Size size) override;

  virtual Size size() const {
    return contents_.size();
  };

protected:
  std::vector<Byte> contents_;
  Size wordSize_;
};

///////////////////////////////////////////////////////////////////////////////

class RomMemDevice : public RamMemDevice {
public:
  RomMemDevice(const char *filename, Size wordSize)
    : RamMemDevice(filename, wordSize) 
  {}

  RomMemDevice(Size size, Size wordSize)
    : RamMemDevice(size, wordSize) 
  {}
  
  ~RomMemDevice();

  void write(Addr addr, const void *data, Size size) override;
};

///////////////////////////////////////////////////////////////////////////////

class MemoryUnit {
public:
  
  struct PageFault {
    PageFault(Addr a, bool nf)
      : faultAddr(a)
      , notFound(nf) 
    {}
    Addr faultAddr;
    bool notFound;
  };

  MemoryUnit(Size pageSize, Size addrBytes, bool disableVm = false);

  void attach(MemDevice &m, Addr start, Addr end);

  void read(Addr addr, void *data, Size size, bool sup);  
  void write(Addr addr, const void *data, Size size, bool sup);

  void tlbAdd(Addr virt, Addr phys, Word flags);
  void tlbRm(Addr va);
  void tlbFlush() {
    tlb_.clear();
  }
private:

  class ADecoder {
  public:
    ADecoder() {}
    
    void read(Addr addr, void *data, Size size);
    void write(Addr addr, const void *data, Size size);
    
    void map(Addr start, Addr end, MemDevice &md);

  private:

    struct mem_accessor_t {
      MemDevice* md;
      Addr addr;
    };
    
    struct entry_t {
      MemDevice *md;
      Addr      start;
      Addr      end;        
    };

    bool lookup(Addr a, Size wordSize, mem_accessor_t*);

    std::vector<entry_t> entries_;
  };

  struct TLBEntry {
    TLBEntry() {}
    TLBEntry(Word pfn, Word flags)
      : pfn(pfn)
      , flags(flags) 
    {}
    Word pfn;
    Word flags;
  };

  TLBEntry tlbLookup(Addr vAddr, Word flagMask);

  std::unordered_map<Addr, TLBEntry> tlb_;
  Size pageSize_;
  Size addrBytes_;
  ADecoder decoder_;  
  bool disableVm_;
};

///////////////////////////////////////////////////////////////////////////////

class RAM : public MemDevice {
public:
  
  RAM(uint32_t num_pages, uint32_t page_size);

  ~RAM();

  void clear();

  Size size() const override;
  void read(Addr addr, void *data, Size size) override;  
  void write(Addr addr, const void *data, Size size) override;

  void loadHexImage(std::string path);

private:

  uint8_t *get(uint32_t address);

  std::vector<uint8_t*> mem_;
  uint32_t page_bits_;
  uint32_t size_;
};

} // namespace vortex