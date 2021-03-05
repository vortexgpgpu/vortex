#pragma once

#include <ostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include "types.h"

namespace vortex {
void *consoleInputThread(void *);
struct BadAddress {};

class MemDevice {
public:
  virtual ~MemDevice() {}
  virtual Size size() const = 0;
  virtual Word read(Addr) = 0;
  virtual void write(Addr, Word) = 0;
  virtual Byte *base() {
    return NULL;
  }
};

///////////////////////////////////////////////////////////////////////////////

class RamMemDevice : public MemDevice {
public:
  RamMemDevice(Size size, Size wordSize);
  RamMemDevice(const char *filename, Size wordSize);
  ~RamMemDevice() {}

  virtual Word read(Addr);  
  virtual void write(Addr, Word);

  virtual Size size() const {
    return contents_.size();
  };

  virtual Byte *base() {
    return &contents_[0];
  }

protected:
  Size wordSize_;
  std::vector<Byte> contents_;
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

  virtual void write(Addr, Word);
};

///////////////////////////////////////////////////////////////////////////////

class Core;

class DiskControllerMemDevice : public MemDevice {
public:
  DiskControllerMemDevice(Size wordSize, Size blockSize, Core &c);

  virtual Word read(Addr);
  virtual void write(Addr, Word);

  virtual Size size() const {
    return uint64_t(wordSize_) * 6;
  }

  void addDisk(Byte *file, Size n) {
    disks_.push_back(Disk(file, n));
  }

private:
   
  enum Status { 
    OK = 0,
    INVALID_DISK,
    INVALID_BLOCK 
  };

  struct Disk {
    Disk(Byte *f, Size n)
      : file(f)
      , blocks(n) 
    {}
    Byte *file;
    Size blocks;
  };

  Word curDisk_; 
  Word curBlock_;
  Word nBlocks_;
  Word physAddr_;
  Word command_;
  Word status_;
  Size wordSize_;
  Size blockSize_;
  Core &core_;
  std::vector<Disk> disks_;
};

///////////////////////////////////////////////////////////////////////////////

class MemoryUnit {
public:
  MemoryUnit(Size pageSize, Size addrBytes, bool disableVm = false);

  void attach(MemDevice &m, Addr start, Addr end);

  struct PageFault {
    PageFault(Addr a, bool nf)
      : faultAddr(a)
      , notFound(nf) 
    {}
    Addr faultAddr;
    bool notFound;
  };

  Word read(Addr, bool sup);
  Word fetch(Addr, bool sup);
  void write(Addr, Word, bool sup, Size);
  void tlbAdd(Addr virt, Addr phys, Word flags);
  void tlbRm(Addr va);

  void tlbFlush() {
    tlb_.clear();
  }

private:

  class ADecoder {
  public:
    ADecoder() {}
    
    Word read(Addr a, bool sup, Size wordSize);
    void write(Addr a, Word w, bool sup, Size wordSize);
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

  void write(Addr addr, Word w)  override;

  Word read(Addr addr) override;  

  Byte *base() override;

  void read(uint32_t address, uint32_t length, uint8_t *data);

  void write(uint32_t address, uint32_t length, uint8_t *data);

  void writeWord(uint32_t address, uint32_t *data);

  void writeHalf(uint32_t address, uint32_t *data);

  void writeByte(uint32_t address, uint32_t *data);

  void loadHexImage(std::string path);

private:

  uint8_t *get(uint32_t address);

  void getBlock(uint32_t address, uint8_t *data);

  void getWord(uint32_t address, uint32_t *data);

  std::vector<uint8_t*> mem_;
  uint32_t page_bits_;
  uint32_t size_;
};

} // namespace vortex