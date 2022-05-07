#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "processorRegs.h"

namespace vortex {
struct BadAddress {};

class MemDevice {
public:
  virtual ~MemDevice() {}
  virtual uint64_t size() const = 0;
  virtual void read(void *data, uint64_t addr, uint64_t size) = 0;
  virtual void write(const void *data, uint64_t addr, uint64_t size) = 0;
};

///////////////////////////////////////////////////////////////////////////////

class RamMemDevice : public MemDevice {
public:
  RamMemDevice(uint64_t size, uint32_t wordSize);
  RamMemDevice(const char *filename, uint32_t wordSize);
  ~RamMemDevice() {}

  void read(void *data, uint64_t addr, uint64_t size) override;  
  void write(const void *data, uint64_t addr, uint64_t size) override;

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

  void write(const void *data, uint64_t addr, uint64_t size) override;
};


class TablePageEntryBase{
    public:
    
    TablePageEntryBase(uint64_t address)
    {
          bytes_ = address;
    }

    bool isValid(){
          return getBit(VIndex);
      }

      bool isExecutable(){
        return getBit(XIndex);
      }

      bool isReadable(){
        return getBit(RIndex);
      }

      bool isWritable(){
        return getBit(WIndex);
      }

      uint64_t getNextTableAddress()
      {
          return (bytes_ >> (DIndex+1)) & (((uint64_t)1 << (PageNumberLength+(uint64_t)1)) - (uint64_t)1); 
      }

      protected:

      static const uint64_t PageNumberLength = 44;
      static const int DIndex = 7;
      static const int AIndex = 6;
      static const int GIndex = 5;
      static const int UIndex = 4;
      static const int XIndex = 3;
      static const int WIndex = 2;
      static const int RIndex = 1;
      static const int VIndex = 0; 

      private:

      bool getBit(int bitNumber){
          return (bytes_ >> bitNumber) & 1;
      }

      uint64_t bytes_;

};

class MultibaleAddressBase{
  public:
  MultibaleAddressBase(uint64_t addr){
    address_ = addr;
  }

  virtual int levelCount() = 0;
  virtual int pageOffsetLength() = 0;
  virtual int offsetInPageLength()=0;
  
  uint64_t getOffsetForLevel(int level)
  {
        uint64_t one64 = 1;
        uint64_t index = level * offsetInPageLength() + pageOffsetLength();
        uint64_t offset = (address_ >>index) & ((one64 << offsetInPageLength()) - one64);
        return offset;
  }

    uint64_t getOffset(){
        uint64_t one64 = 1;
        uint64_t offset = address_ & ((one64 << pageOffsetLength()) - one64);
        return offset;
    }

    uint64_t getVirtualAddress(){
      return address_ >> pageOffsetLength(); 
    }

    protected:

    
    private:
    uint64_t address_;
};


class VirtualAddressFactoryBase{
  public:
  virtual MultibaleAddressBase* createMultitableAddressFromBits(uint64_t addr) = 0;
};

class PhysicalTableEntryFactoryBase{
  public:
  virtual TablePageEntryBase* createPageTableEntryFromBits(uint64_t addr) = 0;
};

///////////////////////////////////////////////////////////////////////////////

class RAM : public MemDevice {
public:
  
  RAM(uint32_t page_size);
  ~RAM();

  void clear();

  uint64_t size() const override;

  void read(void *data, uint64_t addr, uint64_t size) override;  
  void write(const void *data, uint64_t addr, uint64_t size) override;

  void loadBinImage(const char* filename, uint64_t destination);
  void loadHexImage(const char* filename);

  uint8_t& operator[](uint64_t address) {
    return *this->get(address);
  }

  const uint8_t& operator[](uint64_t address) const {
    return *this->get(address);
  }

  uint64_t getPageRootTable();
  uint64_t getFirstFreeTable();

private:

  uint8_t *get(uint64_t address) const;
  uint64_t initializeRootTable();

  uint64_t size_;
  uint32_t page_bits_;  
  mutable std::unordered_map<uint64_t, uint8_t*> pages_;
  mutable uint8_t* last_page_;
  mutable uint64_t last_page_index_;
  uint64_t rootPageTableNumber_ = -1;
  bool isPageRootTableInitialized_ = false;
};


class VirtualDevice: public RAM{
  public:
  VirtualDevice(uint32_t page_size): RAM(page_size){

  }
};
///////////////////////////////////////////////////////////////////////////////


class MemoryUnit {
public:
  
  struct PageFault {
    PageFault(uint64_t a, bool nf)
      : faultAddr(a)
      , notFound(nf) 
    {}
    uint64_t faultAddr;
    bool notFound;
  };

  struct TlbMiss{
    TlbMiss(uint64_t vaddr)
      : faultAddr(vaddr)
      {}
    uint64_t faultAddr;
  };

  MemoryUnit(uint64_t pageSize, uint64_t addrBytes, bool disableVm = false);
  MemoryUnit(uint64_t pageSize, uint64_t addrBytes, uint64_t rootAddress);
  void attach(MemDevice &m, uint64_t start, uint64_t end);
  void read(void *data, uint64_t addr, uint64_t size, bool sup);  
  void write(const void *data, uint64_t addr, uint64_t size, bool sup);
  void attachRAM(RAM &ram, uint64_t start, uint64_t end);
  void attachVirtualDevice(VirtualDevice &virtualDevice);
  void attachSuperVisorregisters(SupervisorRegisterContainer &supervysorRegisters);
  void requestVirtualPage(uint8_t* data,uint64_t virtualAddr);

  void tlbAdd(uint64_t virt, uint64_t phys, uint32_t flags);
  void tlbRm();
  void tlbFlush() {
    tlb_.clear();
  }
private:

  class ADecoder {
  public:
    ADecoder() {}
    
    void read(void *data, uint64_t addr, uint64_t size);
    void write(const void *data, uint64_t addr, uint64_t size);
    
    void map(uint64_t start, uint64_t end, MemDevice &md);

  private:

    struct mem_accessor_t {
      MemDevice* md;
      uint64_t addr;
    };
    
    struct entry_t {
      MemDevice *md;
      uint64_t      start;
      uint64_t      end;        
    };

    bool lookup(uint64_t a, uint32_t wordSize, mem_accessor_t*);

    std::vector<entry_t> entries_;
  };

  struct TLBEntry {
    TLBEntry() {}
    TLBEntry(uint64_t pfn, uint32_t flags)
      : pfn(pfn)
      , flags(flags) 
    {}

    void updateAccessBit(bool isSet){
      isAccessBitSet = isSet;
    }

    uint64_t pfn;
    uint32_t flags;
    bool isAccessBitSet;
  };

  uint64_t translateVirtualToPhysical(uint64_t vAddr, uint32_t flagMask);
  TLBEntry tlbLookup(uint64_t vAddr, uint32_t flagMask);

  void updateTLBIfNeeded();
  void markTableAccessed(uint64_t vAddr);
  void markTableExecutable(uint64_t vAddr);
  uint64_t handlePageFault();
  uint64_t allocate_translation_table();
  void setupAddressFactories();
  uint64_t handleTlbMiss(uint64_t vaddr);

  void* requestPage(uint64_t address){
    void*data;
    ram_->read(data, address, pageSize_);
  }

  std::unordered_map<uint64_t, TLBEntry> tlb_;
  uint64_t pageSize_;
  uint64_t addrBytes_;
  ADecoder decoder_;  
  bool disableVM_;
  uint64_t rootPageAddress_; 
  uint64_t pageLevels_ = 3;
  uint64_t levelPagesInMemory_ = 2;
  RAM* ram_;
  int memoryAccessCount_ = 0;
  VirtualDevice* virtualDevice_;
  SupervisorRegisterContainer* supervisorContainer_;
  //TODO KA: use clever pointers instead
  PhysicalTableEntryFactoryBase* pteFactory_;
  VirtualAddressFactoryBase*  virtualAddressFactory_;


  static const int maxPageTableEntriesCount = 512;
  // Simple implementation of random LRU algorithm.
  // A bit is reset for all entries every "RefreshRateUpdateToMemory"
  static const int RefreshTblRate = 20;
};

class SV39TablePageEntry: public TablePageEntryBase
{
  public:
  SV39TablePageEntry(uint64_t pte): TablePageEntryBase(pte){};
};

class SV39VirtualAddress: public MultibaleAddressBase
{
  public:
  SV39VirtualAddress(uint64_t address):MultibaleAddressBase(address){}
  int levelCount() override { return 3;}
  int pageOffsetLength() override { return  12;}
  protected:
  int offsetInPageLength() override { return 9;} 
};

class SV39PageTableEntryFactory:public  PhysicalTableEntryFactoryBase{
  TablePageEntryBase* createPageTableEntryFromBits(uint64_t addr) override{
    return new SV39TablePageEntry(addr);
  }
};

class SV39VirtualAddressFactory: public VirtualAddressFactoryBase{
  public:
  MultibaleAddressBase* createMultitableAddressFromBits(uint64_t addr) override{
    return new SV39VirtualAddress(addr);
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace vortex