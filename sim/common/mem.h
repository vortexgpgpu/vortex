#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <stdexcept>

namespace vortex {

enum VA_MODE
{
  BARE,
  SV32
};

enum ACCESS_TYPE
{
  LOAD,
  STORE,
  FETCH
};

class Page_Fault_Exception : public std::runtime_error /* or logic_error */
{
public:
    Page_Fault_Exception(const std::string& what = "") : std::runtime_error(what) {}
    uint64_t addr;
    ACCESS_TYPE type;
};

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

///////////////////////////////////////////////////////////////////////////////


class MemoryUnit {
public:
  
  // struct PageFault {
  //   PageFault(uint64_t a, bool nf)
  //     : faultAddr(a)
  //     , notFound(nf) 
  //   {}
  //   uint64_t faultAddr;
  //   bool notFound;
  // };

  MemoryUnit(uint64_t pageSize, uint64_t addrBytes);

  void attach(MemDevice &m, uint64_t start, uint64_t end);

  void read(void *data, uint64_t addr, uint64_t size, ACCESS_TYPE type);  
  void write(const void *data, uint64_t addr, uint64_t size, ACCESS_TYPE type);

  void tlbAdd(uint64_t virt, uint64_t phys, uint32_t flags, uint32_t size_bits);
  void tlbRm(uint64_t va);
  void tlbFlush() {
    tlb_.clear();
  }

  uint32_t get_satp();  
  void set_satp(uint32_t satp);
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
    TLBEntry(uint32_t pfn, uint8_t flags, uint32_t size_bits)
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
    uint8_t flags;
    bool d, a, g, u, x, w, r, v;
    bool mru_bit;
    uint32_t size_bits;


  };

  std::pair<bool, uint64_t> tlbLookup(uint64_t vAddr, ACCESS_TYPE type, uint32_t* size_bits);
  uint64_t vAddr_to_pAddr(uint64_t vAddr, ACCESS_TYPE type);
  std::pair<uint64_t, uint8_t> page_table_walk(uint64_t vAddr_bits, ACCESS_TYPE type, uint32_t* size_bits);

  std::unordered_map<uint64_t, TLBEntry> tlb_;

  uint64_t pageSize_;
  uint64_t addrBytes_;
  ADecoder decoder_;  
  
  uint32_t satp;
  VA_MODE mode;
  uint32_t ptbr;

  std::unordered_set<uint64_t> unique_translations;
  uint64_t TLB_HIT, TLB_MISS, TLB_EVICT, PTW, PERF_UNIQUE_PTW;


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

private:

  uint8_t *get(uint64_t address) const;

  uint64_t size_;
  uint32_t page_bits_;  
  mutable std::unordered_map<uint64_t, uint8_t*> pages_;
  mutable uint8_t* last_page_;
  mutable uint64_t last_page_index_;
};

class PTE_SV32_t 
{

  private:
    uint64_t address;
    uint64_t bits(uint64_t addr, uint8_t s_idx, uint8_t e_idx)
    {
        return (addr >> s_idx) & ((1 << (e_idx - s_idx + 1)) - 1);
    }
    bool bit(uint8_t idx)
    {
        return (address) & (1 << idx);
    }

  public:
    uint64_t ppn[2];
    uint32_t rsw;
    uint32_t flags;
    bool d, a, g, u, x, w, r, v;
    PTE_SV32_t(uint64_t address) : address(address)
    { 
      flags =  bits(address,0,7);
      rsw = bits(address,8,9);
      ppn[0] = bits(address,10,19);
      ppn[1] = bits(address,20,31);

      d = bit(7);
      a = bit(6);
      g = bit(5);
      u = bit(4);
      x = bit(3);
      w = bit(2);
      r = bit(1);
      v = bit(0);
    }
};

class vAddr_SV32_t 
{

  private:
    uint64_t address;
    uint64_t bits(uint64_t addr, uint8_t s_idx, uint8_t e_idx)
    {
        return (addr >> s_idx) & ((1 << (e_idx - s_idx + 1)) - 1);
    }
    bool bit(uint64_t addr, uint8_t idx)
    {
        return (addr) & (1 << idx);
    }

  public:
    uint64_t vpn[2];
    uint64_t pgoff;
    vAddr_SV32_t(uint64_t address) : address(address)
    {
      vpn[0] = bits(address,12,21);
      vpn[1] = bits(address,22,31);
      pgoff = bits(address,0,11);
    }
};

} // namespace vortex