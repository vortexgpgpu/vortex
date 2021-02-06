#pragma once

#include <ostream>
#include <vector>
#include <queue>
#include <unordered_map>
// #include <pthread.h>

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
    virtual Byte *base() { return NULL; } /* Null if unavailable. */
  };

  class RamMemDevice : public MemDevice {
  public:
    RamMemDevice(Size size, Size wordSize);
    RamMemDevice(const char* filename, Size wordSize);
    ~RamMemDevice() {}

    virtual Size size() const { return contents.size(); };
    virtual Word read(Addr);
    virtual void write(Addr, Word);
    virtual Byte *base() { return &contents[0]; }

  protected:
    Size wordSize;
    std::vector<Byte> contents;
  };

  class RomMemDevice : public RamMemDevice {
  public:
    RomMemDevice(const char* filename, Size wordSize) :
      RamMemDevice(filename, wordSize) {}
    RomMemDevice(Size size, Size wordSize) :
      RamMemDevice(size, wordSize) {}
    ~RomMemDevice();

    virtual void write(Addr, Word);
  };

  class Core;

  class DiskControllerMemDevice : public MemDevice {
  public:
    DiskControllerMemDevice(Size wordSize, Size blockSize, Core &c) : 
      wordSize(wordSize), blockSize(blockSize), core(c), disks() {}
    
    void addDisk(Byte *file, Size n) { disks.push_back(Disk(file, n)); }

    virtual Size size() const { return wordSize * 6; }
    virtual Word read(Addr);
    virtual void write(Addr, Word);

  private:
    Word curDisk, curBlock, nBlocks, physAddr, command, status;
    enum Status { OK = 0, INVALID_DISK, INVALID_BLOCK };
    struct Disk {
      Disk(Byte *f, Size n): file(f), blocks(n) {}
      Byte *file;
      Size blocks;
    };
    
    Size wordSize, blockSize;
    Core &core;
    std::vector <Disk> disks;   
  };

  class MemoryUnit {
  public:
    MemoryUnit(Size pageSize, Size addrBytes, bool disableVm = false) : 
      pageSize(pageSize), addrBytes(addrBytes), ad(), disableVm(disableVm)
    {
      if (!disableVm)
        tlb[0] = TLBEntry(0, 077);
    }
    void attach(MemDevice &m, Addr base);

    //Size wordSize();
    struct PageFault { 
      PageFault(Addr a, bool nf) : faultAddr(a), notFound(nf) {}
      Addr faultAddr;
      bool notFound;
    }; /* Thrown on page fault. */

    Word read(Addr, bool sup);  /* For data accesses. */
    Word fetch(Addr, bool sup); /* For instruction accesses. */
    Byte *getPtr(Addr, Size);
    void write(Addr, Word, bool sup, Size);
    void tlbAdd(Addr virt, Addr phys, Word flags);
    void tlbRm(Addr va);
    void tlbFlush() { tlb.clear(); }

  private:
    class ADecoder {
    public:
      ADecoder() : zeroChild(NULL), oneChild(NULL), range(0), md(nullptr) {}
      ADecoder(MemDevice &md, Size range) : 
        zeroChild(NULL), oneChild(NULL), range(range), md(&md) {}
      Byte *getPtr(Addr a, Size sz, Size wordSize);
      Word read(Addr a, bool sup, Size wordSize);
      void write(Addr a, Word w, bool sup, Size wordSize);
      void map(Addr a, MemDevice &md, Size range, Size bit);
    private:
      MemDevice &doLookup(Addr a, Size &bit);
      ADecoder *zeroChild, *oneChild;      
      Size range;
      MemDevice *md;      
    };

    struct TLBEntry {
      TLBEntry() {}
      TLBEntry(Word pfn, Word flags): pfn(pfn), flags(flags) {}
      Word pfn;
      Word flags;      
    };

    Size pageSize, addrBytes;
    
    ADecoder ad;

    std::unordered_map<Addr, TLBEntry> tlb;
    TLBEntry tlbLookup(Addr vAddr, Word flagMask);    

    bool disableVm;
  };

  class RAM : public MemDevice {
  public:
      uint8_t* mem[1 << 12];

      RAM(){
          for(uint32_t i = 0;i < (1 << 12);i++) 
            mem[i] = NULL;
      }
      ~RAM(){
          for(uint32_t i = 0;i < (1 << 12);i++) 
            if(mem[i]) 
              delete [] mem[i];
      }

      void clear(){
          for(uint32_t i = 0;i < (1 << 12);i++)
          {
              if(mem[i])
              { 
                  delete mem[i];
                  mem[i] = NULL;
              }
          }
      }

      uint8_t* get(uint32_t address){

          if(mem[address >> 20] == NULL) {
              uint8_t* ptr = new uint8_t[1024*1024];
              for(uint32_t i = 0;i < 1024*1024;i+=4) {
                  ptr[i + 0] = 0xaa;
                  ptr[i + 1] = 0xbb;
                  ptr[i + 2] = 0xcc;
                  ptr[i + 3] = 0xdd;
              }
              mem[address >> 20] = ptr;
          }
          return &mem[address >> 20][address & 0xFFFFF];
      }

      void read(uint32_t address,uint32_t length, uint8_t *data){
          for(unsigned i = 0;i < length;i++){
              data[i] = (*this)[address + i];
          }
      }

      void write(uint32_t address,uint32_t length, uint8_t *data){
          for(unsigned i = 0;i < length;i++){
              (*this)[address + i] = data[i];
          }
      }

      virtual Size size() const { return -1; }

      void getBlock(uint32_t address, uint8_t *data)
      {
          uint32_t block_number = address & 0xffffff00; // To zero out block offset
          uint32_t bytes_num    = 256;

          this->read(block_number, bytes_num, data);
      }

      void getWord(uint32_t address, uint32_t * data)
      {
          data[0] = 0;

          uint8_t first  = *get(address + 0);
          uint8_t second = *get(address + 1);
          uint8_t third  = *get(address + 2);
          uint8_t fourth = *get(address + 3);


          // std::cout << std::hex;
          // std::cout << "RAM: READING ADDRESS " << address + 0 << " DATA: " << (uint32_t) first << "\n";
          // std::cout << "RAM: READING ADDRESS " << address + 1 << " DATA: " << (uint32_t) second << "\n";
          // std::cout << "RAM: READING ADDRESS " << address + 2 << " DATA: " << (uint32_t) third << "\n";
          // std::cout << "RAM: READING ADDRESS " << address + 3 << " DATA: " << (uint32_t) fourth << "\n";

          data[0] = (data[0] << 0) | fourth;
          data[0] = (data[0] << 8) | third;
          data[0] = (data[0] << 8) | second;
          data[0] = (data[0] << 8) | first;
          // data[0] = (data[0] << 0) | first;
          // data[0] = (data[0] << 8) | second;
          // data[0] = (data[0] << 8) | third;
          // data[0] = (data[0] << 8) | fourth;

          // std::cout << "FINAL DATA: " << data[0] << "\n";

      }

      void writeWord(uint32_t address, uint32_t * data)
      {
          uint32_t data_to_write = *data;

          uint32_t byte_mask = 0xFF;

          for (int i = 0; i < 4; i++)
          {
              // std::cout << "RAM: DATA TO WRITE " << data_to_write << "\n";
              // std::cout << "RAM: DATA TO MASK  " << byte_mask << "\n";
              // std::cout << "RAM: WRITING ADDRESS " << address + i << " DATA: " << (data_to_write & byte_mask) << "\n";
              (*this)[address + i] = data_to_write & byte_mask;
              data_to_write        = data_to_write >> 8;
          }
      }

      void writeHalf(uint32_t address, uint32_t * data)
      {
          uint32_t data_to_write = *data;

          uint32_t byte_mask = 0xFF;

          for (int i = 0; i < 2; i++)
          {
              // std::cout << "RAM: DATA TO WRITE " << data_to_write << "\n";
              // std::cout << "RAM: DATA TO MASK  " << byte_mask << "\n";
              // std::cout << "RAM: WRITING ADDRESS " << address + i << " DATA: " << (data_to_write & byte_mask) << "\n";
              (*this)[address + i] = data_to_write & byte_mask;
              data_to_write        = data_to_write >> 8;
          }
      }

      void writeByte(uint32_t address, uint32_t * data)
      {
          uint32_t data_to_write = *data;

          uint32_t byte_mask = 0xFF;

          (*this)[address] = data_to_write & byte_mask;
          data_to_write    = data_to_write >> 8;

      }

      uint8_t& operator [](uint32_t address) {
          return *get(address);
      }

      virtual void write(Addr addr, Word w)
      {
          uint32_t word = (uint32_t) w;
          writeWord(addr, &word);
      }

      virtual Word read(Addr addr)
      {
          uint32_t w;
          getWord(addr, &w);
          // std::cout << "RAM: read -> " << w << " at addr: " << addr << "\n";
          return (Word) w;
      }

      virtual Byte *base()
      {
          return (Byte *) this->get(0);
      }

  // MEMORY UTILS

  void loadHexImpl(std::string path); 

  };
}