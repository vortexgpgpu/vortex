/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __MEM_H
#define __MEM_H

#include <ostream>
#include <vector>
#include <queue>
#include <map>
#include <pthread.h>

#include "types.h"

namespace Harp {
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
  class ConsoleMemDevice : public MemDevice {
  public:
    ConsoleMemDevice(Size wS, std::ostream &o, Core &core);
    ~ConsoleMemDevice() {} 

    //virtual Size wordSize() const { return wordSize; }
    virtual Size size() const { return wordSize; }
    virtual Word read(Addr) { pthread_mutex_lock(&cBufLock);
                              char c = cBuf.front();
                              cBuf.pop();
                              pthread_mutex_unlock(&cBufLock);
                              return Word(c); }
    virtual void write(Addr a, Word w) { output << char(w); }

    void poll();

    friend void *Harp::consoleInputThread(void *);

  private:
    std::ostream &output;
    Size wordSize;
    Core &core;

    std::queue<char> cBuf;
    pthread_mutex_t cBufLock;
  };

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
    std::vector <Disk> disks;
    Core &core;
    Size wordSize, blockSize;;
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
    void write(Addr, Word, bool sup);
    void tlbAdd(Addr virt, Addr phys, Word flags);
    void tlbRm(Addr va);
    void tlbFlush() { tlb.clear(); }

#ifdef EMU_INSTRUMENTATION
    Addr virtToPhys(Addr va);
#endif

  private:
    class ADecoder {
    public:
      ADecoder() : zeroChild(NULL), oneChild(NULL), range(0) {}
      ADecoder(MemDevice &md, Size range) : 
        zeroChild(NULL), oneChild(NULL), range(range), md(&md) {}
      Byte *getPtr(Addr a, Size sz, Size wordSize);
      Word read(Addr a, bool sup, Size wordSize);
      void write(Addr a, Word w, bool sup, Size wordSize);
      void map(Addr a, MemDevice &md, Size range, Size bit);
    private:
      MemDevice &doLookup(Addr a, Size &bit);
      ADecoder *zeroChild, *oneChild;
      MemDevice *md;
      Size range;
    };

    ADecoder ad;

    struct TLBEntry {
      TLBEntry() {}
      TLBEntry(Word pfn, Word flags): pfn(pfn), flags(flags) {}
      Word flags;
      Word pfn;
    };

    std::map<Addr, TLBEntry> tlb;
    TLBEntry tlbLookup(Addr vAddr, Word flagMask);

    Size pageSize, addrBytes;

    bool disableVm;
  };
};

#endif
