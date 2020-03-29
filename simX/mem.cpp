/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <stdlib.h>
// #include <pthread.h>

#include "include/debug.h"
#include "include/types.h"
#include "include/util.h"
#include "include/mem.h"
#include "include/core.h"

using namespace std;
using namespace Harp;

RamMemDevice::RamMemDevice(const char *filename, Size wordSize) :
  wordSize(wordSize), contents()
{
  ifstream input(filename);

  if (!input) {
    cout << "Error reading file \"" << filename << "\" into RamMemDevice.\n";
    std::abort();
  }

  do { contents.push_back(input.get()); } while (input);

  while (contents.size() % wordSize) contents.push_back(0x00);
}

RamMemDevice::RamMemDevice(Size size, Size wordSize) : 
  wordSize(wordSize), contents(size) {}

void RomMemDevice::write(Addr, Word) {
  cout << "Attempt to write to ROM.\n";
  std::abort();
}

Word RamMemDevice::read(Addr addr) {
  D(2, "RAM read, addr=0x" << hex << addr);
  Word w = readWord(contents, addr, wordSize - addr%wordSize);
  return w;
}

void RamMemDevice::write(Addr addr, Word w) {
  D(2, "RAM write, addr=0x" << hex << addr);
  writeWord(contents, addr, wordSize - addr%wordSize, w);
}

MemDevice &MemoryUnit::ADecoder::doLookup(Addr a, Size &bit) {
  if (range == 0 || (a&((1ll<<bit)-1)) >= range) {
    ADecoder *p(((a>>bit)&1)?oneChild:zeroChild);
    if (p) { bit--; return p->doLookup(a, bit); }
    else {cout << "lookup of 0x" << hex << a << " failed.\n";
          throw BadAddress();}
  } else {
    return *md;
  }
}

void MemoryUnit::ADecoder::map(Addr a, MemDevice &m, Size r, Size bit)
{
  if ((1llu << bit) <= r) {
    md = &m;
    range = m.size();
  } else {
    ADecoder *&child(((a>>bit)&1)?oneChild:zeroChild);
    if (!child) child = new ADecoder();
    child->map(a, m, r, bit-1);
  }
}

Byte *MemoryUnit::ADecoder::getPtr(Addr a, Size sz, Size wordSize) {
  Size bit = wordSize - 1;
  MemDevice &m(doLookup(a, bit));
  a &= (2<<bit)-1;
  if (a + sz <= m.size()) return m.base() + a;
  return NULL;
}

Word MemoryUnit::ADecoder::read(Addr a, bool sup, Size wordSize) {
  Size bit = wordSize - 1;
  MemDevice &m(doLookup(a, bit));
  a &= (2<<bit)-1;
  // std::cout << std::hex << "ADecoder::read(Addr " << a << ", sup " << sup << ", wordSize " << wordSize << " -> ";
  // std::cout << "Data: " << m.read(a) << "\n";
  return m.read(a);
}

void MemoryUnit::ADecoder::write(Addr a, Word w, bool sup, Size wordSize) {
  Size bit = wordSize - 1;
  MemDevice &m(doLookup(a, bit));

  RAM & r = (RAM &) m;
  // a &= (2<<bit)-1;
  // std::cout << std::hex << "ADecoder::write(Addr " << a << ", w " << w << ", sup " << sup << ", wordSize " << wordSize << "\n";
  Word before = m.read(a);
  Word new_word = w;

  // if (a == 0x00010000)
  // {
  //   fprintf(stderr, "%c", w);
  // }

  if (wordSize == 8)
  {
    r.writeByte(a, &w);
  }
  else if (wordSize == 16)
  {
    r.writeHalf(a, &w);
  }
  else
  {
    r.writeWord(a, &w);
  }
  // m.write(a, new_word);
}

Byte *MemoryUnit::getPtr(Addr a, Size s) {
  return ad.getPtr(a, s, addrBytes*8);
}

void MemoryUnit::attach(MemDevice &m, Addr base) {
  ad.map(base, m, m.size(), addrBytes*8 - 1);
}

MemoryUnit::TLBEntry MemoryUnit::tlbLookup(Addr vAddr, Word flagMask) {
  map<Addr, MemoryUnit::TLBEntry>::iterator i;
  if ((i = tlb.find(vAddr/pageSize)) != tlb.end()) {
    TLBEntry &t = i->second;
    if (t.flags & flagMask) return t;
    else {
      D(2, "Page fault on addr 0x" << hex << vAddr << "(bad flags)");
      throw PageFault(vAddr, false);
    }
  } else {
    D(2, "Page fault on addr 0x" << hex << vAddr << "(not in TLB)");
    throw PageFault(vAddr, true);
  }
}

#ifdef EMU_INSTRUMENTATION
Addr MemoryUnit::virtToPhys(Addr vAddr) {
  TLBEntry t = tlbLookup(vAddr, 077);
  return t.pfn*pageSize + vAddr%pageSize;
}
#endif

Word MemoryUnit::read(Addr vAddr, bool sup) {
  Addr pAddr;
  if (disableVm) {
    pAddr = vAddr;
  } else {
    Word flagMask = sup?8:1;
    TLBEntry t = tlbLookup(vAddr, flagMask);
    pAddr = t.pfn*pageSize + vAddr%pageSize;
  }
  // std::cout << "MU::write: About to read: " << std::hex << pAddr << " = " << (ad.read(pAddr, sup, 8*addrBytes)) << " with " << std::dec << (8*addrBytes) << "\n";
  return ad.read(pAddr, sup, 8*addrBytes);
}

Word MemoryUnit::fetch(Addr vAddr, bool sup) {
  Addr pAddr;

  if (disableVm) {
    pAddr = vAddr;
  } else {
    Word flagMask = sup?32:4;
    TLBEntry t = tlbLookup(vAddr, flagMask);
    pAddr = t.pfn*pageSize + vAddr%pageSize;
  }

  Word instruction = ad.read(pAddr, sup, 8*addrBytes);

  return instruction;
}

void MemoryUnit::write(Addr vAddr, Word w, bool sup, Size bytes) {
  Addr pAddr;

  if (disableVm) {
    pAddr = vAddr;
  } else {
    Word flagMask = sup?16:2;
    TLBEntry t = tlbLookup(vAddr, flagMask);
    pAddr = t.pfn*pageSize + vAddr%pageSize;
  }
  // std::cout << "MU::write: About to write: " << std::hex << pAddr << " = " << w << " with " << std::dec << 8*bytes << "\n";
  ad.write(pAddr, w, sup, 8*bytes);
  // std::cout << std::hex << "reading same address: " << (this->read(vAddr, sup)) << "\n";
}

void MemoryUnit::tlbAdd(Addr virt, Addr phys, Word flags) {
  D(1, "tlbAdd(0x" << hex << virt << ", 0x" << phys << ", 0x" << flags << ')');
  tlb[virt/pageSize] = TLBEntry(phys/pageSize, flags);
}

void MemoryUnit::tlbRm(Addr va) {
  if (tlb.find(va/pageSize) != tlb.end()) tlb.erase(tlb.find(va/pageSize));
}

void *Harp::consoleInputThread(void* arg_vp) {
  // ConsoleMemDevice *arg = (ConsoleMemDevice *)arg_vp;
  // char c;
  // while (cin) {
  //   c = cin.get();
  //   pthread_mutex_lock(&arg->cBufLock);
  //   arg->cBuf.push(c);
  //   pthread_mutex_unlock(&arg->cBufLock);
  // }
  // cout << "Console input ended. Exiting.\n";
  // exit(4);
  return nullptr;
}

// ConsoleMemDevice::ConsoleMemDevice(Size wS, std::ostream &o, Core &core,
//                                    bool batch) : 
//   wordSize(wS), output(o), core(core), cBuf()
// {
//   // Create a console input thread if we are running in interactive mode.
//   if (!batch) {
//     pthread_t *thread = new pthread_t;
//     pthread_create(thread, NULL, consoleInputThread, (void*)this);
//   }
//   pthread_mutex_init(&cBufLock, NULL);
// }

// void ConsoleMemDevice::poll() {
//   pthread_mutex_lock(&cBufLock);
//   if (!cBuf.empty()) core.interrupt(8);
//   pthread_mutex_unlock(&cBufLock);
// }

Word DiskControllerMemDevice::read(Addr a) {
  switch (a/8) {
    case 0: return curDisk;
    case 1: return curBlock;
    case 2: return disks[curDisk].blocks * blockSize;
    case 3: return physAddr;
    case 4: return command;
    case 5: return status;
    default:
      cout << "Attempt to read invalid disk controller register.\n";
      std::abort();
  }
}

void DiskControllerMemDevice::write(Addr a, Word w) {
  switch (a/8) {
    case 0: if (w <= disks.size()) {
              curDisk = w;
              status = OK;
            } else {
              status = INVALID_DISK;
            }
            break;
    case 1: if (w < disks[curDisk].blocks) {
              curBlock = w;
            } else {
              status = INVALID_BLOCK;
            }
            break;
    case 2: nBlocks = w >= disks[curDisk].blocks?disks[curDisk].blocks - 1 : w;
            status = OK;
            break;
    case 3: physAddr = w;
            status = OK;
            break;
    case 4: if (w == 0) {
            } else {
            }
            cout << "TODO: Implement disk read and write!\n";
            break;
  }
}

static uint32_t hti_old(char c) {
      if (c >= 'A' && c <= 'F')
          return c - 'A' + 10;
      if (c >= 'a' && c <= 'f')
          return c - 'a' + 10;
      return c - '0';
  }

static uint32_t hToI_old(char *c, uint32_t size) {
    uint32_t value = 0;
    for (uint32_t i = 0; i < size; i++) {
        value += hti_old(c[i]) << ((size - i - 1) * 4);
    }
    return value;
}

void RAM::loadHexImpl(std::string path) {
      this->clear();
      FILE *fp = fopen(&path[0], "r");
      if(fp == 0){
          std::cout << path << " not found" << std::endl;
      }

      //Preload 0x0 <-> 0x80000000 jumps      
      ((uint32_t*)this->get(0))[0] = 0xf1401073;
      ((uint32_t*)this->get(0))[1] = 0xf1401073;      
      ((uint32_t*)this->get(0))[2] = 0x30101073;
      ((uint32_t*)this->get(0))[3] = 0x800000b7;
      ((uint32_t*)this->get(0))[4] = 0x000080e7;
      
      ((uint32_t*)this->get(0x80000000))[0] = 0x00000097;

      ((uint32_t*)this->get(0xb0000000))[0] = 0x01C02023;
      
      ((uint32_t*)this->get(0xf00fff10))[0] = 0x12345678;

      ((uint32_t*)this->get(0x70000000))[0] = 0x00008067;

      {
        uint32_t init_addr = 0x70000004;
        for (int off = 0; off < 1024; off+=4) {
          uint32_t new_addr = init_addr+off;
          ((uint32_t*)this->get(new_addr))[0] = 0x00000000;
        }
      }

      {
        uint32_t init_addr = 0x71000000;
        for (int off = 0; off < 1024; off+=4) {
          uint32_t new_addr = init_addr+off;
          ((uint32_t*)this->get(new_addr))[0] = 0x00000000;
        }
      }

      {
        uint32_t init_addr = 0x72000000;
        for (int off = 0; off < 1024; off+=4) {
          uint32_t new_addr = init_addr+off;
          ((uint32_t*)this->get(new_addr))[0] = 0x00000000;
        }
      }

      fseek(fp, 0, SEEK_END);
      uint32_t size = ftell(fp);
      fseek(fp, 0, SEEK_SET);
      char* content = new char[size];
      int x = fread(content, 1, size, fp);

      if (!x) { 
        std::cout << "COULD NOT READ FILE\n"; std::abort();
      }

      int offset = 0;
      char* line = content;
      // std::cout << "WHTA\n";
      while (1) {
          if (line[0] == ':') {
              uint32_t byteCount = hToI_old(line + 1, 2);
              uint32_t nextAddr = hToI_old(line + 3, 4) + offset;
              uint32_t key = hToI_old(line + 7, 2);
              switch (key) {
              case 0:
                  for (uint32_t i = 0; i < byteCount; i++) {
                      unsigned add = nextAddr + i;
                      *(this->get(add)) = hToI_old(line + 9 + i * 2, 2);
                      // std::cout << "lhi: Address: " << std::hex <<(add) << "\tValue: " << std::hex << hToI_old(line + 9 + i * 2, 2) << std::endl;
                  }
                  break;
              case 2:
  //              cout << offset << endl;
                  offset = hToI_old(line + 9, 4) << 4;
                  break;
              case 4:
  //              cout << offset << endl;
                  offset = hToI_old(line + 9, 4) << 16;
                  break;
              default:
  //              cout << "??? " << key << endl;
                  break;
              }
          }

          while (*line != '\n' && size != 0) {
              line++;
              size--;
          }

          if (size <= 1)
              break;

          line++;
          size--;
      }      

      if (content) 
        delete[] content;
  }