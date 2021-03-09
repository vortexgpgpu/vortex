#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <assert.h>

#include "debug.h"
#include "types.h"
#include "util.h"
#include "mem.h"
#include "core.h"

using namespace vortex;

RamMemDevice::RamMemDevice(const char *filename, Size wordSize) 
  : wordSize_(wordSize) {
  std::ifstream input(filename);

  if (!input) {
    std::cout << "Error reading file \"" << filename << "\" into RamMemDevice.\n";
    std::abort();
  }

  do {
    contents_.push_back(input.get());
  } while (input);

  while (contents_.size() & (wordSize-1))
    contents_.push_back(0x00);
}

RamMemDevice::RamMemDevice(Size size, Size wordSize)
  : contents_(size) 
  , wordSize_(wordSize)
{}

void RamMemDevice::read(Addr addr, void *data, Size size) {
  auto addr_end = addr + size;
  if ((addr & (wordSize_-1))
   || (addr_end & (wordSize_-1)) 
   || (addr_end <= contents_.size())) {
    std::cout << "lookup of 0x" << std::hex << (addr_end-1) << " failed.\n";
    throw BadAddress();
  }  
  
  const Byte *s = contents_.data() + addr;
  for (Byte *d = (Byte*)data, *de = d + size; d != de;) {
    *d++ = *s++;
  }
}

void RamMemDevice::write(Addr addr, const void *data, Size size) {
  auto addr_end = addr + size;
  if ((addr & (wordSize_-1))
   || (addr_end & (wordSize_-1)) 
   || (addr_end <= contents_.size())) {
    std::cout << "lookup of 0x" << std::hex << (addr_end-1) << " failed.\n";
    throw BadAddress();
  }

  const Byte *s = (const Byte*)data;
  for (Byte *d = contents_.data() + addr, *de = d + size; d != de;) {
    *d++ = *s++;
  }
}

///////////////////////////////////////////////////////////////////////////////

void RomMemDevice::write(Addr /*addr*/, const void* /*data*/, Size /*size*/) {
  std::cout << "attempt to write to ROM.\n";
  std::abort();
}

///////////////////////////////////////////////////////////////////////////////

bool MemoryUnit::ADecoder::lookup(Addr a, Size wordSize, mem_accessor_t* ma) {
  Addr e = a + (wordSize - 1);
  assert(e >= a);
  for (auto iter = entries_.rbegin(), iterE = entries_.rend(); iter != iterE; ++iter) {
    if (a >= iter->start && e <= iter->end) {
      ma->md   = iter->md;
      ma->addr = a - iter->start;
      return true;
    }
  }
  return false;
}

void MemoryUnit::ADecoder::map(Addr a, Addr e, MemDevice &m) {
  assert(e >= a);
  entry_t entry{&m, a, e};
  entries_.emplace_back(entry);
}

void MemoryUnit::ADecoder::read(Addr addr, void *data, Size size) {
  mem_accessor_t ma;
  if (!this->lookup(addr, size, &ma)) {
    std::cout << "lookup of 0x" << std::hex << addr << " failed.\n";
    throw BadAddress();
  }      
  ma.md->read(ma.addr, data, size);
}

void MemoryUnit::ADecoder::write(Addr addr, const void *data, Size size) {
  mem_accessor_t ma;
  if (!this->lookup(addr, size, &ma)) {
    std::cout << "lookup of 0x" << std::hex << addr << " failed.\n";
    throw BadAddress();
  }
  ma.md->write(ma.addr, data, size);
}

///////////////////////////////////////////////////////////////////////////////

MemoryUnit::MemoryUnit(Size pageSize, Size addrBytes, bool disableVm)
  : pageSize_(pageSize)
  , addrBytes_(addrBytes)
  , disableVm_(disableVm) {
  if (!disableVm) {
    tlb_[0] = TLBEntry(0, 077);
  }
}

void MemoryUnit::attach(MemDevice &m, Addr start, Addr end) {
  decoder_.map(start, end, m);
}

MemoryUnit::TLBEntry MemoryUnit::tlbLookup(Addr vAddr, Word flagMask) {
  auto iter = tlb_.find(vAddr / pageSize_);
  if (iter != tlb_.end()) {
    if (iter->second.flags & flagMask)
      return iter->second;
    else {
      D(2, "Page fault on addr 0x" << std::hex << vAddr << "(bad flags)");
      throw PageFault(vAddr, false);
    }
  } else {
    D(2, "Page fault on addr 0x" << std::hex << vAddr << "(not in TLB)");
    throw PageFault(vAddr, true);
  }
}

void MemoryUnit::read(Addr addr, void *data, Size size, bool sup) {
  Addr pAddr;
  if (disableVm_) {
    pAddr = addr;
  } else {
    Word flagMask = sup ? 8 : 1;
    TLBEntry t = this->tlbLookup(addr, flagMask);
    pAddr = t.pfn * pageSize_ + addr % pageSize_;
  }
  return decoder_.read(pAddr, data, size);
}

void MemoryUnit::write(Addr addr, const void *data, Size size, bool sup) {
  Addr pAddr;
  if (disableVm_) {
    pAddr = addr;
  } else {
    Word flagMask = sup ? 16 : 2;
    TLBEntry t = tlbLookup(addr, flagMask);
    pAddr = t.pfn * pageSize_ + addr % pageSize_;
  }
  decoder_.write(pAddr, data, size);
}

void MemoryUnit::tlbAdd(Addr virt, Addr phys, Word flags) {
  D(1, "tlbAdd(0x" << std::hex << virt << ", 0x" << phys << ", 0x" << flags << ')');
  tlb_[virt / pageSize_] = TLBEntry(phys / pageSize_, flags);
}

void MemoryUnit::tlbRm(Addr va) {
  if (tlb_.find(va / pageSize_) != tlb_.end())
    tlb_.erase(tlb_.find(va / pageSize_));
}

///////////////////////////////////////////////////////////////////////////////

RAM::RAM(uint32_t num_pages, uint32_t page_size) 
  : page_bits_(log2ceil(page_size)) {    
    assert(ispow2(page_size));
  mem_.resize(num_pages, NULL);
  uint64_t sizel = uint64_t(mem_.size()) << page_bits_;
  size_ = (sizel <= 0xFFFFFFFF) ? sizel : 0xffffffff;
}

RAM::~RAM() {
  this->clear();
}

void RAM::clear() {
  for (auto& page : mem_) {
    delete[] page;
    page = NULL;
  }
}

Size RAM::size() const {
  return size_;
}

uint8_t *RAM::get(uint32_t address) {
  uint32_t page_size   = 1 << page_bits_;
  uint32_t page_index  = address >> page_bits_;
  uint32_t byte_offset = address & ((1 << page_bits_) - 1);

  uint8_t* &page = mem_.at(page_index);
  if (page == NULL) {
    uint8_t *ptr = new uint8_t[page_size];
    // set uninitialized data to "baadf00d"
    for (uint32_t i = 0; i < page_size; ++i) {
      ptr[i] = (0xbaadf00d >> ((i & 0x3) * 8)) & 0xff;
    }
    page = ptr;
  }
  return page + byte_offset;
}

void RAM::read(Addr addr, void *data, Size size) {
  Byte* d = (Byte*)data;
  for (unsigned i = 0; i < size; i++) {
    d[i] = *this->get(addr + i);
  }
}

void RAM::write(Addr addr, const void *data, Size size) {
  const Byte* s = (const Byte*)data;
  for (unsigned i = 0; i < size; i++) {
    *this->get(addr + i) = s[i];
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

void RAM::loadHexImage(std::string path) {
  this->clear();
  FILE *fp = fopen(&path[0], "r");
  if (fp == 0) {
    std::cout << path << " not found" << std::endl;
  }

  fseek(fp, 0, SEEK_END);
  uint32_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char *content = new char[size];
  int x = fread(content, 1, size, fp);
  if (!x) {
    std::cout << "COULD NOT READ FILE\n";
    std::abort();
  }

  int offset = 0;
  char *line = content;
  
  while (1) {
    if (line[0] == ':') {
      uint32_t byteCount = hToI_old(line + 1, 2);
      uint32_t nextAddr = hToI_old(line + 3, 4) + offset;
      uint32_t key = hToI_old(line + 7, 2);
      switch (key) {
      case 0:
        for (uint32_t i = 0; i < byteCount; i++) {
          unsigned add = nextAddr + i;
          *this->get(add) = hToI_old(line + 9 + i * 2, 2);
        }
        break;
      case 2:
        offset = hToI_old(line + 9, 4) << 4;
        break;
      case 4:
        offset = hToI_old(line + 9, 4) << 16;
        break;
      default:
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