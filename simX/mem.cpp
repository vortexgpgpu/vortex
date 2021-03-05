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

  while (contents_.size() % wordSize)
    contents_.push_back(0x00);
}

RamMemDevice::RamMemDevice(Size size, Size wordSize)
  : wordSize_(wordSize)
  , contents_(size) 
{}

void RomMemDevice::write(Addr, Word) {
  std::cout << "Attempt to write to ROM.\n";
  std::abort();
}

Word RamMemDevice::read(Addr addr) {
  D(2, "RAM read, addr=0x" << std::hex << addr);
  Word w = readWord(contents_, addr, wordSize_ - addr % wordSize_);
  return w;
}

void RamMemDevice::write(Addr addr, Word w) {
  D(2, "RAM write, addr=0x" << std::hex << addr);
  writeWord(contents_, addr, wordSize_ - addr % wordSize_, w);
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

Word MemoryUnit::ADecoder::read(Addr a, bool /*sup*/, Size wordSize) {
  mem_accessor_t ma;
  if (!this->lookup(a, wordSize, &ma)) {
    std::cout << "lookup of 0x" << std::hex << a << " failed.\n";
    throw BadAddress();
  }      
  return ma.md->read(ma.addr);
}

void MemoryUnit::ADecoder::write(Addr a, Word w, bool /*sup*/, Size wordSize) {
  mem_accessor_t ma;
  if (!this->lookup(a, wordSize, &ma)) {
    std::cout << "lookup of 0x" << std::hex << a << " failed.\n";
    throw BadAddress();
  }
  RAM *ram = (RAM *)ma.md;
  if (wordSize == 8) {
    ram->writeByte(ma.addr, &w);
  } else if (wordSize == 16) {
    ram->writeHalf(ma.addr, &w);
  } else {
    ram->writeWord(ma.addr, &w);
  }
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

Word MemoryUnit::read(Addr vAddr, bool sup) {
  Addr pAddr;
  if (disableVm_) {
    pAddr = vAddr;
  } else {
    Word flagMask = sup ? 8 : 1;
    TLBEntry t = this->tlbLookup(vAddr, flagMask);
    pAddr = t.pfn * pageSize_ + vAddr % pageSize_;
  }
  return decoder_.read(pAddr, sup, addrBytes_);
}

Word MemoryUnit::fetch(Addr vAddr, bool sup) {
  Addr pAddr;

  if (disableVm_) {
    pAddr = vAddr;
  } else {
    Word flagMask = sup ? 32 : 4;
    TLBEntry t = this->tlbLookup(vAddr, flagMask);
    pAddr = t.pfn * pageSize_ + vAddr % pageSize_;
  }

  Word instruction = decoder_.read(pAddr, sup, addrBytes_);
  return instruction;
}

void MemoryUnit::write(Addr vAddr, Word w, bool sup, Size bytes) {
  Addr pAddr;

  if (disableVm_) {
    pAddr = vAddr;
  } else {
    Word flagMask = sup ? 16 : 2;
    TLBEntry t = tlbLookup(vAddr, flagMask);
    pAddr = t.pfn * pageSize_ + vAddr % pageSize_;
  }
  decoder_.write(pAddr, w, sup, bytes);
}

void MemoryUnit::tlbAdd(Addr virt, Addr phys, Word flags) {
  D(1, "tlbAdd(0x" << std::hex << virt << ", 0x" << phys << ", 0x" << flags << ')');
  tlb_[virt / pageSize_] = TLBEntry(phys / pageSize_, flags);
}

void MemoryUnit::tlbRm(Addr va) {
  if (tlb_.find(va / pageSize_) != tlb_.end())
    tlb_.erase(tlb_.find(va / pageSize_));
}

void *vortex::consoleInputThread(void * /*arg_vp*/) {
  //--
  return nullptr;
}

///////////////////////////////////////////////////////////////////////////////

DiskControllerMemDevice::DiskControllerMemDevice(Size wordSize, Size blockSize, Core &c)
  : wordSize_(wordSize)
  , blockSize_(blockSize)
  , core_(c)
{}

Word DiskControllerMemDevice::read(Addr a) {
  switch (a / 8) {
  case 0:
    return curDisk_;
  case 1:
    return curBlock_;
  case 2:
    return disks_[curDisk_].blocks * blockSize_;
  case 3:
    return physAddr_;
  case 4:
    return command_;
  case 5:
    return status_;
  default:
    std::cout << "Attempt to read invalid disk controller register.\n";
    std::abort();
  }
}

void DiskControllerMemDevice::write(Addr a, Word w) {
  switch (a / 8) {
  case 0:
    if (w <= disks_.size()) {
      curDisk_ = w;
      status_ = OK;
    } else {
      status_ = INVALID_DISK;
    }
    break;
  case 1:
    if (w < disks_[curDisk_].blocks) {
      curBlock_ = w;
    } else {
      status_ = INVALID_BLOCK;
    }
    break;
  case 2:
    nBlocks_ = w >= disks_[curDisk_].blocks ? disks_[curDisk_].blocks - 1 : w;
    status_ = OK;
    break;
  case 3:
    physAddr_ = w;
    status_ = OK;
    break;
  case 4:
    std::cout << "TODO: Implement disk read and write!\n";
    break;
  }
}

///////////////////////////////////////////////////////////////////////////////

RAM::RAM(uint32_t num_pages, uint32_t page_size) 
  : page_bits_(log2ceil(page_size)) {    
    assert(page_size >= 4);
    assert(ispow2(page_size));
  mem_.resize(num_pages, NULL);
  uint64_t sizel = uint64_t(mem_.size()) << page_bits_;
  size_ = (sizel <= 0xFFFFFFFF) ? sizel : 0xffffffff;
}

RAM::~RAM() {
  for (auto& page : mem_) {
    delete[] page;
  }
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
  uint32_t page_size   = 14 << page_bits_;
  uint32_t page_index  = address >> page_bits_;
  uint32_t byte_offset = address & ((1 << page_bits_) - 1);

  uint8_t* &page = mem_.at(page_index);
  if (page == NULL) {
    uint8_t *ptr = new uint8_t[page_size];
    for (uint32_t i = 0; i < (page_size / 4); ++i) {
      ((uint32_t*)ptr)[i] = 0xddccbbaa;
    }
    page = ptr;
  }
  return page + byte_offset;
}

void RAM::read(uint32_t address, uint32_t length, uint8_t *data) {
  for (unsigned i = 0; i < length; i++) {
    data[i] = *this->get(address + i);
  }
}

void RAM::write(uint32_t address, uint32_t length, uint8_t *data) {
  for (unsigned i = 0; i < length; i++) {
    *this->get(address + i) = data[i];
  }
}

Byte *RAM::base() {
  return (Byte *)this->get(0);
}

void RAM::getBlock(uint32_t address, uint8_t *data) {
  uint32_t block_number = address & 0xffffff00; // To zero out block offset
  uint32_t bytes_num = 256;
  this->read(block_number, bytes_num, data);
}

void RAM::getWord(uint32_t address, uint32_t *data) {
  data[0] = 0;

  uint8_t first  = *get(address + 0);
  uint8_t second = *get(address + 1);
  uint8_t third  = *get(address + 2);
  uint8_t fourth = *get(address + 3);

  data[0] = (data[0] << 0) | fourth;
  data[0] = (data[0] << 8) | third;
  data[0] = (data[0] << 8) | second;
  data[0] = (data[0] << 8) | first;
}

void RAM::writeWord(uint32_t address, uint32_t *data) {
  uint32_t data_to_write = *data;
  uint32_t byte_mask = 0xFF;

  for (int i = 0; i < 4; i++) {
    *this->get(address + i) = data_to_write & byte_mask;
    data_to_write = data_to_write >> 8;
  }
}

void RAM::writeHalf(uint32_t address, uint32_t *data) {
  uint32_t data_to_write = *data;
  uint32_t byte_mask = 0xFF;

  for (int i = 0; i < 2; i++) {
    *this->get(address + i) = data_to_write & byte_mask;
    data_to_write = data_to_write >> 8;
  }
}

void RAM::writeByte(uint32_t address, uint32_t *data) {
  uint32_t data_to_write = *data;
  uint32_t byte_mask = 0xFF;

  *this->get(address) = data_to_write & byte_mask;
  data_to_write = data_to_write >> 8;
}

void RAM::write(Addr addr, Word w) {
  uint32_t word = (uint32_t)w;
  writeWord(addr, &word);
}

Word RAM::read(Addr addr) {
  uint32_t w;
  getWord(addr, &w);
  return (Word)w;
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