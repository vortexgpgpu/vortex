#pragma once

#include <stdio.h>
#include <stdint.h>

class RAM {
private:

  mutable uint8_t *mem_[(1 << 12)];      

  uint8_t *get(uint32_t address) const {
    uint32_t block_addr   = address >> 20;
    uint32_t block_offset = address & 0x000FFFFF;
    if (mem_[block_addr] == NULL) {
      mem_[block_addr] = new uint8_t[(1 << 20)];
    }
    return mem_[block_addr] + block_offset;
  }

public:

  RAM() {
    for (uint32_t i = 0; i < (1 << 12); i++) {
      mem_[i] = NULL;
    }
  }

  ~RAM() {
    this->clear();
  }

  size_t size() const {
    return (1ull << 32);
  }

  void clear() {
    for (uint32_t i = 0; i < (1 << 12); i++) {
      if (mem_[i]) {
        delete [] mem_[i];
        mem_[i] = NULL;
      }
    }
  }

  void read(uint32_t address, uint32_t length, uint8_t *data) const {
    for (unsigned i = 0; i < length; i++) {
      data[i] = *this->get(address + i);
    }
  }

  void write(uint32_t address, uint32_t length, const uint8_t *data) {
    for (unsigned i = 0; i < length; i++) {
      *this->get(address + i) = data[i];
    }
  }

  uint8_t& operator[](uint32_t address) {
    return *get(address);
  }

  const uint8_t& operator[](uint32_t address) const {
    return *get(address);
  }
};