#pragma once

#include <stdint.h>

namespace vortex {

class ArchDef;
class RAM;

class Processor {
public:
  Processor(const ArchDef& arch);
  ~Processor();

  void attach_ram(RAM* mem);

  int run();

  void write_csr(uint32_t addr, uint64_t value);

private:
  class Impl;
  Impl* impl_;
};

}