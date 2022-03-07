#pragma once

#include <stdint.h>

namespace vortex {

class Arch;
class RAM;

class Processor {
public:
  Processor(const Arch& arch);
  ~Processor();

  void attach_ram(RAM* mem);

  int run();

  void write_dcr(uint32_t addr, uint64_t value);

private:
  class Impl;
  Impl* impl_;
};

}