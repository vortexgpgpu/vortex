#pragma once

#include <stdint.h>

namespace vortex {

class RAM;

class Processor {
public:
  
  Processor();
  ~Processor();

  void attach_ram(RAM* ram);

  int run();

  void write_dcr(uint32_t addr, uint32_t value);

private:

  class Impl;
  Impl* impl_;
};

}