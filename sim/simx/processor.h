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

  uint32_t get_satp();//added
  void set_satp(uint32_t satp);//added
private:
  class Impl;
  Impl* impl_;

  uint32_t satp;//added
};

}