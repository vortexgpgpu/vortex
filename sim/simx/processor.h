#pragma once

namespace vortex {

class ArchDef;
class RAM;

class Processor {
public:
  Processor(const ArchDef& arch);
  ~Processor();

  void attach_ram(RAM* mem);

  int run();

private:
  class Impl;
  Impl* impl_;
};

}