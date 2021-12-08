#pragma once

namespace vortex {

class RAM;

class Processor {
public:
  
  Processor();
  ~Processor();

  void attach_ram(RAM* ram);

  int run();

private:

  class Impl;
  Impl* impl_;
};

}