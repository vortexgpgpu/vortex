#pragma once

namespace vortex {

class RAM;

class Processor {
public:
  
  Processor();
  virtual ~Processor();

  void attach_ram(RAM* ram);

  void reset();

  int run();

private:

  class Impl;
  Impl* impl_;
};

}