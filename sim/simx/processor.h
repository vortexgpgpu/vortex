#pragma once

namespace vortex {

class ArchDef;
class RAM;
class VirtualDevice;

class Processor {
public:
  Processor(const ArchDef& arch);
  ~Processor();

  void attach_ram(RAM* mem);
  void attachVirtualDevice(VirtualDevice* virtualDevice);

  int run();

private:
  class Impl;
  Impl* impl_;
};

}