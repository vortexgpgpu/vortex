#pragma once

#include "core.h"

namespace vortex {

class Processor : public SimObject<Processor> {
public:
  SimPort<MemReq> MemReqPort;
  SimPort<MemRsp> MemRspPort;
  
  Processor(const SimContext& ctx, const ArchDef& arch);
  ~Processor();

  void attach_ram(RAM* mem);

  bool check_exit(int* exitcode);

  void step(uint64_t cycle);

private:
  class Impl;
  Impl* impl_;
};

}