#pragma once

#include "core.h"

namespace vortex {

class Processor {
public:
  typedef std::shared_ptr<Processor> Ptr;
  
  Processor(const ArchDef& arch);
  ~Processor();

  void attach_ram(RAM* mem);

  int run();

private:
  std::vector<Core::Ptr> cores_;  
  std::vector<Cache::Ptr> l2caches_;  
  std::vector<Switch<MemReq, MemRsp>::Ptr> l2_mem_switches_;
  Cache::Ptr l3cache_;
  Switch<MemReq, MemRsp>::Ptr l3_mem_switch_;
  MemSim::Ptr memsim_;
};

}