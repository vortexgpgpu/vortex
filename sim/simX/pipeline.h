
#pragma once

#include <memory>
#include <iostream>
#include <util.h>
#include "types.h"
#include "debug.h"

namespace vortex {

struct pipeline_state_t {
  //--    
  int         wid;  
  ThreadMask  tmask;
  Word        PC;

  //--
  bool        stall_warp;
  int         rdest_type;
  int         rdest;
  RegMask     used_iregs;
  RegMask     used_fregs;
  RegMask     used_vregs;

  //- 
  ExeType     exe_type; 
  std::vector<uint64_t> mem_addrs;
  
  //--
  union {
    struct {        
      uint8_t load : 1;
      uint8_t store: 1;
      uint8_t fence : 1;
      uint8_t prefetch: 1;
    } lsu;
    struct {
      AluType type;
    } alu;
    struct {
      FpuType type;
    } fpu;
    struct {
      GpuType type;
    } gpu;
  };

  // stats
  uint64_t icache_latency;
  uint64_t dcache_latency;
};

class PipelineStage : public Queue<pipeline_state_t> {
protected:
  const char* name_;
  friend std::ostream &operator<<(std::ostream &, const pipeline_state_t&);

public:
  PipelineStage(const char* name = nullptr) 
    : name_(name) 
  {}
};

inline std::ostream &operator<<(std::ostream &os, const pipeline_state_t& state) {
  os << "stall_warp="   << state.stall_warp;
  os << ", wid="        << state.wid;
  os << ", PC="         << std::hex << state.PC;
  os << ", used_iregs=" << state.used_iregs;
  os << ", used_fregs=" << state.used_fregs;
  os << ", used_vregs=" << state.used_vregs;
  os << std::endl;
  return os;
}

}