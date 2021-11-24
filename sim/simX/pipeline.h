
#pragma once

#include <memory>
#include <iostream>
#include <util.h>
#include "types.h"
#include "archdef.h"
#include "debug.h"

namespace vortex {

struct pipeline_trace_t {
  //--
  uint64_t    id;
  
  //--
  int         cid;
  int         wid;  
  ThreadMask  tmask;
  Word        PC;

  //--
  bool        fetch_stall;
  bool        pipeline_stall;

  //--
  bool        wb;  
  RegType     rdest_type;
  int         rdest;

  //--
  RegMask     used_iregs;
  RegMask     used_fregs;
  RegMask     used_vregs;

  //- 
  ExeType     exe_type; 

  //--
  std::vector<std::vector<uint64_t>> mem_addrs;
  
  //--
  union {
    struct {        
      LsuType type;
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
  uint64_t tex_latency;

  pipeline_trace_t(uint64_t id_, const ArchDef& arch) {
    id  = id_;
    cid = 0;
    wid = 0;
    tmask.reset();
    PC  = 0;
    fetch_stall = false;
    pipeline_stall = false;
    wb  = false;
    rdest = 0;
    rdest_type = RegType::None;
    used_iregs.reset();
    used_fregs.reset();
    used_vregs.reset();
    exe_type = ExeType::NOP;
    mem_addrs.resize(arch.num_threads());    
    icache_latency = 0;
    dcache_latency = 0;
    tex_latency    = 0;
  }

  bool check_stalled(bool stall) {
    bool old = pipeline_stall;
    pipeline_stall = stall;
    return stall ? old : true;
  }
};

inline std::ostream &operator<<(std::ostream &os, const pipeline_trace_t& state) {
  os << "coreid=" << state.cid << ", wid=" << state.wid << ", PC=" << std::hex << state.PC;
  os << ", wb=" << state.wb;
  if (state.wb) {
     os << ", rd=" << state.rdest_type << std::dec << state.rdest;
  }
  os << ", ex=" << state.exe_type;
  os << " (#" << std::dec << state.id << ")";
  return os;
}

class PipelineStage : public Queue<pipeline_trace_t*> {
protected:
  const char* name_;

public:
  PipelineStage(const char* name = nullptr) 
    : name_(name) 
  {}
};

}