#pragma once

#include <string>
#include <vector>
#include <stack>
#include <unordered_map>
#include <set>

#include "debug.h"
#include "types.h"
#include "archdef.h"
#include "decode.h"
#include "mem.h"
#include "warp.h"
#include "trace.h"

namespace vortex {

class Core {
public:
  Core(const ArchDef &arch, Decoder &decoder, MemoryUnit &mem, Word id = 0);
  ~Core();

  bool running() const;

  void getCacheDelays(trace_inst_t *);
  void warpScheduler();
  void fetch();
  void decode();
  void scheduler();
  void execute_unit();
  void load_store();
  void writeback();

  void step();

  void printStats() const;

  Word id() const {
    return id_;
  }

  Warp& warp(int i) {
    return warps_[i];
  }

  Decoder& decoder() {
    return decoder_;
  }

  MemoryUnit& mem() {
    return mem_;
  }

  const ArchDef& arch() const {
    return arch_;
  }  

  Word interruptEntry() const {
    return interruptEntry_;
  }

  unsigned long num_instructions() const {
    return num_instructions_;
  }

  unsigned long num_steps() const {
    return steps_;
  }

private: 

  bool renameTable_[32][32];
  bool vecRenameTable_[32];
  bool stalled_warps_[32];
  bool foundSchedule_;

  Word id_;
  const ArchDef &arch_;
  Decoder &decoder_;
  MemoryUnit &mem_;
  std::vector<Warp> warps_;  
  std::unordered_map<Word, std::set<Warp *>> barriers_;  
  int schedule_w_;
  uint64_t steps_;
  uint64_t num_instructions_;
  Word interruptEntry_;
  bool release_warp_;
  int release_warp_num_;  

  trace_inst_t inst_in_fetch_;
  trace_inst_t inst_in_decode_;
  trace_inst_t inst_in_scheduler_;
  trace_inst_t inst_in_exe_;
  trace_inst_t inst_in_lsu_;
  trace_inst_t inst_in_wb_;
};

} // namespace vortex