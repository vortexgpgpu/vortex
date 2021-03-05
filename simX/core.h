#pragma once

#include <string>
#include <vector>
#include <list>
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

  const ArchDef& arch() const {
    return arch_;
  }  

  Word interruptEntry() const {
    return interruptEntry_;
  }

  unsigned long num_insts() const {
    return num_insts_;
  }

  unsigned long num_steps() const {
    return steps_;
  } 

  Word get_csr(Addr addr, int tid, int wid);
  
  void set_csr(Addr addr, Word value);

  void barrier(int bar_id, int count, int warp_id);

  Word icache_fetch(Addr, bool sup);

  Word dcache_read(Addr, bool sup);

  void dcache_write(Addr, Word, bool sup, Size);  

private: 

  void fetch();
  void decode();
  void scheduler();
  void execute_unit();
  void load_store();
  void writeback();

  void getCacheDelays(trace_inst_t *);
  void warpScheduler();

  std::vector<std::vector<bool>> iRenameTable_;
  std::vector<std::vector<bool>> fRenameTable_;
  std::vector<bool> vRenameTable_;
  std::vector<bool> stalled_warps_;

  Word id_;
  const ArchDef &arch_;
  Decoder &decoder_;
  MemoryUnit &mem_;
#ifdef SM_ENABLE
  RAM shared_mem_;
#endif
  std::vector<Warp> warps_;  
  std::vector<WarpMask> barriers_;  
  std::vector<Word> csrs_;
  int schedule_w_;
  uint64_t steps_;
  uint64_t num_insts_;
  Word interruptEntry_;
  bool foundSchedule_;

  trace_inst_t inst_in_fetch_;
  trace_inst_t inst_in_decode_;
  trace_inst_t inst_in_scheduler_;
  trace_inst_t inst_in_exe_;
  trace_inst_t inst_in_lsu_;
  trace_inst_t inst_in_wb_;
};

} // namespace vortex