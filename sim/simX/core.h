#pragma once

#include <string>
#include <vector>
#include <list>
#include <stack>
#include <queue>
#include <unordered_map>
#include <memory>
#include <set>
#include <simobject.h>
#include "debug.h"
#include "types.h"
#include "archdef.h"
#include "decode.h"
#include "mem.h"
#include "warp.h"
#include "pipeline.h"
#include "cache.h"
#include "ibuffer.h"
#include "scoreboard.h"
#include "exeunit.h"
#include "tex_unit.h"

namespace vortex {

class Core : public SimObject<Core> {
public:
  Core(const SimContext& ctx, const ArchDef &arch, Word id);
  ~Core();

  void attach_ram(RAM* ram);

  bool running() const;

  void step(uint64_t cycle);

  Word id() const {
    return id_;
  }

  Warp& warp(int i) {
    return *warps_.at(i);
  }

  const Decoder& decoder() {
    return decoder_;
  }

  const ArchDef& arch() const {
    return arch_;
  }

  unsigned long stats_insts() const {
    return stats_insts_;
  } 

  Word getIRegValue(int reg) const {
    return warps_.at(0)->getIRegValue(reg);
  }

  Word get_csr(Addr addr, int tid, int wid);
  
  void set_csr(Addr addr, Word value, int tid, int wid);

  void barrier(int bar_id, int count, int warp_id);

  Word icache_read(Addr, Size);

  Word dcache_read(Addr, Size);

  void dcache_write(Addr, Word, Size);

  Word tex_read(uint32_t unit, Word lod, Word u, Word v, std::vector<uint64_t>* mem_addrs);

  void trigger_ecall();

  void trigger_ebreak();

  bool check_exit() const;

private:

  void fetch(uint64_t cycle);
  void decode(uint64_t cycle);
  void issue(uint64_t cycle);
  void execute(uint64_t cycle);
  void commit(uint64_t cycle);

  void warp_scheduler(uint64_t cycle);

  void writeToStdOut(Addr addr, Word data);

  Word id_;
  const ArchDef arch_;
  const Decoder decoder_;
  MemoryUnit mmu_;
  RAM shared_mem_;
  std::vector<TexUnit> tex_units_;

  std::vector<std::shared_ptr<Warp>> warps_;  
  std::vector<WarpMask> barriers_;  
  std::vector<Word> csrs_;
  std::vector<Byte> fcsrs_;
  std::vector<IBuffer> ibuffers_;
  Scoreboard scoreboard_;
  std::vector<ExeUnit::Ptr> exe_units_;
  Cache::Ptr icache_;
  Cache::Ptr dcache_;
  Switch<MemReq, MemRsp>::Ptr l1_mem_switch_;
  std::vector<Switch<MemReq, MemRsp>::Ptr> dcache_switch_;

  PipelineStage fetch_stage_;
  PipelineStage decode_stage_;
  PipelineStage issue_stage_;
  PipelineStage execute_stage_;
  PipelineStage commit_stage_;  
  
  HashTable<pipeline_trace_t*> pending_icache_;
  WarpMask stalled_warps_;  
  uint32_t last_schedule_wid_;
  uint32_t issued_instrs_;
  uint32_t committed_instrs_;
  bool ecall_;
  bool ebreak_;

  std::unordered_map<int, std::stringstream> print_bufs_;
  
  uint64_t stats_insts_;

  friend class LsuUnit;
  friend class GpuUnit;

public:
  SlavePort<MemRsp>  MemRspPort;
  MasterPort<MemReq> MemReqPort;
};

} // namespace vortex