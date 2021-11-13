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

namespace vortex {

class Core : public SimObject<Core> {
public:
  Core(const SimContext& ctx, const ArchDef &arch, Decoder &decoder, MemoryUnit &mem, Word id);
  ~Core();

  bool running() const;

  void step(uint64_t cycle);

  void printStats() const;

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

  Word icache_fetch(Addr);

  Word dcache_read(Addr, Size);

  void dcache_write(Addr, Word, Size);

  void trigger_ebreak();

  bool check_ebreak() const;

private:

  void fetch();
  void decode();
  void issue();
  void execute();
  void commit();

  void warp_scheduler();

  void icache_handleCacheReponse(const MemRsp& response, uint32_t port_id);

  void writeToStdOut(Addr addr, Word data);

  Word id_;
  const ArchDef& arch_;
  const Decoder& decoder_;
  MemoryUnit& mem_;
#ifdef SM_ENABLE
  RAM shared_mem_;
#endif 

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
  SlavePort<MemRsp> icache_rsp_port_;
  std::vector<SlavePort<MemRsp>> dcache_rsp_port_;

  PipelineStage fetch_stage_;
  PipelineStage decode_stage_;
  PipelineStage issue_stage_;
  PipelineStage execute_stage_;
  PipelineStage commit_stage_;  
  
  HashTable<pipeline_state_t> pending_icache_;
  WarpMask stalled_warps_;  
  uint32_t last_schedule_wid_;
  uint32_t pending_instrs_;
  bool ebreak_;

  std::unordered_map<int, std::stringstream> print_bufs_;
  uint64_t stats_insts_;
  uint64_t stats_loads_;
  uint64_t stats_stores_;

  friend class LsuUnit;

public:
  SlavePort<MemRsp>  MemRspPort;
  MasterPort<MemReq> MemReqPort;
};

} // namespace vortex