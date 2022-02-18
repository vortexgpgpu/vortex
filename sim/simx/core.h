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
#include "cachesim.h"
#include "sharedmem.h"
#include "ibuffer.h"
#include "scoreboard.h"
#include "exeunit.h"
#include "texunit.h"
#include "rasterunit.h"
#include "ropunit.h"

namespace vortex {

class Core : public SimObject<Core> {
public:
  struct PerfStats {
    uint64_t instrs;
    uint64_t ibuf_stalls;
    uint64_t scrb_stalls;
    uint64_t alu_stalls;
    uint64_t lsu_stalls;
    uint64_t csr_stalls;
    uint64_t fpu_stalls;
    uint64_t gpu_stalls;
    uint64_t loads;
    uint64_t stores;
    uint64_t branches;
    uint64_t mem_reads;
    uint64_t mem_writes;
    uint64_t mem_latency;
    uint64_t tex_reads;
    uint64_t tex_latency;

    PerfStats() 
      : instrs(0)
      , ibuf_stalls(0)
      , scrb_stalls(0)
      , alu_stalls(0)
      , lsu_stalls(0)
      , csr_stalls(0)
      , fpu_stalls(0)
      , gpu_stalls(0)
      , loads(0)
      , stores(0)
      , branches(0)
      , mem_reads(0)
      , mem_writes(0)
      , mem_latency(0)
      , tex_reads(0)
      , tex_latency(0)
    {}
  };

  SimPort<MemRsp> MemRspPort;
  SimPort<MemReq> MemReqPort;

  Core(const SimContext& ctx, const ArchDef &arch, uint32_t id);
  ~Core();

  void attach_ram(RAM* ram);

  bool running() const;

  void reset();

  void tick();

  uint32_t id() const {
    return id_;
  }

  const ArchDef& arch() const {
    return arch_;
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

  uint32_t getIRegValue(int reg) const {
    return warps_.at(0)->getIRegValue(reg);
  }

  uint32_t get_csr(uint32_t addr, uint32_t tid, uint32_t wid);
  
  void set_csr(uint32_t addr, uint32_t value, uint32_t tid, uint32_t wid);

  WarpMask wspawn(uint32_t num_warps, uint32_t nextPC);
  
  WarpMask barrier(uint32_t bar_id, uint32_t count, uint32_t warp_id);

  AddrType get_addr_type(uint64_t addr);

  void icache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_write(const void* data, uint64_t addr, uint32_t size);

  uint32_t tex_read(uint32_t stage, uint32_t lod, uint32_t u, uint32_t v, std::vector<mem_addr_size_t>* mem_addrs);

  void trigger_ecall();

  void trigger_ebreak();

  bool check_exit() const;

private:

  void schedule();
  void fetch();
  void decode();
  void execute();
  void commit();
  
  void writeToStdOut(const void* data, uint64_t addr, uint32_t size);

  void cout_flush();

  uint32_t id_;
  const ArchDef& arch_;
  const Decoder decoder_;
  MemoryUnit mmu_;  
  TexUnit tex_unit_;
  RasterUnit raster_unit_;
  RopUnit rop_unit_;

  std::vector<std::shared_ptr<Warp>> warps_;  
  std::vector<WarpMask> barriers_;  
  std::vector<uint32_t> csrs_;
  std::vector<Byte> fcsrs_;
  std::vector<IBuffer> ibuffers_;
  Scoreboard scoreboard_;
  std::vector<ExeUnit::Ptr> exe_units_;
  CacheSim::Ptr icache_;
  CacheSim::Ptr dcache_;
  SharedMem::Ptr sharedmem_;
  Switch<MemReq, MemRsp>::Ptr l1_mem_switch_;
  std::vector<Switch<MemReq, MemRsp>::Ptr> dcache_switch_;

  PipelineLatch fetch_latch_;
  PipelineLatch decode_latch_;
  
  HashTable<pipeline_trace_t*> pending_icache_;
  WarpMask active_warps_;
  WarpMask stalled_warps_;
  uint64_t issued_instrs_;
  uint64_t committed_instrs_;  
  bool ecall_;
  bool ebreak_;

  std::unordered_map<int, std::stringstream> print_bufs_;
  
  PerfStats perf_stats_;
  uint64_t perf_mem_pending_reads_;

  friend class Warp;
  friend class LsuUnit;
  friend class AluUnit;
  friend class CsrUnit;
  friend class FpuUnit;
  friend class GpuUnit;
};

} // namespace vortex