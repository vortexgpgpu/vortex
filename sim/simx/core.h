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
#include "arch.h"
#include "decode.h"
#include "mem.h"
#include "warp.h"
#include "pipeline.h"
#include "cache_sim.h"
#include "shared_mem.h"
#include "ibuffer.h"
#include "scoreboard.h"
#include "exe_unit.h"
#include "tex_unit.h"
#include "raster_unit.h"
#include "rop_unit.h"
#include "dcrs.h"

namespace vortex {

class Cluster;

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
    uint64_t tex_issue_stalls;
    uint64_t rop_issue_stalls;
    uint64_t raster_issue_stalls;
    uint64_t ifetches;
    uint64_t loads;
    uint64_t stores;
    uint64_t ifetch_latency;
    uint64_t load_latency;

    PerfStats() 
      : instrs(0)
      , ibuf_stalls(0)
      , scrb_stalls(0)
      , alu_stalls(0)
      , lsu_stalls(0)
      , csr_stalls(0)
      , fpu_stalls(0)
      , gpu_stalls(0)
      , tex_issue_stalls(0)
      , rop_issue_stalls(0)
      , raster_issue_stalls(0)
      , ifetches(0)
      , loads(0)
      , stores(0)
      , ifetch_latency(0)
      , load_latency(0)
    {}
  };

  std::vector<SimPort<MemReq>> icache_req_ports;
  std::vector<SimPort<MemRsp>> icache_rsp_ports;

  std::vector<SimPort<MemReq>> dcache_req_ports;
  std::vector<SimPort<MemRsp>> dcache_rsp_ports;

  Core(const SimContext& ctx, 
       uint32_t core_id, 
       Cluster* cluster,
       const Arch &arch, 
       const DCRS &dcrs,
       SharedMem::Ptr  sharedmem,
       std::vector<RasterUnit::Ptr>& raster_units,
       std::vector<RopUnit::Ptr>& rop_units,
       std::vector<TexUnit::Ptr>& tex_units);

  ~Core();

  void attach_ram(RAM* ram);

  bool running() const;

  void reset();

  void tick();

  uint32_t id() const {
    return core_id_;
  }

  const Arch& arch() const {
    return arch_;
  }

  const DCRS& dcrs() const {
    return dcrs_;
  }

  uint32_t getIRegValue(int reg) const {
    return warps_.at(0)->getIRegValue(reg);
  }

  uint32_t get_csr(uint32_t addr, uint32_t tid, uint32_t wid);
  
  void set_csr(uint32_t addr, uint32_t value, uint32_t tid, uint32_t wid);

  WarpMask wspawn(uint32_t num_warps, Word nextPC);
  
  WarpMask barrier(uint32_t bar_id, uint32_t count, uint32_t warp_id);

  AddrType get_addr_type(uint64_t addr);

  void icache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_write(const void* data, uint64_t addr, uint32_t size);

  void trigger_ecall();

  void trigger_ebreak();

  bool check_exit() const;

  uint32_t raster_idx() {
    auto ret = raster_idx_++;
    raster_idx_ %= raster_units_.size();
    return ret;
  }

  uint32_t rop_idx() {
    auto ret = rop_idx_++;
    rop_idx_ %= rop_units_.size();
    return ret;
  }

  uint32_t tex_idx() {
    auto ret = tex_idx_++;
    tex_idx_ %= tex_units_.size();
    return ret;
  }

private:

  void schedule();
  void fetch();
  void decode();
  void execute();
  void commit();
  
  void writeToStdOut(const void* data, uint64_t addr, uint32_t size);

  void cout_flush();

  uint32_t core_id_;
  const Arch& arch_;
  const DCRS &dcrs_;
  
  const Decoder decoder_;
  MemoryUnit mmu_;

  std::vector<std::shared_ptr<Warp>> warps_;  
  std::vector<WarpMask> barriers_;
  std::vector<Byte> fcsrs_;
  std::vector<IBuffer> ibuffers_;
  Scoreboard scoreboard_;
  std::vector<ExeUnit::Ptr>    exe_units_;  
  std::vector<RasterUnit::Ptr> raster_units_;
  std::vector<RopUnit::Ptr>    rop_units_;
  std::vector<TexUnit::Ptr>    tex_units_;
  SharedMem::Ptr  sharedmem_;

  PipelineLatch fetch_latch_;
  PipelineLatch decode_latch_;
  
  HashTable<pipeline_trace_t*> pending_icache_;
  WarpMask active_warps_;
  WarpMask stalled_warps_;
  uint64_t issued_instrs_;
  uint64_t committed_instrs_;  
  bool ecall_;
  bool ebreak_;

  uint64_t pending_ifetches_;

  std::unordered_map<int, std::stringstream> print_bufs_;

  std::vector<std::vector<CSRs>> csrs_;
  
  PerfStats perf_stats_;
  
  Cluster* cluster_;

  uint32_t raster_idx_;
  uint32_t rop_idx_;
  uint32_t tex_idx_;

  friend class Warp;
  friend class LsuUnit;
  friend class AluUnit;
  friend class CsrUnit;
  friend class FpuUnit;
  friend class GpuUnit;
  friend class TexUnit;
  friend class RasterAgent;
  friend class RopAgent;
  friend class TexAgent;
};

} // namespace vortex