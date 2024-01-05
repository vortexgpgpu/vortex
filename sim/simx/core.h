// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
#include <mem.h>
#include "debug.h"
#include "types.h"
#include "arch.h"
#include "decode.h"
#include "warp.h"
#include "pipeline.h"
#include "cache_sim.h"
#include "shared_mem.h"
#include "ibuffer.h"
#include "scoreboard.h"
#include "operand.h"
#include "dispatcher.h"
#include "exe_unit.h"
#include "dcrs.h"

namespace vortex {

class Socket;

using TraceSwitch = Mux<pipeline_trace_t*>;

class Core : public SimObject<Core> {
public:
  struct PerfStats {
    uint64_t cycles;
    uint64_t instrs;
    uint64_t sched_idle;
    uint64_t sched_stalls;
    uint64_t ibuf_stalls;
    uint64_t scrb_stalls;
    uint64_t scrb_alu;
    uint64_t scrb_fpu;
    uint64_t scrb_lsu;
    uint64_t scrb_sfu;
    uint64_t scrb_wctl;
    uint64_t scrb_csrs;
    uint64_t ifetches;
    uint64_t loads;
    uint64_t stores;
    uint64_t ifetch_latency;
    uint64_t load_latency;

    PerfStats() 
      : cycles(0)
      , instrs(0)
      , sched_idle(0)
      , sched_stalls(0)
      , ibuf_stalls(0)
      , scrb_stalls(0)
      , scrb_alu(0)
      , scrb_fpu(0)
      , scrb_lsu(0)
      , scrb_sfu(0)
      , scrb_wctl(0)
      , scrb_csrs(0)
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
       Socket* socket,
       const Arch &arch, 
       const DCRS &dcrs);

  ~Core();

  void reset();

  void tick();

  void attach_ram(RAM* ram);

  bool running() const;

  void resume();

  uint32_t id() const {
    return core_id_;
  }

  Socket* socket() const {
    return socket_;
  }

  const Arch& arch() const {
    return arch_;
  }

  const DCRS& dcrs() const {
    return dcrs_;
  }

  uint32_t get_csr(uint32_t addr, uint32_t tid, uint32_t wid);
  
  void set_csr(uint32_t addr, uint32_t value, uint32_t tid, uint32_t wid);

  void wspawn(uint32_t num_warps, Word nextPC);
  
  void barrier(uint32_t bar_id, uint32_t count, uint32_t warp_id);

  AddrType get_addr_type(uint64_t addr);

  void icache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_write(const void* data, uint64_t addr, uint32_t size);

  void dcache_amo_reserve(uint64_t addr);

  bool dcache_amo_check(uint64_t addr);

  void trigger_ecall();

  void trigger_ebreak();

  bool check_exit(Word* exitcode, bool riscv_test) const;

private:

  void schedule();
  void fetch();
  void decode();
  void issue();
  void execute();
  void commit();
  
  void writeToStdOut(const void* data, uint64_t addr, uint32_t size);

  void cout_flush();

  uint32_t core_id_;
  Socket* socket_;
  const Arch& arch_;
  const DCRS &dcrs_;
  
  const Decoder decoder_;
  MemoryUnit mmu_;

  std::vector<std::shared_ptr<Warp>> warps_;  
  std::vector<WarpMask> barriers_;
  std::vector<Byte> fcsrs_;
  std::vector<IBuffer> ibuffers_;
  Scoreboard scoreboard_;
  std::vector<Operand::Ptr> operands_;
  std::vector<Dispatcher::Ptr> dispatchers_;
  std::vector<ExeUnit::Ptr> exe_units_;
  SharedMem::Ptr shared_mem_;
  std::vector<SMemDemux::Ptr> smem_demuxs_;

  PipelineLatch fetch_latch_;
  PipelineLatch decode_latch_;
  
  HashTable<pipeline_trace_t*> pending_icache_;
  WarpMask active_warps_;
  WarpMask stalled_warps_;
  uint64_t issued_instrs_;
  uint64_t committed_instrs_;
  bool exited_;

  uint64_t pending_ifetches_;

  std::unordered_map<int, std::stringstream> print_bufs_;

  std::vector<std::vector<CSRs>> csrs_;
  
  PerfStats perf_stats_;
  
  std::vector<TraceSwitch::Ptr> commit_arbs_;

  uint32_t commit_exe_;
  uint32_t ibuffer_idx_;

  friend class Warp;
  friend class LsuUnit;
  friend class AluUnit;
  friend class FpuUnit;
  friend class SfuUnit;
};

} // namespace vortex
