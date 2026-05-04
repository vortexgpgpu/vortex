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

#include <memory>
#include <simobject.h>
#include <mempool.h>
#include "types.h"
#include "instr_trace.h"
#include "VX_config.h"

namespace vortex {

class Socket;
class ProcessorImpl;
class Scheduler;
class CsrUnit;
class TcuUnit;
class SfuUnit;
class LocalMem;
class LocalMemSwitch;
class MemCoalescer;

class Core : public SimObject<Core> {
public:
  struct PerfStats {
    uint64_t cycles = 0;
    uint64_t instrs = 0;
    uint64_t sched_idle = 0;
    uint64_t active_warps = 0;
    uint64_t stalled_warps = 0;
    uint64_t issued_warps = 0;
    uint64_t issued_threads = 0;
    uint64_t fetch_stalls = 0;
    uint64_t ibuf_stalls = 0;
    uint64_t scrb_stalls = 0;
    uint64_t opds_stalls = 0;
    uint64_t alu_stalls = 0;
    uint64_t fpu_stalls = 0;
    uint64_t lsu_stalls = 0;
    uint64_t sfu_stalls = 0;
  #ifdef EXT_TCU_ENABLE
    uint64_t tcu_stalls = 0;
  #endif
    uint64_t branches   = 0;
    uint64_t divergence = 0;
    uint64_t alu_instrs = 0;
    uint64_t fpu_instrs = 0;
    uint64_t lsu_instrs = 0;
    uint64_t sfu_instrs = 0;
  #ifdef EXT_TCU_ENABLE
    uint64_t tcu_instrs = 0;
  #endif
    uint64_t ifetches = 0;
    uint64_t loads = 0;
    uint64_t stores = 0;
    uint64_t ifetch_latency = 0;
    uint64_t load_latency = 0;
  };

  std::vector<SimChannel<MemReq>> icache_req_out;
  std::vector<SimChannel<MemRsp>> icache_rsp_in;

  std::vector<SimChannel<MemReq>> dcache_req_out;
  std::vector<SimChannel<MemRsp>> dcache_rsp_in;

  Core(const SimContext& ctx,
       const char* name,
       uint32_t core_id,
       Socket* socket
  );

  ~Core();

  bool running() const;

  void resume(uint32_t wid);

  bool has_pending_instrs(uint32_t wid) const;

  void barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t wid, bool is_sync_bar);

  bool barrier_wait(uint32_t bar_id, uint32_t phase, uint32_t wid);

  void global_barrier_resume(uint32_t bar_id);

  void barrier_event_attach(uint32_t bar_id);

  void barrier_event_release(uint32_t bar_id);

  bool wspawn(uint32_t num_warps, Word nextPC);

  bool setTmask(uint32_t wid, const ThreadMask& tmask);

  uint32_t id() const {
    return core_id_;
  }

  Socket* socket() const {
    return socket_;
  }

  const std::shared_ptr<LocalMem>& local_mem() const;

  const std::shared_ptr<MemCoalescer>& mem_coalescer(uint32_t idx) const;

  // Used by LsuUnit to drive the per-block load/store switch.
  const std::shared_ptr<LocalMemSwitch>& lmem_switch(uint32_t idx) const;

  ProcessorImpl* processor() const;

  Scheduler&  scheduler();
  CsrUnit&    csr_unit();
  uint32_t    mpm_class() const;

  int dcr_write(uint32_t addr, uint32_t value);

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value);

#ifdef VM_ENABLE
  // SATP write — invoked by CsrUnit on kernel `csrw satp`. Fans out to
  // both per-core MMUs (dcache + icache). Translation itself happens
  // asynchronously inside the Mmu SimObject; LSU/fetch emit VAs.
  void set_satp(uint64_t satp);
#endif


#ifdef EXT_TCU_ENABLE
  std::shared_ptr<TcuUnit>& tcu_unit();
#endif

  std::shared_ptr<SfuUnit> sfu_unit();

  PoolAllocator<instr_trace_t, 64>& trace_pool();

  const PerfStats& perf_stats() const;
  PerfStats& perf_stats();

  int get_exitcode() const;

protected:
  void on_reset();
  void on_tick();

private:
  uint32_t core_id_;
  Socket*  socket_;

  class Impl;
  Impl* impl_;

  friend class SimObject<Core>;
};

} // namespace vortex
