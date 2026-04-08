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

#include <vector>
#include <list>
#include <simobject.h>
#include "types.h"
#include "emulator.h"
#include "cache_sim.h"
#include "local_mem.h"
#include "scoreboard.h"

#ifdef EXT_V_ENABLE
#include "vpu/voperands.h"
#include "vpu/vec_unit.h"
#else
#include "operands.h"
#endif

#include "dispatcher.h"
#include "func_unit.h"
#include "mem_coalescer.h"
#include "VX_config.h"

namespace vortex {

class Socket;
class Arch;

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
  #ifdef EXT_V_ENABLE
    uint64_t vinstrs = 0;
    uint64_t vpu_stalls = 0;
  #endif
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
  #ifdef EXT_V_ENABLE
    uint64_t vpu_instrs = 0;
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
       Socket* socket,
       const Arch &arch
  );

  ~Core();

  void reset();

  void tick();

  void attach_ram(RAM* ram);
#ifdef VM_ENABLE
  void set_satp(uint64_t satp);
#endif

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

  const Arch& arch() const {
    return arch_;
  }

  Socket* socket() const {
    return socket_;
  }

  const LocalMem::Ptr& local_mem() const {
    return local_mem_;
  }

  const MemCoalescer::Ptr& mem_coalescer(uint32_t idx) const {
    return mem_coalescers_.at(idx);
  }

  void mem_read(void* data, uint64_t addr, uint32_t size) {
    return emulator_.mem_read(data, addr, size);
  }

  void mem_write(const void* data, uint64_t addr, uint32_t size) {
    return emulator_.mem_write(data, addr, size);
  }

  int dcr_write(uint32_t addr, uint32_t value);

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value);


#ifdef EXT_TCU_ENABLE
  TensorUnit::Ptr& tensor_unit() {
    return tensor_unit_;
  }
#endif

#ifdef EXT_V_ENABLE
  VecUnit::Ptr& vec_unit() {
    return vec_unit_;
  }
#endif

  auto& trace_pool() {
    return trace_pool_;
  }

  const PerfStats& perf_stats() const;
  PerfStats& perf_stats();

  int get_exitcode() const;

private:

  void schedule();
  void fetch();
  void decode();
  void issue();
  void execute();
  void commit();

  uint32_t core_id_;
  Socket* socket_;
  const Arch& arch_;

#ifdef EXT_TCU_ENABLE
  TensorUnit::Ptr tensor_unit_;
#endif

#ifdef EXT_V_ENABLE
  VecUnit::Ptr vec_unit_;
#endif


  Emulator emulator_;

  std::vector<TFifo<instr_trace_t*>::Ptr> ibuffers_;
  Scoreboard scoreboard_;
  std::vector<Operands::Ptr> operands_;
  std::vector<Dispatcher::Ptr> dispatchers_;
  std::vector<FuncUnit::Ptr> func_units_;
  LocalMem::Ptr local_mem_;
  std::vector<LocalMemSwitch::Ptr> lmem_switch_;
  std::vector<MemCoalescer::Ptr> mem_coalescers_;

  TFifo<instr_trace_t*> fetch_latch_;
  TFifo<instr_trace_t*> decode_latch_;
  instr_trace_t* trace_to_schedule_;

  HashTable<instr_trace_t*> pending_icache_;
  std::list<instr_trace_t*, PoolAllocator<instr_trace_t*, 64>> pending_instrs_;

  uint64_t pending_ifetches_;

  mutable PerfStats perf_stats_;

  std::vector<TraceArbiter::Ptr> commit_arbs_;

  uint32_t commit_exe_;
  std::vector<Arbiter> ibuffer_arbs_;

  // FU lock: prevent warp interleaving during multi-uop sequences
  BitVector<> fu_locked_;

  PoolAllocator<instr_trace_t, 64> trace_pool_;

  friend class LsuUnit;
  friend class AluUnit;
  friend class FpuUnit;
  friend class SfuUnit;
};

} // namespace vortex
