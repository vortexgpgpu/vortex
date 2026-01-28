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

#ifndef __WARP_H
#define __WARP_H

#include <vector>
#include <sstream>
#include <stack>
#include <mem.h>
#include "types.h"
#include "instr.h"
#ifdef EXT_TCU_ENABLE
#include "tcu/tensor_unit.h"
#endif
#ifdef EXT_V_ENABLE
#include "vpu/vec_unit.h"
#endif

class DebugModule;  // Forward declaration (global scope)

namespace vortex {

class Arch;
class DCRS;
class Core;
class Instr;
class instr_trace_t;

struct ipdom_entry_t {
  ThreadMask  orig_tmask;
  Word        else_PC;
  bool        fallthrough;

  ipdom_entry_t(const ThreadMask &tmask, Word PC)
    : orig_tmask (tmask)
    , else_PC    (PC)
    , fallthrough(false)
  {}
};

///////////////////////////////////////////////////////////////////////////////

struct warp_t {
  std::vector<std::vector<Word>>    ireg_file;
  std::vector<std::vector<uint64_t>>freg_file;
  std::deque<Instr::Ptr>            ibuffer;
  std::stack<ipdom_entry_t>         ipdom_stack;
  ThreadMask                        tmask;
  Word                              PC;
  Byte                              fcsr;
  uint32_t                          uuid;

  warp_t(uint32_t num_threads);

  void reset(uint64_t startup_addr);
};

///////////////////////////////////////////////////////////////////////////////

struct wspawn_t {
  bool      valid;
  uint32_t  num_warps;
  Word      nextPC;
};

///////////////////////////////////////////////////////////////////////////////

class Emulator {
public:
  Emulator(const Arch &arch, const DCRS &dcrs, Core* core);

  ~Emulator();

  void reset();

  void attach_ram(RAM* ram);

#ifdef VM_ENABLE
  void set_satp(uint64_t satp) ;
#endif

  instr_trace_t* step();

  bool running() const;

  void suspend(uint32_t wid);

  void resume(uint32_t wid);

  bool barrier(uint32_t bar_id, uint32_t count, uint32_t wid);

  // Async barrier arrive: returns token (current generation)
  uint32_t barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t wid);

  // Async barrier wait: uses token to determine which phase to wait for
  bool barrier_wait(uint32_t bar_id, uint32_t token, uint32_t wid);

  bool wspawn(uint32_t num_warps, Word nextPC);

  bool setTmask(uint32_t wid, const ThreadMask& tmask);

  int get_exitcode() const;

  void dcache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_write(const void* data, uint64_t addr, uint32_t size);

  const auto& active_warps() const {
    return active_warps_;
  }

  const auto& stalled_warps() const {
    return stalled_warps_;
  }

private:

  uint32_t fetch(uint32_t wid, uint64_t uuid);

  void decode(uint32_t code, uint32_t wid, uint64_t uuid);

  instr_trace_t* execute(const Instr &instr, uint32_t wid);

  void fetch_registers(std::vector<reg_data_t>& out, uint32_t wid, uint32_t src_index, const RegOpd& reg);

  void icache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_amo_reserve(uint64_t addr);

  bool dcache_amo_check(uint64_t addr);

  void writeToStdOut(const void* data, uint64_t addr, uint32_t size);

  void cout_flush();

  Word get_csr(uint32_t addr, uint32_t wid, uint32_t tid);

  void set_csr(uint32_t addr, Word value, uint32_t wid, uint32_t tid);

  uint32_t get_fpu_rm(uint32_t funct3, uint32_t wid, uint32_t tid);

  void update_fcrs(uint32_t fflags, uint32_t wid, uint32_t tid);

  // temporarily added for riscv-vector tests
  // TODO: remove once ecall/ebreak are supported
  void trigger_ecall();
  void trigger_ebreak();

  const Arch& arch_;
  const DCRS& dcrs_;
  Core*       core_;
  ::DebugModule* debug_module_;

  std::vector<warp_t> warps_;
  WarpMask    active_warps_;
  WarpMask    stalled_warps_;
  std::vector<WarpMask> barriers_;
  std::unordered_map<int, std::stringstream> print_bufs_;
  MemoryUnit  mmu_;
  uint32_t    ipdom_size_;
  Word        csr_mscratch_;
  wspawn_t    wspawn_;

  struct AsyncBarrier {
    #define MAX_WARPS 32

    WarpMask arrived_mask;
    WarpMask waiting_mask;
    uint32_t arrived_count;
    uint32_t expect_count;
    uint32_t generation;
    
    std::array<uint32_t, MAX_WARPS> wait_phase;

    AsyncBarrier()
        : arrived_count(0)
        , expect_count(0)
        , generation(0)
    {
        arrived_mask.reset();
        waiting_mask.reset();
        wait_phase.fill(0);
    }

    void reset_for_next_gen() {
        arrived_mask.reset();
        waiting_mask.reset();
        arrived_count = 0;
        generation = 0;
        expect_count = 0;
        wait_phase.fill(0);
    }
};


std::vector<AsyncBarrier> async_barriers_;  

  struct ClusterAsyncBarrier {
    WarpMask arrived_warps;
    uint32_t expect_cores;
    uint32_t token;
    bool token_valid;
    bool core_arrived;

    ClusterAsyncBarrier()
        : expect_cores(0)
        , token(0)
        , token_valid(false)
        , core_arrived(false) {
      arrived_warps.reset();
    }

    void reset() {
      arrived_warps.reset();
      expect_cores = 0;
      token = 0;
      token_valid = false;
      core_arrived = false;
    }
  };

  std::vector<ClusterAsyncBarrier> cluster_async_barriers_;


#ifdef EXT_TCU_ENABLE
  TensorUnit::Ptr tensor_unit_;
#endif

#ifdef EXT_V_ENABLE
  VecUnit::Ptr vec_unit_;
#endif

  std::vector<AsyncBarrier> async_barriers_;
  PoolAllocator<Instr, 64> instr_pool_;
};

}

#endif