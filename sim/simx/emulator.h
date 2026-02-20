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
#include "tensor_unit.h"
#endif
#ifdef EXT_V_ENABLE
#include "vec_unit.h"
#endif

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

  uint32_t barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t wid, bool is_async_bar);

  bool barrier_wait(uint32_t bar_id, uint32_t phase, uint32_t wid);

  void barrier_tx_start(uint32_t bar_id);

  void barrier_tx_done(uint32_t bar_id);

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

  uint32_t get_barrier_phase(uint32_t bar_id) const;

  void release_local_barrier(uint32_t bar_id);

  // temporarily added for riscv-vector tests
  // TODO: remove once ecall/ebreak are supported
  void trigger_ecall();
  void trigger_ebreak();

  const Arch& arch_;
  const DCRS& dcrs_;
  Core*       core_;

  std::vector<warp_t> warps_;
  WarpMask    active_warps_;
  WarpMask    stalled_warps_;
  std::vector<warp_barrier_t> barriers_;
  std::unordered_map<int, std::stringstream> print_bufs_;
  MemoryUnit  mmu_;
  uint32_t    ipdom_size_;
  Word        csr_mscratch_;
  wspawn_t    wspawn_;

  PoolAllocator<Instr, 64> instr_pool_;
};

}

#endif
