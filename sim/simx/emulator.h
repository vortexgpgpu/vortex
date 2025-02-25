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

namespace vortex {

class Arch;
class DCRS;
class Core;
class Instr;
class instr_trace_t;

class Emulator {
public:
  Emulator(const Arch &arch,
           const DCRS &dcrs,
           Core* core);

  ~Emulator();

  void clear();

  void attach_ram(RAM* ram);
#ifdef VM_ENABLE
  void set_satp(uint64_t satp) ;
#endif

  instr_trace_t* step();

  bool running() const;

  void suspend(uint32_t wid);

  void resume(uint32_t wid);

  bool barrier(uint32_t bar_id, uint32_t count, uint32_t wid);

  bool wspawn(uint32_t num_warps, Word nextPC);

  int get_exitcode() const;

  Word get_tiles();
  Word get_tc_size();
  Word get_tc_num();

  void dcache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_write(const void* data, uint64_t addr, uint32_t size);

private:

  struct ipdom_entry_t {
    ipdom_entry_t(const ThreadMask &orig_tmask, const ThreadMask &else_tmask, Word PC)
      : orig_tmask (orig_tmask)
      , else_tmask (else_tmask)
      , PC         (PC)
      , fallthrough(false)
    {}

    ThreadMask  orig_tmask;
    ThreadMask  else_tmask;
    Word        PC;
    bool        fallthrough;
  };

  struct vtype_t {
    uint32_t vill;
    uint32_t vma;
    uint32_t vta;
    uint32_t vsew;
    uint32_t vlmul;
  };

  union reg_data_t {
    Word     u;
    WordI    i;
    WordF    f;
    float    f32;
    double   f64;
    uint32_t u32;
    uint64_t u64;
    int32_t  i32;
    int64_t  i64;
  };

  struct warp_t {
    warp_t(const Arch& arch);
    void clear(uint64_t startup_addr);

    Word                              PC;
    ThreadMask                        tmask;
    std::vector<std::vector<Word>>    ireg_file;
    std::vector<std::vector<uint64_t>>freg_file;
    std::stack<ipdom_entry_t>         ipdom_stack;
    Byte                              fcsr;
#ifdef EXT_V_ENABLE
    std::vector<std::vector<Byte>>    vreg_file;
    vtype_t                           vtype;
    uint32_t                          vl;
    Word                              vlmax;
#endif
    uint32_t                          uuid;
  };

  struct wspawn_t {
    bool valid;
    uint32_t num_warps;
    Word nextPC;
  };

  std::shared_ptr<Instr> decode(uint32_t code) const;

  void execute(const Instr &instr, uint32_t wid, instr_trace_t *trace);

#ifdef EXT_V_ENABLE
  void loadVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata);
  void storeVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata);
  void executeVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata, std::vector<reg_data_t> &rddata);
#endif

  void icache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_amo_reserve(uint64_t addr);

  bool dcache_amo_check(uint64_t addr);

  void writeToStdOut(const void* data, uint64_t addr, uint32_t size);

  void cout_flush();

  Word get_csr(uint32_t addr, uint32_t tid, uint32_t wid);

  void set_csr(uint32_t addr, Word value, uint32_t tid, uint32_t wid);

  uint32_t get_fpu_rm(uint32_t func3, uint32_t tid, uint32_t wid);

  void update_fcrs(uint32_t fflags, uint32_t tid, uint32_t wid);

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
  std::vector<WarpMask> barriers_;
  std::unordered_map<int, std::stringstream> print_bufs_;
  MemoryUnit  mmu_;
  uint32_t    ipdom_size_;
  Word        csr_mscratch_;
  wspawn_t    wspawn_;
  std::vector<Word> scratchpad;
  uint32_t mat_size;
  uint32_t tc_size;
  uint32_t tc_num;
  std::vector<std::vector<std::unordered_map<uint32_t, uint32_t>>> csrs_;
};

}

#endif