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

enum Constants {
  width_opcode= 7,
  width_reg   = 5,
  width_func2 = 2,
  width_func3 = 3,
  width_func6 = 6,
  width_func7 = 7,
  width_mop   = 3,
  width_vmask = 1,
  width_i_imm = 12,
  width_j_imm = 20,
  width_v_zimm = 11,
  width_v_ma = 1,
  width_v_ta = 1,
  width_v_sew = 3,
  width_v_lmul = 3,
  width_aq    = 1,
  width_rl    = 1,

  shift_opcode= 0,
  shift_rd    = width_opcode,
  shift_func3 = shift_rd + width_reg,
  shift_rs1   = shift_func3 + width_func3,
  shift_rs2   = shift_rs1 + width_reg,
  shift_func2 = shift_rs2 + width_reg,
  shift_func7 = shift_rs2 + width_reg,
  shift_rs3   = shift_func7 + width_func2,
  shift_vmop  = shift_func7 + width_vmask,
  shift_vnf   = shift_vmop + width_mop,
  shift_func6 = shift_func7 + width_vmask,
  shift_vset  = shift_func7 + width_func6,
  shift_v_sew = width_v_lmul,
  shift_v_ta  = shift_v_sew + width_v_sew,
  shift_v_ma  = shift_v_ta + width_v_ta,

  mask_opcode = (1 << width_opcode) - 1,
  mask_reg    = (1 << width_reg)   - 1,
  mask_func2  = (1 << width_func2) - 1,
  mask_func3  = (1 << width_func3) - 1,
  mask_func6  = (1 << width_func6) - 1,
  mask_func7  = (1 << width_func7) - 1,
  mask_i_imm  = (1 << width_i_imm) - 1,
  mask_j_imm  = (1 << width_j_imm) - 1,
  mask_v_zimm = (1 << width_v_zimm) - 1,
  mask_v_ma   = (1 << width_v_ma) - 1,
  mask_v_ta   = (1 << width_v_ta) - 1,
  mask_v_sew  = (1 << width_v_sew) - 1,
  mask_v_lmul  = (1 << width_v_lmul) - 1,
};

struct vtype {
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

  struct warp_t {
    warp_t(const Arch& arch);
    void clear(uint64_t startup_addr);

    Word                              PC;
    ThreadMask                        tmask;
    std::vector<std::vector<Word>>    ireg_file;
    std::vector<std::vector<uint64_t>>freg_file;
    std::vector<std::vector<Byte>>    vreg_file;
    std::stack<ipdom_entry_t>         ipdom_stack;
    Byte                              fcsr;
    uint32_t                          uuid;

    struct vtype vtype;
    uint32_t vl;
    Word VLMAX;
  };

  struct wspawn_t {
    bool valid;
    uint32_t num_warps;
    Word nextPC;
  };

  std::shared_ptr<Instr> decode(uint32_t code) const;

  void execute(const Instr &instr, uint32_t wid, instr_trace_t *trace);

  void executeVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata, std::vector<reg_data_t> &rddata);

  void loadVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata);

  void storeVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata);

  void icache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_amo_reserve(uint64_t addr);

  bool dcache_amo_check(uint64_t addr);

  void writeToStdOut(const void* data, uint64_t addr, uint32_t size);

  void cout_flush();

  Word get_csr(uint32_t addr, uint32_t tid, uint32_t wid);

  void set_csr(uint32_t addr, Word value, uint32_t tid, uint32_t wid);

  uint32_t get_fpu_rm(uint32_t func3, uint32_t tid, uint32_t wid);

  void update_fcrs(uint32_t fflags, uint32_t tid, uint32_t wid);

  void trigger_ecall(); // Re-added for riscv-vector test functionality

  void trigger_ebreak(); // Re-added for riscv-vector test functionality

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