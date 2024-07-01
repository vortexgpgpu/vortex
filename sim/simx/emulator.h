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

  instr_trace_t* step();

  bool running() const;

  void suspend(uint32_t wid);

  void resume(uint32_t wid);

  bool barrier(uint32_t bar_id, uint32_t count, uint32_t wid);

  bool wspawn(uint32_t num_warps, Word nextPC);

  int get_exitcode() const;

private:

  struct ipdom_entry_t {
    ipdom_entry_t(const ThreadMask &tmask, Word PC);
    ipdom_entry_t(const ThreadMask &tmask);

    ThreadMask  tmask;
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
    std::stack<ipdom_entry_t>         ipdom_stack;
    Byte                              fcsr;
    uint32_t                          uuid;
  };

  struct wspawn_t {
    bool valid;
    uint32_t num_warps;
    Word nextPC;
  };

  std::shared_ptr<Instr> decode(uint32_t code) const;

  void execute(const Instr &instr, uint32_t wid, instr_trace_t *trace);

  void icache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_write(const void* data, uint64_t addr, uint32_t size);

  void dcache_amo_reserve(uint64_t addr);

  bool dcache_amo_check(uint64_t addr);

  void writeToStdOut(const void* data, uint64_t addr, uint32_t size);

  void cout_flush();

  Word get_csr(uint32_t addr, uint32_t tid, uint32_t wid);

  void set_csr(uint32_t addr, Word value, uint32_t tid, uint32_t wid);

  uint32_t get_fpu_rm(uint32_t func3, uint32_t tid, uint32_t wid);

  void update_fcrs(uint32_t fflags, uint32_t tid, uint32_t wid);

  const Arch& arch_;
  const DCRS& dcrs_;
  Core*       core_;
  std::vector<warp_t> warps_;
  WarpMask    active_warps_;
  WarpMask    stalled_warps_;
  std::vector<WarpMask> barriers_;
  std::unordered_map<int, std::stringstream> print_bufs_;
  MemoryUnit  mmu_;
  Word        csr_mscratch_;
  wspawn_t    wspawn_;
};

}

#endif