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
#include <stack>
#include <simobject.h>
#include "types.h"
#include "instr.h"
#include "cta_dispatcher.h"
#include "barrier_unit.h"

namespace vortex {

class Core;
class Instr;
class instr_trace_t;

// IPDOM stack entry — one per nested SPLIT/JOIN.
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

// Per-CTA CSR snapshot (block/grid/thread indices, lmem base) populated at
// CTA dispatch and read by CSR reads in the warp.
struct cta_csrs_t {
  uint32_t cta_id;
  uint32_t cta_rank;
  uint32_t cta_size;
  uint32_t thread_idx[3];
  uint32_t block_idx[3];
  uint32_t block_dim[3];
  uint32_t grid_dim[3];
  uint64_t lmem_addr;

  cta_csrs_t()
    : cta_id(0)
    , cta_rank(0)
    , cta_size(0)
    , lmem_addr(0)
  {
    thread_idx[0] = thread_idx[1] = thread_idx[2] = 0;
    block_idx[0]  = block_idx[1]  = block_idx[2]  = 0;
    block_dim[0]  = block_dim[1]  = block_dim[2]  = 1;
    grid_dim[0]   = grid_dim[1]   = grid_dim[2]   = 1;
  }
};

struct warp_t {
  // Register files (ireg_file/freg_file) live in OpcUnit —
  // see operands.h / opc_unit.h for routing.
  std::stack<ipdom_entry_t>         ipdom_stack;
  ThreadMask                        tmask;
  Word                              PC;
  Byte                              fcsr;
  uint32_t                          uuid;

  // Per-warp MSCRATCH (holds kernel arg pointer, set at CTA dispatch)
  Word                              mscratch;

  // CTA CSR values set at dispatch time
  cta_csrs_t                        cta_csrs;

  warp_t(uint32_t num_threads);

  void reset();
};

// Per-core warp lifecycle owner: holds warp register state, barrier state,
// and the CTA dispatcher. All methods that read or mutate warp/barrier state
// live here.
class Scheduler : public SimObject<Scheduler> {
public:
  Scheduler(const SimContext& ctx, const char* name, Core* core);
  ~Scheduler();

  // ----- Warp lifecycle -----
  instr_trace_t* schedule(const WarpMask& warp_mask);
  void suspend(uint32_t wid);
  void resume(uint32_t wid);
  // Advance the warp's PC by `inc` bytes (called at decode with 2 or 4
  // depending on is_rvc; mirrors RTL warp_pcs update on decode_sched_if).
  void advance_pc(uint32_t wid, uint32_t inc);
  bool running() const;
  bool wspawn(uint32_t num_warps, Word nextPC);
  bool setTmask(uint32_t wid, const ThreadMask& tmask);

  // ----- Barriers -----
  // Barrier handling lives on BarrierUnit (a child SimObject of Scheduler).
  // Callers should reach it via `core_->scheduler().barrier_unit().X()`.
  BarrierUnit& barrier_unit() { return *barrier_unit_; }

  // CSR access lives on CsrUnit. FpuUnit reaches its fcsr helpers via
  // core_->csr_unit().

  // ----- Trap helpers -----
  void trigger_ecall();
  void trigger_ebreak();

  // ----- Accessors -----
  warp_t& warp(uint32_t wid) { return warps_.at(wid); }
  uint32_t ipdom_size() const { return ipdom_size_; }
  const auto& active_warps() const { return active_warps_; }
  const auto& stalled_warps() const { return stalled_warps_; }

protected:
  void on_reset();

private:
  struct wspawn_t {
    bool      valid;
    uint32_t  num_warps;
    Word      nextPC;
  };

  void activate_warp(uint32_t wid, const cta_warp_record_t& rec);

  Core* core_;

  CtaDispatcher::Ptr cta_dispatcher_;
  BarrierUnit::Ptr   barrier_unit_;

  std::vector<warp_t> warps_;
  WarpMask active_warps_;
  WarpMask stalled_warps_;
  uint32_t ipdom_size_;
  wspawn_t wspawn_;
  uint32_t mpm_class_;

  friend class SimObject<Scheduler>;
};

}
