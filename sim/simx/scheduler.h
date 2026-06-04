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
  uint64_t entry;
  uint64_t lmem_addr;
  uint32_t cluster_size;

  cta_csrs_t()
    : cta_id(0)
    , cta_rank(0)
    , cta_size(0)
    , entry(0)
    , lmem_addr(0)
    , cluster_size(1)
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

  // Per-warp machine-mode trap CSRs. Synchronous ECALL/EBREAK traps
  // redirect the warp PC to mtvec and snapshot the faulting PC/cause
  // here; MRET restores the PC from mepc. See Scheduler::raise_trap.
  Word                              mstatus = 0;
  Word                              mtvec   = 0;
  Word                              mepc    = 0;
  Word                              mcause  = 0;
  Word                              mtval   = 0;
  // Saved active-thread mask. Snapshotted on raise_trap entry and
  // restored on mret. Phase-2 RTU callback narrows the running tmask
  // to only-yielded-lanes during the dispatcher; the pre-yield mask
  // lives here until mret restores it. (See proposal §4.6.)
  ThreadMask                        mscratch_tmask;

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
  // Pass the trace whose decode is firing so a stale post-trap fetch
  // (trace->trap_epoch trails the warp's current trap_epoch_) can be
  // discarded without clobbering the trap-set mtvec.
  void advance_pc(const instr_trace_t* trace, uint32_t inc);
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
  // Synchronous trap entry: snapshot the faulting PC into mepc, set
  // mcause, and redirect the warp PC to mtvec. trap_pc is the PC of the
  // faulting instruction (trace->PC), not the decode-advanced warp.PC.
  // Also snapshots the active tmask into mscratch_tmask.
  void raise_trap(uint32_t wid, Word cause, Word trap_pc);
  // Async trap entry: like raise_trap, but also narrows the running
  // tmask to `new_tmask` so the handler sees only the lanes that
  // actually need to run. Used by RtuCore to dispatch AHS/IS callbacks
  // on the subset of lanes whose rays yielded (proposal §4.6, option-c).
  // Caller must already have parked the warp at a suitable rendezvous
  // (e.g. vx_rt_wait); the caller is responsible for that constraint
  // because this method has no way to verify it.
  void raise_async_trap(uint32_t wid, Word cause, Word trap_pc, const ThreadMask& new_tmask);
  // Trap return: restore the warp PC from mepc and the tmask from
  // mscratch_tmask (MRET/SRET/URET).
  void mret(uint32_t wid);
  void trigger_ecall(uint32_t wid, Word trap_pc);
  void trigger_ebreak(uint32_t wid, Word trap_pc);

  // ----- Accessors -----
  warp_t& warp(uint32_t wid) { return warps_.at(wid); }
  uint32_t ipdom_size() const { return ipdom_size_; }
  const auto& active_warps() const { return active_warps_; }
  const auto& stalled_warps() const { return stalled_warps_; }
  // True while a warp is between async-trap entry and matching mret —
  // used by SfuUnit's RTU callback drain to serialize multiple CB_YIELDs
  // for the same warp (Phase 3-A2 divergent-SBT path).
  bool in_async_trap(uint32_t wid) const { return in_async_trap_.at(wid); }
  // Monotonic per-warp trap epoch. Bumped on every async-trap entry so
  // a stale post-trap fetch (pre-trap schedule whose icache rsp arrives
  // after flush_warp_pipeline) can be detected at advance_pc and
  // discarded instead of over-advancing warp.PC past the trap mtvec.
  uint32_t trap_epoch(uint32_t wid) const { return trap_epoch_.at(wid); }

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
  WarpMask stalled_warps_;       // registered (current) state read by schedule()
  WarpMask stalled_warps_next_;  // next-state written by suspend()/resume()
  // Per-warp gate set on async-trap entry, cleared on mret. Lets the
  // RTU callback drain decide when it is safe to fire a follow-on
  // CB_YIELD on the same warp.
  std::vector<bool> in_async_trap_;
  // Per-warp monotonic trap epoch; ++ on every raise_async_trap. Used
  // by advance_pc to discard stale post-fetch traces (see header).
  std::vector<uint32_t> trap_epoch_;
  // Per-warp scoreboard reservations lifted at async-trap entry and
  // re-installed at mret (RTU callback-trap; see Scoreboard::snapshot_warp).
  std::vector<std::vector<instr_trace_t*>> async_trap_snapshot_;
  uint32_t ipdom_size_;
  wspawn_t wspawn_;
  uint32_t mpm_class_;

  friend class SimObject<Scheduler>;
};

}
