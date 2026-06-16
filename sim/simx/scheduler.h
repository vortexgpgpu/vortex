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
#include <queue>
#include <array>
#include "types.h"
#include "instr.h"
#include "cta_dispatcher.h"
#include "barrier_unit.h"
#ifdef VX_CFG_EXT_RASTER_ENABLE
#include <vx_gfx_abi.h>
#endif

namespace vortex {

class Core;
class Instr;
class instr_trace_t;

// IPDOM stack entry — one per nested SPLIT/JOIN.
//
// SCS (Schedulable Convergence Stack) extension: on the first JOIN the arriving
// subgroup's *forward* PC (post-join) is captured here (passed_tmask/passed_pc).
// Normally that subgroup is restored at reconvergence (second JOIN); but if the
// sibling subgroup spins (a SIMT deadlock), the stall watchdog resumes the
// captured subgroup forward instead — breaking the deadlock without per-thread
// PCs. See Scheduler::scs_rotate().
struct ipdom_entry_t {
  ThreadMask  orig_tmask;
  Word        else_PC;
  bool        fallthrough;

  ThreadMask  passed_tmask;   // subgroup that reached the join first
  Word        passed_pc;      // its forward (post-join) PC
  bool        has_passed;

  ipdom_entry_t(const ThreadMask &tmask, Word PC)
    : orig_tmask  (tmask)
    , else_PC     (PC)
    , fallthrough (false)
    , passed_tmask(tmask)
    , passed_pc   (0)
    , has_passed  (false)
  {}
};

// SCS: an independently-schedulable divergent subgroup. Beyond a resume
// {mask, PC} it carries its OWN reconvergence nesting (a snapshot of the IPDOM
// stack), so two subgroups parked at different program points never share or
// corrupt each other's reconvergence state — the defect that made a single
// {mask, PC} side-channel conflate distinct blocking loops. The pool is
// conceptually unbounded; real hardware caps the on-chip split table at K and
// spills the remainder to a warp-private memory region. SimX models the
// unbounded logical set directly with a vector (= the spilled pool), so the
// bounded-K spill case is exercised without a separate spill model.
struct scs_split_t {
  ThreadMask                 tmask;
  Word                       pc;
  std::stack<ipdom_entry_t>  ipdom;   // per-subgroup reconvergence nesting

  scs_split_t(const ThreadMask& m, Word p)
    : tmask(m), pc(p) {}
  scs_split_t(const ThreadMask& m, Word p, const std::stack<ipdom_entry_t>& s)
    : tmask(m), pc(p), ipdom(s) {}
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

  // SCS forward-progress state.
  uint32_t                          scs_stall = 0;   // no-progress watchdog counter
  Word                              scs_maxpc = 0;   // furthest PC reached (progress probe)
  ThreadMask                        scs_orig;        // union of all masks seen (full participation)
  ThreadMask                        scs_done;        // lanes that have permanently exited the kernel
  ThreadMask                        scs_parked;      // lanes held in non-current subgroups (pending+runnable)
  uint32_t                          scs_cooldown = 0;// spin back-off: cycles to skip scheduling
  // A vx_pred that masks off lanes in a (possibly blocking) loop records them
  // here as a DISTINCT pending subgroup with its own resume PC and reconvergence
  // snapshot. Pending subgroups are cancelled if the loop reconverges normally
  // (pred empties), or committed to scs_runnable on a forward-progress stall.
  std::vector<scs_split_t>          scs_pending;     // maskoff subgroups awaiting fold (cancellable)
  std::vector<scs_split_t>          scs_runnable;    // committed runnable subgroups (round-robin pool)

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

  // CTA CSR values set at dispatch time
  cta_csrs_t                        cta_csrs;

#ifdef VX_CFG_EXT_RASTER_ENABLE
  // Per-lane fragment stamp, delivered WITH the launch (the raster engine packs
  // it into the launch message) and read back as the FRAG_* CSRs. The shader no
  // longer fetches its own stamp out of a window.
  std::array<graphics::frag_payload_t, VX_CFG_NUM_THREADS> frag;
#endif

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
  // depending on is_rvc; mirrors the hardware warp-PC update at decode).
  void advance_pc(const instr_trace_t* trace, uint32_t inc);
  bool running() const;
  bool wspawn(uint32_t num_warps, Word nextPC);
  bool setTmask(uint32_t wid, const ThreadMask& tmask);

  CtaDispatcher* cta_dispatcher() const { return cta_dispatcher_.get(); }

#ifdef VX_CFG_EXT_RASTER_ENABLE
  // ----- Fragment Work Distributor (FWD) -----
  // The per-core FWD turns rasterized quad-waves into launched fragment warps.
  // SfuUnit owns the raster-bus I/O and feeds ready waves here; injection
  // (activate_warp + LMEM payload seed) and epoch accounting live in the
  // scheduler because they touch warp lifecycle.
  struct FwdWave {
    ThreadMask tmask;
    std::array<graphics::frag_payload_t, VX_CFG_NUM_THREADS> payload;
    FwdWave() : tmask(VX_CFG_NUM_THREADS) {}
  };
  // Arm from the RASTER fragment-dispatch descriptor (RASTER_FRAG_* DCRs):
  // remember the FS entry PC and arg pointer the distributor launches each
  // fragment warp with. There is no driver warp in the push model — the raster
  // engine launches fragment warps directly.
  void fwd_arm(Word frag_entry, Word frag_param);
  bool fwd_armed() const { return fwd_armed_; }
  // Ready-wave queue admission (bounded so SfuUnit paces RasterReqs).
  bool fwd_wave_queue_full() const;
  void fwd_push_wave(const FwdWave& wave);
  void fwd_mark_drained() { fwd_drained_ = true; }
  // Outstanding-RasterReq budget so SfuUnit doesn't over-request.
  bool fwd_can_request() const;
  void fwd_on_request()  { ++fwd_reqs_outstanding_; }
  void fwd_on_response() { if (fwd_reqs_outstanding_) --fwd_reqs_outstanding_; }
  // Epoch complete: producer drained AND every launched wave retired.
  bool fwd_done() const;
  void fwd_disarm();
#endif

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
  void raise_trap(uint32_t wid, Word cause, Word trap_pc);
  // Trap return: restore the warp PC from mepc (MRET/SRET/URET).
  void mret(uint32_t wid);
  void trigger_ecall(uint32_t wid, Word trap_pc);
  void trigger_ebreak(uint32_t wid, Word trap_pc);

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

#ifdef VX_CFG_EXT_RASTER_ENABLE
  // Inject as many ready fragment waves as there are free warp slots (called
  // each cycle from schedule()).
  void fwd_try_inject();
#endif

  // SCS: snapshot the warp's current running subgroup (mask, PC, reconvergence
  // nesting) as a schedulable split, and install a split as the running one.
  scs_split_t scs_capture_current(warp_t& warp);
  void        scs_install(warp_t& warp, scs_split_t&& split);

  // SCS: commit any pending parked subgroup (IPDOM-passed, or vx_pred-masked
  // spin-loop acquirers) from the cancellable pending list into the runnable
  // pool with its own resume PC and reconvergence snapshot.
  void scs_fold_parked(warp_t& warp);

  // SCS: on a stall, switch the warp to a different runnable subgroup (a parked
  // post-join group, or a deferred one), deferring the spinning group. Returns
  // true if it switched; false if there was nothing to switch to (pure spin).
  bool scs_rotate(warp_t& warp);

  // SCS: pull the next runnable subgroup with live (non-exited) lanes into the
  // running slot; drops stale all-exited entries. Returns true on success.
  bool scs_resume_next(warp_t& warp);

  Core* core_;

  CtaDispatcher::Ptr cta_dispatcher_;
  BarrierUnit::Ptr   barrier_unit_;

  std::vector<warp_t> warps_;
  WarpMask active_warps_;
  WarpMask stalled_warps_;       // registered (current) state read by schedule()
  WarpMask stalled_warps_next_;  // next-state written by suspend()/resume()
  uint32_t ipdom_size_;
  wspawn_t wspawn_;
  uint32_t mpm_class_;

#ifdef VX_CFG_EXT_RASTER_ENABLE
  // Fragment Work Distributor state (per core).
  bool                fwd_armed_            = false;
  bool                fwd_drained_          = false;
  Word                fwd_frag_entry_       = 0;
  Word                fwd_frag_param_       = 0;
  uint64_t            fwd_launched_         = 0;
  uint64_t            fwd_retired_          = 0;
  uint32_t            fwd_reqs_outstanding_ = 0;
  std::queue<FwdWave> fwd_waves_;
  std::vector<bool>   fwd_is_fragment_;     // per-wid: this warp is an injected fragment wave
#endif

  friend class SimObject<Scheduler>;
};

}
