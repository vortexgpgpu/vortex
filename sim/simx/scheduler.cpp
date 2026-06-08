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

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <util.h>

#include "scheduler.h"
#include "instr_trace.h"
#include "instr.h"
#include "core.h"
#include "scoreboard.h"
#include "socket.h"
#include "cluster.h"
#include "processor_impl.h"
#include "local_mem.h"

using namespace vortex;

warp_t::warp_t(uint32_t num_threads)
  : tmask(num_threads)
  , PC(0)
  , uuid(0)
  , mscratch(0)
  , mscratch_tmask(num_threads)
  , cta_csrs()
{
}

void warp_t::reset() {
  this->tmask.reset();
  this->PC   = 0;
  this->uuid = 0;
  this->fcsr = 0;
  this->mstatus = 0;
  this->mtvec   = 0;
  this->mepc    = 0;
  this->mcause  = 0;
  this->mtval   = 0;
  this->mscratch_tmask.reset();
  // Register files live in OpcUnit and are reset there.
}

///////////////////////////////////////////////////////////////////////////////

Scheduler::Scheduler(const SimContext& ctx, const char* name, Core* core)
    : SimObject<Scheduler>(ctx, name)
    , core_(core)
    , warps_(VX_CFG_NUM_WARPS, VX_CFG_NUM_THREADS)
    , in_async_trap_(VX_CFG_NUM_WARPS, false)
    , trap_epoch_(VX_CFG_NUM_WARPS, 0)
    , last_mret_cycle_(VX_CFG_NUM_WARPS, 0)
    , async_trap_snapshot_(VX_CFG_NUM_WARPS)
    , ipdom_size_(VX_CFG_NUM_THREADS - 1)
{
  std::srand(50);

  // create child SimObjects (CTA dispatcher + barrier unit). Both are
  // registered with SimPlatform and get their own do_reset()/do_tick() calls.
  char sname[128];
  snprintf(sname, sizeof(sname), "%s-cta", name);
  cta_dispatcher_ = SimPlatform::instance().create_object<CtaDispatcher>(sname, core);
  snprintf(sname, sizeof(sname), "%s-barrier", name);
  barrier_unit_ = SimPlatform::instance().create_object<BarrierUnit>(sname, core, this);
}

Scheduler::~Scheduler() {}

void Scheduler::on_reset() {
  for (auto& warp : warps_) {
    warp.reset();
  }

  stalled_warps_.reset();
  stalled_warps_next_.reset();
  active_warps_.reset();
  std::fill(in_async_trap_.begin(), in_async_trap_.end(), false);
  std::fill(trap_epoch_.begin(), trap_epoch_.end(), 0);
  // Sequencers live on Core now; Core::on_reset() resets them.
  wspawn_.valid = false;

  // cta_dispatcher_ and barrier_unit_ are SimObjects — SimPlatform calls
  // their do_reset() directly.
}

void Scheduler::activate_warp(uint32_t wid, const cta_warp_record_t& rec) {
  auto& warp = warps_[wid];

  // Reusing a warp for the next CTA skips the one-time prologue and rewinds to
  // the kernel's per-CTA dispatch window — a fixed 20-byte (5-instruction)
  // sequence that reloads the entry pointer and kargs before re-calling.
  warp.PC       = rec.do_init ? rec.PC : (warp.PC - 20);
  warp.tmask    = rec.tmask;
  warp.mscratch = rec.mscratch;

  warp.cta_csrs.cta_id        = rec.cta_id;
  warp.cta_csrs.cta_rank      = rec.cta_rank;
  warp.cta_csrs.cta_size      = rec.cta_size;
  warp.cta_csrs.thread_idx[0] = rec.thread_idx[0];
  warp.cta_csrs.thread_idx[1] = rec.thread_idx[1];
  warp.cta_csrs.thread_idx[2] = rec.thread_idx[2];
  warp.cta_csrs.block_idx[0]  = rec.block_idx[0];
  warp.cta_csrs.block_idx[1]  = rec.block_idx[1];
  warp.cta_csrs.block_idx[2]  = rec.block_idx[2];
  warp.cta_csrs.block_dim[0]  = rec.block_dim[0];
  warp.cta_csrs.block_dim[1]  = rec.block_dim[1];
  warp.cta_csrs.block_dim[2]  = rec.block_dim[2];
  warp.cta_csrs.grid_dim[0]   = rec.grid_dim[0];
  warp.cta_csrs.grid_dim[1]   = rec.grid_dim[1];
  warp.cta_csrs.grid_dim[2]   = rec.grid_dim[2];
  warp.cta_csrs.entry         = rec.entry;
  warp.cta_csrs.lmem_addr     = rec.lmem_addr;
  warp.cta_csrs.cluster_size  = rec.cluster_size;

  while (!warp.ipdom_stack.empty()) warp.ipdom_stack.pop();

  active_warps_.set(wid);
  // CTA activation is not the registered suspend/resume path; clear both the
  // current and next stall state so the freshly-dispatched warp is immediately
  // schedulable (no spurious one-cycle stall from the registered state).
  stalled_warps_.reset(wid);
  stalled_warps_next_.reset(wid);

  DP(3, "*** dispatch CTA warp: cid=" << core_->id()
     << ", wid=" << wid << ", cta_id=" << warp.cta_csrs.cta_id
     << ", rank=" << warp.cta_csrs.cta_rank << "/" << warp.cta_csrs.cta_size
     << ", tmask=" << warp.tmask
     << ", PC=0x" << std::hex << warp.PC << std::dec
     << ", blockIdx=(" << warp.cta_csrs.block_idx[0] << "," << warp.cta_csrs.block_idx[1] << ")"
     << ", mscratch=0x" << std::hex << warp.mscratch << std::dec);
}

instr_trace_t* Scheduler::schedule(const WarpMask& warp_mask) {
  int scheduled_warp = -1;

  // Dispatch one CTA warp
  {
    uint32_t wid;
    cta_warp_record_t rec;
    if (cta_dispatcher_->step(active_warps_, &wid, &rec)) {
      activate_warp(wid, rec);
    }
  }

  // process pending wspawn when we are down to a single active warp
  if (wspawn_.valid && active_warps_.count() == 1) {
    DP(3, "*** Activate " << (wspawn_.num_warps-1) << " warps at PC: " << std::hex << wspawn_.nextPC << std::dec);
    auto spawning_mscratch = warps_.at(0).mscratch;
    for (uint32_t i = 1; i < wspawn_.num_warps; ++i) {
      auto& warp = warps_.at(i);
      warp.PC = wspawn_.nextPC;
      warp.tmask.set(0);
      warp.mscratch = spawning_mscratch;
      active_warps_.set(i);
      // wspawn activation, like CTA dispatch: immediate (both current + next).
      stalled_warps_.reset(i);
      stalled_warps_next_.reset(i);
      DT(3, core_->name() << " warp-state: wid=" << i << ", active=true, stalled=false, tmask=" << warp.tmask);
    }
    wspawn_.valid = false;
    this->resume(0);
  }

  // pick next ready warp
  for (size_t wid = 0, nw = VX_CFG_NUM_WARPS; wid < nw; ++wid) {
    if (active_warps_.test(wid) && !stalled_warps_.test(wid) && warp_mask.test(wid)) {
      scheduled_warp = wid;
      break;
    }
  }

  instr_trace_t* trace = nullptr;
  if (scheduled_warp != -1) {
    // get scheduled warp
    auto& warp = warps_.at(scheduled_warp);
    assert(warp.tmask.any());

    // Generate UUID
    uint64_t uuid = 0;
  #ifndef NDEBUG
    {
      uint32_t instr_id = warp.uuid++;
      uint32_t g_wid = core_->id() * VX_CFG_NUM_WARPS + scheduled_warp;
      uuid = (uint64_t(g_wid) << 32) | instr_id;
    }
  #endif

    // Allocate trace with header (fetch reads instruction word into trace->code,
    // decode fills in the rest of the metadata).
    trace = core_->trace_pool().allocate(1);
    new (trace) instr_trace_t(uuid);
    trace->cid    = core_->id();
    trace->wid    = scheduled_warp;
    trace->cta_id = warp.cta_csrs.cta_id;
    trace->PC     = warp.PC;
    trace->tmask  = warp.tmask;
    // PRISM RTU §6/§8.6: stamp trap_epoch so advance_pc can discard
    // a stale post-trap fetch (trap-epoch trailing the warp's current
    // trap_epoch_) without clobbering the trap-set mtvec.
    trace->trap_epoch = trap_epoch_.at(scheduled_warp);

    // PC is advanced at decode (+2 for RVC, +4 otherwise) — matches
    // RTL VX_scheduler updating warp_pcs on decode_sched_if.valid.
    // Branch/JAL/JALR commit later overrides warp.PC with the
    // resolved target.

    // Suspend warp until decode resumes it (non-stalling) or commit (stalling).
    this->suspend(scheduled_warp);
  }

  // Clock the registered warp-stall state. The suspend above, or a resume from
  // decode/commit/FU earlier this cycle, drives stalled_warps_next_ and only
  // becomes visible to the pick loop next cycle — so a warp released as its
  // instruction resolves is never re-scheduled the same cycle.
  stalled_warps_ = stalled_warps_next_;
  return trace;
}

bool Scheduler::running() const {
  return active_warps_.any() || cta_dispatcher_->running();
}

// suspend()/resume() drive the next-state; schedule() clocks it into the
// registered stalled_warps_ at the end of the cycle, so the change is observed
// only next cycle. Asserts check the next-state being mutated, not the
// registered value.
void Scheduler::suspend(uint32_t wid) {
  assert(active_warps_.test(wid));
  assert(!stalled_warps_next_.test(wid));
  stalled_warps_next_.set(wid);
  DT(3, core_->name() << " warp-state: wid=" << wid << ", stalled=true");
}

void Scheduler::resume(uint32_t wid) {
  assert(active_warps_.test(wid));
  assert(stalled_warps_next_.test(wid));
  stalled_warps_next_.reset(wid);
  DT(3, core_->name() << " warp-state: wid=" << wid << ", stalled=false");
}

void Scheduler::advance_pc(const instr_trace_t* trace, uint32_t inc) {
  // Drop stale post-trap fetches. A trace whose trap_epoch trails the
  // warp's current epoch was scheduled BEFORE the most recent async
  // trap; if we let it advance warp.PC now we'd step past the
  // trap-set mtvec and the dispatcher's first instruction would never
  // execute (the bug that broke the Phase 5 MISS test).
  if (trace->trap_epoch != trap_epoch_.at(trace->wid)) {
    return;
  }
  warps_.at(trace->wid).PC += inc;
}

bool Scheduler::setTmask(uint32_t wid, const ThreadMask& tmask) {
  auto& warp = warps_.at(wid);
  if (warp.tmask != tmask) {
    DT(3, core_->name() << " warp-state: wid=" << wid << ", tmask=" << tmask);
  }
  warp.tmask = tmask;
  // deactivate warp if no active threads
  if (!tmask.any()) {
    active_warps_.reset(wid);
    cta_dispatcher_->warp_done(wid);
    return false;
  }
  return true;
}

bool Scheduler::wspawn(uint32_t num_warps, Word nextPC) {
  num_warps = std::min<uint32_t>(num_warps, VX_CFG_NUM_WARPS);
  if (num_warps < 2 && active_warps_.count() == 1)
    return true; // nothing to do
  // schedule wspawn
  wspawn_.valid = true;
  wspawn_.num_warps = num_warps;
  wspawn_.nextPC = nextPC;
  return false;
}

// Barrier handling lives in BarrierUnit. See barrier_unit.{h,cpp}.

// RISC-V machine-mode synchronous exception cause codes (mcause).
// Standard 0..15; 24..31 reserved for custom by the privileged spec.
namespace {
  constexpr Word TRAP_CAUSE_BREAKPOINT   = 3;
  constexpr Word TRAP_CAUSE_ECALL_MMODE  = 11;
}

void Scheduler::raise_trap(uint32_t wid, Word cause, Word trap_pc) {
  auto& warp = warps_.at(wid);
  warp.mepc   = trap_pc;
  warp.mcause = cause;
  warp.mtval  = 0;
  warp.mscratch_tmask = warp.tmask;
  // Redirect to the handler. Low 2 bits of mtvec are the MODE field;
  // v1 supports direct mode only, so mask them off.
  warp.PC = warp.mtvec & ~Word(3);
  DT(3, core_->name() << " trap: wid=" << wid << ", cause=" << cause
     << ", mepc=0x" << std::hex << trap_pc << ", mtvec=0x" << warp.mtvec << std::dec);
}

void Scheduler::raise_async_trap(uint32_t wid, Word cause, Word trap_pc, const ThreadMask& new_tmask) {
  // Flush this warp's unissued instructions BEFORE the trap CSRs are
  // written, so the resume PC reflects the oldest flushed instruction
  // (where the warp will re-fetch on mret). Real RISC-V trap entry
  // flushes the pipeline for the trapping context; SimX needs the same
  // because otherwise ibuf_inflight stays pegged at IBUF_SIZE and the
  // post-trap fetch can't make progress.
  Word resume_pc = core_->flush_warp_pipeline(wid);
  if (resume_pc == 0) {
    // ibuffer was empty — use the caller's PC as-is.
    resume_pc = trap_pc;
  }
  this->raise_trap(wid, cause, resume_pc);
  auto& warp = warps_.at(wid);
  warp.tmask = new_tmask;
  in_async_trap_.at(wid) = true;
  // Re-activate the warp if a flushed wstall instruction had suspended it. A
  // macro-op (e.g. the WAIT2 hit-window GETWF/GETW, or TRACE2) sets fetch_stall,
  // which suspends the warp until that op commits; if the async trap flushes it
  // mid-flight it never commits, so its resume_warp never fires. The trap is
  // taking over the warp to run the dispatcher, so resume it here. Idempotent:
  // only resume if currently stalled.
  if (stalled_warps_next_.test(wid))
    this->resume(wid);
  // Lift the warp's outstanding scoreboard reservations (the parked
  // vx_rt_wait's rd) so the callback dispatcher can save/restore the full
  // register context without deadlocking on a reservation only its own
  // cb_ret can release. Re-installed at mret. (RTU callback-trap, §4.6.)
  async_trap_snapshot_.at(wid) = core_->scoreboard().snapshot_warp(wid);
  // Bump the per-warp trap epoch so any pre-trap fetch still in flight
  // (fetch_latch_ / pending icache rsp) can be detected at advance_pc
  // and discarded — its decoded trace.trap_epoch will be one behind.
  ++trap_epoch_.at(wid);
  DT(3, core_->name() << " async-trap: wid=" << wid
     << ", new_tmask=0x" << std::hex << warp.tmask.to_ulong() << std::dec
     << ", mepc=0x" << std::hex << warp.mepc << std::dec);
}

void Scheduler::mret(uint32_t wid) {
  auto& warp = warps_.at(wid);
  warp.PC    = warp.mepc;
  warp.tmask = warp.mscratch_tmask;
  // Re-install the reservations lifted at trap entry so the resumed
  // kernel's vx_rt_get_after still stalls until the ray's TERMINAL lands.
  // The matching TERMINAL writeback is held off until in_async_trap clears
  // (SfuUnit), so the dispatcher's epilogue restore can't clobber the
  // status word.
  core_->scoreboard().restore_warp(async_trap_snapshot_.at(wid));
  async_trap_snapshot_.at(wid).clear();
  in_async_trap_.at(wid) = false;
  last_mret_cycle_.at(wid) = SimPlatform::instance().cycles();
  DT(3, core_->name() << " mret: wid=" << wid
     << ", mepc=0x" << std::hex << warp.mepc << std::dec
     << ", restored tmask=0x" << std::hex << warp.tmask.to_ulong() << std::dec);
}

void Scheduler::trigger_ecall(uint32_t wid, Word trap_pc) {
  this->raise_trap(wid, TRAP_CAUSE_ECALL_MMODE, trap_pc);
}

void Scheduler::trigger_ebreak(uint32_t wid, Word trap_pc) {
  this->raise_trap(wid, TRAP_CAUSE_BREAKPOINT, trap_pc);
}
