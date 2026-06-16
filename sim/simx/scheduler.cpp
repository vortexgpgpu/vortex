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
#include "kmu/kmu.h"
#include "sfu_unit.h"

using namespace vortex;

warp_t::warp_t(uint32_t num_threads)
  : tmask(num_threads)
  , PC(0)
  , uuid(0)
  , scs_orig(num_threads)
  , scs_done(num_threads)
  , scs_parked(num_threads)
  , mscratch(0)
  , cta_csrs()
{
}

void warp_t::reset() {
  this->tmask.reset();
  // PC is excluded: a slot that has already run the startup resumes at its
  // dispatch window, which is derived from where the previous CTA left this
  // register, and a launch is not a device reset. It is zero-initialized at
  // construction, and every activation path assigns the PC before it is read.
  this->uuid = 0;
  this->fcsr = 0;
  this->mstatus = 0;
  this->mtvec   = 0;
  this->mepc    = 0;
  this->mcause  = 0;
  this->mtval   = 0;
  this->scs_stall = 0;
  this->scs_maxpc = 0;
  this->scs_orig.reset();
  this->scs_done.reset();
  this->scs_parked.reset();
  this->scs_cooldown = 0;
  this->scs_pending.clear();
  this->scs_runnable.clear();
  // Register files live in OpcUnit and are reset there.
}

///////////////////////////////////////////////////////////////////////////////

Scheduler::Scheduler(const SimContext& ctx, const char* name, Core* core)
    : SimObject<Scheduler>(ctx, name)
    , core_(core)
    , warps_(VX_CFG_NUM_WARPS, VX_CFG_NUM_THREADS)
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

#ifdef VX_CFG_EXT_RASTER_ENABLE
  fwd_is_fragment_.assign(VX_CFG_NUM_WARPS, false);
#endif
}

Scheduler::~Scheduler() {}

void Scheduler::on_reset() {
  for (auto& warp : warps_) {
    warp.reset();
  }

  stalled_warps_.reset();
  stalled_warps_next_.reset();
  active_warps_.reset();
  // Sequencers live on Core now; Core::on_reset() resets them.
  wspawn_.valid = false;

#ifdef VX_CFG_EXT_RASTER_ENABLE
  fwd_armed_   = false;
  fwd_drained_ = false;
  fwd_launched_ = 0;
  fwd_retired_  = 0;
  fwd_reqs_outstanding_ = 0;
  std::queue<FwdWave> empty;
  std::swap(fwd_waves_, empty);
  std::fill(fwd_is_fragment_.begin(), fwd_is_fragment_.end(), false);
#endif

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

  // SCS: clear per-warp forward-progress state when reusing a warp for a new CTA,
  // so stale parked/exited sets from the previous CTA do not leak in.
  warp.scs_stall = 0;
  warp.scs_maxpc = 0;
  warp.scs_orig.reset();
  warp.scs_done.reset();
  warp.scs_parked.reset();
  warp.scs_cooldown = 0;
  warp.scs_pending.clear();
  warp.scs_runnable.clear();

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

// SCS: number of no-progress issues before the warp switches subgroups. Small
// relative to VX_DBG_STALL_TIMEOUT so deadlocks are broken well before the
// pipeline's debug stall detector trips.
static constexpr uint32_t SCS_STALL_TIMEOUT = 64;

// SCS: cycles a purely-spinning warp backs off (skips scheduling) so it stops
// saturating shared dcache resources that a lock holder needs to make progress.
static constexpr uint32_t SCS_BACKOFF = 256;

// SCS: snapshot the warp's current running subgroup as a schedulable split. The
// IPDOM stack is moved out (the running slot is about to hold a different
// subgroup), giving the captured subgroup its own private reconvergence nesting.
scs_split_t Scheduler::scs_capture_current(warp_t& warp) {
  scs_split_t s(warp.tmask, warp.PC, warp.ipdom_stack);
  while (!warp.ipdom_stack.empty()) warp.ipdom_stack.pop();
  return s;
}

// SCS: install a schedulable split as the running subgroup — its mask, PC and
// reconvergence nesting all become the warp's live state.
void Scheduler::scs_install(warp_t& warp, scs_split_t&& split) {
  warp.tmask        = split.tmask;
  warp.PC           = split.pc;
  warp.ipdom_stack  = std::move(split.ipdom);
  warp.scs_maxpc    = warp.PC;
  warp.scs_stall    = 0;
  warp.scs_parked  &= ~split.tmask;  // these lanes are now running, not parked
}

// SCS: the spinning subgroup yields to another runnable subgroup so a parked
// one (e.g. a lock holder) can make progress and release its resource.
//
// Parked subgroups live in scs_runnable and are scheduled strictly round-robin.
// This fairness is essential: a deferred lock holder must eventually run even
// while other subgroups keep spinning, otherwise a holder stuck behind the
// queue can deadlock another warp waiting on its lock. Each subgroup carries its
// own resume PC and reconvergence snapshot, so subgroups parked at distinct
// program points (e.g. several different blocking loops) never conflate.
//
// This multi-path escape only triggers under stall, so normal code (which never
// trips the watchdog) is unaffected.

// Commit any pending parked subgroup into the runnable pool. A pending subgroup
// is one set aside at a reconvergence point (IPDOM `passed`), or masked off by
// vx_pred in a (blocking) loop; committing it makes it round-robin schedulable.
void Scheduler::scs_fold_parked(warp_t& warp) {
  if (!warp.ipdom_stack.empty() && warp.ipdom_stack.top().has_passed) {
    auto entry = warp.ipdom_stack.top();
    warp.ipdom_stack.pop();
    // The passed subgroup reconverged past this level; it carries the remaining
    // (shallower) nesting as its private stack.
    warp.scs_runnable.emplace_back(entry.passed_tmask, entry.passed_pc, warp.ipdom_stack);
    warp.scs_parked |= entry.passed_tmask;
  }
  // Move all cancellable pending maskoff subgroups into the runnable pool; their
  // lanes are now committed to a parked subgroup (no longer grabbable by a
  // restore in the current subgroup).
  for (auto& p : warp.scs_pending) {
    warp.scs_parked |= p.tmask;
    warp.scs_runnable.push_back(std::move(p));
  }
  warp.scs_pending.clear();
}

// Pull the next runnable subgroup with live (non-exited) lanes into the running
// slot; drop stale entries whose lanes have all exited.
bool Scheduler::scs_resume_next(warp_t& warp) {
  while (!warp.scs_runnable.empty()) {
    scs_split_t next = std::move(warp.scs_runnable.front());
    warp.scs_runnable.erase(warp.scs_runnable.begin());
    ThreadMask live = next.tmask & ~warp.scs_done;
    warp.scs_parked &= ~next.tmask;
    if (live.any()) {
      next.tmask = live;
      this->scs_install(warp, std::move(next));
      return true;
    }
  }
  return false;
}

bool Scheduler::scs_rotate(warp_t& warp) {
  this->scs_fold_parked(warp);
  if (warp.scs_runnable.empty())
    return false; // nothing to switch to (pure cross-warp spin)

  // Round-robin: defer the spinning group (snapshotting its own reconvergence
  // nesting), then resume the oldest queued subgroup with live lanes.
  warp.scs_runnable.push_back(this->scs_capture_current(warp));
  warp.scs_parked |= warp.tmask;
  if (this->scs_resume_next(warp)) {
    DT(3, core_->name() << " SCS-rotate: tmask=" << warp.tmask << ", PC=0x" << std::hex << warp.PC << std::dec);
    return true;
  }
  return false;
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

#ifdef VX_CFG_EXT_RASTER_ENABLE
  // Inject ready fragment waves into free warp slots.
  if (fwd_armed_)
    fwd_try_inject();
#endif

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

  // SCS spin back-off: tick down cooldowns of all schedulable warps so a warp
  // that is purely spinning (waiting on a lock held by another warp) stops
  // hammering shared memory, freeing dcache resources for the holder to release.
  for (size_t wid = 0, nw = VX_CFG_NUM_WARPS; wid < nw; ++wid) {
    if (active_warps_.test(wid) && !stalled_warps_.test(wid) && warps_.at(wid).scs_cooldown > 0)
      --warps_.at(wid).scs_cooldown;
  }

  // pick next ready warp (skip warps in spin back-off)
  for (size_t wid = 0, nw = VX_CFG_NUM_WARPS; wid < nw; ++wid) {
    if (active_warps_.test(wid) && !stalled_warps_.test(wid) && warp_mask.test(wid)
     && warps_.at(wid).scs_cooldown == 0) {
      scheduled_warp = wid;
      break;
    }
  }

  instr_trace_t* trace = nullptr;
  if (scheduled_warp != -1) {
    // get scheduled warp
    auto& warp = warps_.at(scheduled_warp);
    assert(warp.tmask.any());

    // SCS forward-progress watchdog. A warp that keeps issuing without reaching
    // new code (PC never exceeds its max) is spinning. Once it has parked work
    // (deferred subgroups, or a subgroup masked off in a spin loop), it must
    // time-slice unconditionally: a lane that merely *appears* to advance (e.g.
    // acquires one lock then spins on the next) would otherwise keep resetting
    // the watchdog and starve a deferred lock holder that another warp is
    // waiting on. With no parked work, reset on genuine forward progress so
    // normal code never rotates.
    bool has_parked = !warp.scs_runnable.empty() || !warp.scs_pending.empty()
                   || (!warp.ipdom_stack.empty() && warp.ipdom_stack.top().has_passed);
    if (!has_parked && warp.PC > warp.scs_maxpc) {
      warp.scs_maxpc = warp.PC;
      warp.scs_stall = 0;
    } else if (++warp.scs_stall >= SCS_STALL_TIMEOUT) {
      warp.scs_stall = 0;
      if (!this->scs_rotate(warp)) {
        // Pure cross-warp spin (nothing of our own to run): back off so the
        // warp stops saturating shared dcache resources, letting the lock
        // holder in another warp get serviced and release.
        warp.scs_cooldown = SCS_BACKOFF;
      }
    }

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

    // PC is advanced at decode (+2 for RVC, +4 otherwise) — matches
    // the hardware warp-PC update at decode.
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
  return active_warps_.any() || cta_dispatcher_->running()
#ifdef VX_CFG_EXT_RASTER_ENABLE
    || fwd_armed_
#endif
    ;
}

bool Scheduler::yield_warp(uint32_t wid) {
  auto& warp = warps_.at(wid);
  // PC was already advanced past the yield at decode, so scs_rotate captures the
  // yielding split at its post-yield resume point and switches to a sibling.
  // If nothing is runnable the warp simply continues — yield is a no-op.
  this->scs_rotate(warp);
  return true;
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
  warps_.at(trace->wid).PC += inc;
}

bool Scheduler::setTmask(uint32_t wid, const ThreadMask& tmask_in) {
  auto& warp = warps_.at(wid);
  // SCS: a subgroup resumed early (e.g. lock holder) may have already exited the
  // kernel; never let a later mask-restore (e.g. vx_pred fallback) reactivate
  // those exited lanes with stale state (scs_done). Likewise, a mask restore
  // (JOIN orig_tmask / PRED rs2 fallback) must never reabsorb lanes that are
  // running in another schedulable subgroup (scs_parked) — that conflation is
  // exactly the multi-loop corruption this design eliminates. Exited lanes are
  // recorded by TMC only; parked lanes by the fold/rotate path.
  ThreadMask tmask = tmask_in & ~warp.scs_done & ~warp.scs_parked;
  if (warp.tmask != tmask) {
    DT(3, core_->name() << " warp-state: wid=" << wid << ", tmask=" << tmask);
  }
  warp.tmask = tmask;
  warp.scs_orig = warp.scs_orig | tmask;  // SCS: track full warp participation
  // deactivate warp if no active threads
  if (!tmask.any()) {
    // SCS: a finished subgroup yields to a deferred runnable subgroup (if any)
    // rather than retiring the warp, so deferred work still completes.
    // Fold any still-parked subgroup (e.g. lock holders masked off by vx_pred)
    // into the pool so a finishing lane never strands them — otherwise their
    // held locks would never be released and other warps would deadlock.
    this->scs_fold_parked(warp);
    // Resume the next deferred subgroup that still has live (non-exited) lanes.
    if (this->scs_resume_next(warp)) {
      DT(3, core_->name() << " SCS-resume(subgroup-done): wid=" << wid << ", tmask=" << warp.tmask << ", PC=0x" << std::hex << warp.PC << std::dec);
      return true;
    }
    active_warps_.reset(wid);
#ifdef VX_CFG_EXT_RASTER_ENABLE
    // An injected fragment wave retiring closes one FWD epoch slot.
    if (fwd_is_fragment_[wid]) {
      fwd_is_fragment_[wid] = false;
      ++fwd_retired_;
    }
#endif
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
  // Redirect to the handler. Low 2 bits of mtvec are the MODE field;
  // v1 supports direct mode only, so mask them off.
  warp.PC = warp.mtvec & ~Word(3);
  DT(3, core_->name() << " trap: wid=" << wid << ", cause=" << cause
     << ", mepc=0x" << std::hex << trap_pc << ", mtvec=0x" << warp.mtvec << std::dec);
}

void Scheduler::mret(uint32_t wid) {
  auto& warp = warps_.at(wid);
  warp.PC    = warp.mepc;
  DT(3, core_->name() << " mret: wid=" << wid
     << ", mepc=0x" << std::hex << warp.mepc << std::dec);
}

void Scheduler::trigger_ecall(uint32_t wid, Word trap_pc) {
  this->raise_trap(wid, TRAP_CAUSE_ECALL_MMODE, trap_pc);
}

void Scheduler::trigger_ebreak(uint32_t wid, Word trap_pc) {
  this->raise_trap(wid, TRAP_CAUSE_BREAKPOINT, trap_pc);
}

#ifdef VX_CFG_EXT_RASTER_ENABLE
///////////////////////////////////////////////////////////////////////////////
// Fragment Work Distributor (FWD).
///////////////////////////////////////////////////////////////////////////////

// Per-warp LMEM band stride. The payload itself is seeded into the register
// window (FWD-5), so the FS reads no LMEM; this only gives each injected warp a
// distinct lmem_addr base (the FS declares no LMEM of its own).
static constexpr uint32_t kFwdPayloadStride =
  VX_CFG_NUM_THREADS * uint32_t(sizeof(graphics::frag_payload_t));

void Scheduler::fwd_arm(Word frag_entry, Word frag_param) {
  fwd_armed_      = true;
  fwd_drained_    = false;
  fwd_frag_entry_ = frag_entry;
  fwd_frag_param_ = frag_param;
  fwd_launched_   = 0;
  fwd_retired_    = 0;
  fwd_reqs_outstanding_ = 0;
}

bool Scheduler::fwd_wave_queue_full() const {
  return fwd_waves_.size() >= VX_CFG_NUM_WARPS;
}

void Scheduler::fwd_push_wave(const FwdWave& wave) {
  fwd_waves_.push(wave);
}

bool Scheduler::fwd_can_request() const {
  // Bound in-flight work to ~NUM_WARPS waves (queued + outstanding requests).
  return !fwd_drained_
      && (fwd_waves_.size() + fwd_reqs_outstanding_) < VX_CFG_NUM_WARPS;
}

bool Scheduler::fwd_done() const {
  return fwd_armed_ && fwd_drained_ && fwd_waves_.empty()
      && (fwd_reqs_outstanding_ == 0)
      && (fwd_launched_ == fwd_retired_);
}

void Scheduler::fwd_disarm() {
  fwd_armed_ = false;
  core_->fwd_done_out.send({core_->id()});
}

void Scheduler::fwd_try_inject() {
  // A fresh warp slot begins at the program image base (where __vx_cta_entry is
  // linked), exactly as a KMU-launched CTA does; the per-CTA dispatch window
  // reads VX_CSR_CTA_ENTRY (= rec.entry, the FS function) and VX_CSR_MSCRATCH
  // (= rec.mscratch, the FS args) and calls into the shader. The image base
  // is the KMU's startup PC (set by the grid-less draw kick, persists across
  // SimPlatform reset); the FS entry/arg come from the RASTER_FRAG_* descriptor.
  // A slot that already ran the startup for these lanes rewinds into the
  // dispatch window instead -- see activate_warp.
  const Word startup_pc =
    Word(core_->socket()->cluster()->processor()->kmu().startup_pc());

  while (!fwd_waves_.empty()) {
    // Find any free warp slot — there is no driver warp to skip in the push model.
    int wid = -1;
    for (uint32_t w = 0; w < VX_CFG_NUM_WARPS; ++w) {
      if (!active_warps_.test(w)) { wid = int(w); break; }
    }
    if (wid < 0) break;  // no free slot this cycle

    const FwdWave& wave = fwd_waves_.front();

    cta_warp_record_t rec;  // ThreadMask member needs sizing; assigned below
    // A slot that has already run the startup for these lanes rewinds into the
    // dispatch window instead of re-entering at the image base, exactly as a
    // compute CTA does. The state is the dispatcher's because a slot is shared
    // between the two launch paths.
    rec.do_init  = cta_dispatcher_->slot_needs_init(uint32_t(wid), wave.tmask);
    rec.PC       = startup_pc;                                         // image base (__vx_cta_entry)
    rec.entry    = fwd_frag_entry_;                                    // FS function entry (CTA_ENTRY)
    rec.mscratch = fwd_frag_param_;                                    // FS args pointer
    rec.param    = fwd_frag_param_;
    rec.cta_id   = 0;
    rec.cta_rank = 0;
    rec.cta_size = 1;
    rec.thread_idx[0] = rec.thread_idx[1] = rec.thread_idx[2] = 0;
    // A fragment warp is not a CTA: it has no block index, and its "block" is
    // exactly the lanes the packer filled — four per packed quad, so an empty quad
    // slot costs no threads.
    rec.block_idx[0]  = rec.block_idx[1]  = rec.block_idx[2]  = 0;
    rec.block_dim[0]  = uint32_t(wave.tmask.count());
    rec.block_dim[1]  = rec.block_dim[2]  = 1;
    rec.grid_dim[0]   = rec.grid_dim[1]   = rec.grid_dim[2]   = 1;
    rec.lmem_addr     = uint64_t(VX_MEM_LMEM_BASE_ADDR) + uint64_t(wid) * kFwdPayloadStride;
    rec.cluster_size  = 1;
    rec.tmask         = wave.tmask;

    activate_warp(uint32_t(wid), rec);
    cta_dispatcher_->mark_slot_inited(uint32_t(wid), wave.tmask);

    // The stamp arrives with the launch: land it in this warp's launch registers
    // (read back as the FRAG_* CSRs). No window tenancy, no slot, no window op.
    warps_.at(wid).frag = wave.payload;

    fwd_is_fragment_[wid] = true;
    ++fwd_launched_;
    fwd_waves_.pop();
  }
}
#endif // VX_CFG_EXT_RASTER_ENABLE
