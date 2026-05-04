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
  , cta_csrs()
{
}

void warp_t::reset() {
  this->tmask.reset();
  this->PC   = 0;
  this->uuid = 0;
  this->fcsr = 0;
  // Register files live in OpcUnit and are reset there.
}

///////////////////////////////////////////////////////////////////////////////

Scheduler::Scheduler(const SimContext& ctx, const char* name, Core* core)
    : SimObject<Scheduler>(ctx, name)
    , core_(core)
    , warps_(NUM_WARPS, NUM_THREADS)
    , ipdom_size_(NUM_THREADS - 1)
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
  active_warps_.reset();
  // Sequencers live on Core now; Core::on_reset() resets them.
  wspawn_.valid = false;

  // cta_dispatcher_ and barrier_unit_ are SimObjects — SimPlatform calls
  // their do_reset() directly.
}

void Scheduler::activate_warp(uint32_t wid, const cta_warp_record_t& rec) {
  auto& warp = warps_[wid];

  // if executing next CTA on same warp, we can skip prolog and jump to kernel_main at PC-12 (see vx_start.S)
  warp.PC       = rec.do_init ? rec.PC : (warp.PC - 12);
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
  warp.cta_csrs.lmem_addr     = rec.lmem_addr;

  while (!warp.ipdom_stack.empty()) warp.ipdom_stack.pop();

  active_warps_.set(wid);
  stalled_warps_.reset(wid);

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
      stalled_warps_.reset(i);
      DT(3, core_->name() << " warp-state: wid=" << i << ", active=true, stalled=false, tmask=" << warp.tmask);
    }
    wspawn_.valid = false;
    this->resume(0);
  }

  // pick next ready warp
  for (size_t wid = 0, nw = NUM_WARPS; wid < nw; ++wid) {
    if (active_warps_.test(wid) && !stalled_warps_.test(wid) && warp_mask.test(wid)) {
      scheduled_warp = wid;
      break;
    }
  }

  if (scheduled_warp == -1)
    return nullptr;

  // get scheduled warp
  auto& warp = warps_.at(scheduled_warp);
  assert(warp.tmask.any());

  // Generate UUID
  uint64_t uuid = 0;
#ifndef NDEBUG
  {
    uint32_t instr_id = warp.uuid++;
    uint32_t g_wid = core_->id() * NUM_WARPS + scheduled_warp;
    uuid = (uint64_t(g_wid) << 32) | instr_id;
  }
#endif

  // Allocate trace with header (fetch reads instruction word into trace->code,
  // decode fills in the rest of the metadata).
  auto trace = core_->trace_pool().allocate(1);
  new (trace) instr_trace_t(uuid);
  trace->cid   = core_->id();
  trace->wid   = scheduled_warp;
  trace->PC    = warp.PC;
  trace->tmask = warp.tmask;

  // PC is advanced at decode (matches RTL: VX_scheduler advances warp_pcs
  // on decode_sched_if.valid using is_rvc to pick +2 or +4). Branch/JAL/
  // JALR commit later overrides warp.PC with the resolved target.

  // Suspend warp until decode resumes it (non-stalling) or commit (stalling)
  this->suspend(scheduled_warp);

  return trace;
}

bool Scheduler::running() const {
  return active_warps_.any() || cta_dispatcher_->running();
}

void Scheduler::suspend(uint32_t wid) {
  assert(active_warps_.test(wid));
  assert(!stalled_warps_.test(wid));
  stalled_warps_.set(wid);
  DT(3, core_->name() << " warp-state: wid=" << wid << ", stalled=true");
}

void Scheduler::resume(uint32_t wid) {
  assert(active_warps_.test(wid));
  assert(stalled_warps_.test(wid));
  stalled_warps_.reset(wid);
  DT(3, core_->name() << " warp-state: wid=" << wid << ", stalled=false");
}

void Scheduler::advance_pc(uint32_t wid, uint32_t inc) {
  warps_.at(wid).PC += inc;
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
  num_warps = std::min<uint32_t>(num_warps, NUM_WARPS);
  if (num_warps < 2 && active_warps_.count() == 1)
    return true; // nothing to do
  // schedule wspawn
  wspawn_.valid = true;
  wspawn_.num_warps = num_warps;
  wspawn_.nextPC = nextPC;
  return false;
}

// Barrier handling lives in BarrierUnit. See barrier_unit.{h,cpp}.

// ecall/ebreak trap by deactivating all warps (used by riscv-vector tests)
void Scheduler::trigger_ecall() {
  active_warps_.reset();
}
void Scheduler::trigger_ebreak() {
  active_warps_.reset();
}
