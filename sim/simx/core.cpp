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
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "types.h"
#include "mem.h"
#include "core.h"
#include "socket.h"
#include "cluster.h"
#include "processor_impl.h"
#include "debug.h"
#include "constants.h"

using namespace vortex;

Core::Core(const SimContext& ctx,
           const char* name,
           uint32_t core_id,
           Socket* socket
           )
  : SimObject(ctx, name)
  , icache_req_out(1, this)
  , icache_rsp_in(1, this)
  , dcache_req_out(DCACHE_NUM_REQS, this)
  , dcache_rsp_in(DCACHE_NUM_REQS, this)
  , core_id_(core_id)
  , socket_(socket)
  , sequencers_(NUM_WARPS)
  , mpm_class_(0)
  , ibuffers_(NUM_WARPS)
  , operands_(ISSUE_WIDTH)
  , dispatchers_((uint32_t)FUType::Count)
  , func_units_((uint32_t)FUType::Count)
  , lmem_switch_(NUM_LSU_BLOCKS)
  , mem_coalescers_(NUM_LSU_BLOCKS)
  , fetch_latch_(ctx, "fetch_latch", 2, 2)
  , decode_latch_(ctx, "decode_latch", 1, 2)
  , pending_icache_(NUM_WARPS)
  , commit_arbs_(ISSUE_WIDTH)
  , ibuffer_arbs_(ISSUE_WIDTH, {ArbiterType::GTO, PER_ISSUE_WARPS})
  , fu_locked_(ISSUE_WIDTH, BitVector<>((uint32_t)FUType::Count, 0))
  , ibuf_inflight_(NUM_WARPS, 0)
{
  char sname[100];

  // create scheduler (SimObject) — also creates the CTA dispatcher.
  snprintf(sname, 100, "%s-scheduler", name);
  scheduler_ = SimPlatform::instance().create_object<Scheduler>(sname, this);

  // create decoder (SimObject)
  snprintf(sname, 100, "%s-decoder", name);
  decoder_ = SimPlatform::instance().create_object<Decoder>(sname, instr_pool_);

  // create scoreboard (SimObject)
  snprintf(sname, 100, "%s-scoreboard", name);
  scoreboard_ = SimPlatform::instance().create_object<Scoreboard>(sname);

  // create per-warp sequencers (SimObject)
  for (uint32_t w = 0; w < NUM_WARPS; ++w) {
    snprintf(sname, 100, "%s-sequencer%d", name, w);
    sequencers_.at(w) = SimPlatform::instance().create_object<Sequencer>(sname, this, instr_pool_);
  }

  // create operands
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    snprintf(sname, 100, "%s-operand%d", name, iw);
    operands_.at(iw) = Operands::Create(sname, this);
  }

  // create ibuffers
  for (uint32_t i = 0; i < ibuffers_.size(); ++i) {
    snprintf(sname, 100, "%s-ibuffer%d", name, i);
    ibuffers_.at(i) = TFifo<instr_trace_t*>::Create(sname, 1, IBUF_SIZE);
  }

  // create the memory coalescer
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    snprintf(sname, 100, "%s-coalescer%d", name, b);
    mem_coalescers_.at(b) = MemCoalescer::Create(sname, LSU_CHANNELS, DCACHE_CHANNELS, DCACHE_WORD_SIZE, LSUQ_OUT_SIZE, 1);
  }

  // create local memory
  snprintf(sname, 100, "%s-lmem", name);
  local_mem_ = LocalMem::Create(sname, LocalMem::Config{
    (1 << LMEM_LOG_SIZE),
    LSU_WORD_SIZE,
    LSU_NUM_REQS,
    log2ceil(LMEM_NUM_BANKS),
    false
  });

  // create lmem switch
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    snprintf(sname, 100, "%s-lmem_switch%d", name, b);
    lmem_switch_.at(b) = LocalMemSwitch::Create(sname, 1);
  }

  // create dcache adapter
  std::vector<LsuMemAdapter::Ptr> lsu_dcache_adapter(NUM_LSU_BLOCKS);
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    snprintf(sname, 100, "%s-lsu_dcache_adapter%d", name, b);
    lsu_dcache_adapter.at(b) = LsuMemAdapter::Create(sname, DCACHE_CHANNELS, 1);
  }

  // create per-block lmem adapters
  std::vector<LsuMemAdapter::Ptr> lsu_lmem_adapter(NUM_LSU_BLOCKS);
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    snprintf(sname, 100, "%s-lsu_lmem_adapter%d", name, b);
    lsu_lmem_adapter.at(b) = LsuMemAdapter::Create(sname, LSU_CHANNELS, 1);
  }

  // connect lmem switch to per-block adapters
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    lmem_switch_.at(b)->ReqOutLmem.bind(&lsu_lmem_adapter.at(b)->ReqIn);
    lsu_lmem_adapter.at(b)->RspOut.bind(&lmem_switch_.at(b)->RspInLmem);
  }

  // connect per-block lmem adapters to local memory ports
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    for (uint32_t c = 0; c < LSU_CHANNELS; ++c) {
      lsu_lmem_adapter.at(b)->ReqOut.at(c).bind(&local_mem_->Inputs.at(b * LSU_CHANNELS + c));
      local_mem_->Outputs.at(b * LSU_CHANNELS + c).bind(&lsu_lmem_adapter.at(b)->RspIn.at(c));
    }
  }

  if ((NUM_LSU_LANES > 1) && (DCACHE_WORD_SIZE > LSU_WORD_SIZE)) {
    // connect memory coalescer
    for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
      lmem_switch_.at(b)->ReqOutDC.bind(&mem_coalescers_.at(b)->ReqIn);
      mem_coalescers_.at(b)->RspOut.bind(&lmem_switch_.at(b)->RspInDC);

      mem_coalescers_.at(b)->ReqOut.bind(&lsu_dcache_adapter.at(b)->ReqIn);
      lsu_dcache_adapter.at(b)->RspOut.bind(&mem_coalescers_.at(b)->RspIn);
    }
  } else {
    // bypass memory coalescer
    for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
      lmem_switch_.at(b)->ReqOutDC.bind(&lsu_dcache_adapter.at(b)->ReqIn);
      lsu_dcache_adapter.at(b)->RspOut.bind(&lmem_switch_.at(b)->RspInDC);
    }
  }

  // connect dcache adapter
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    for (uint32_t c = 0; c < DCACHE_CHANNELS; ++c) {
      uint32_t p = b * DCACHE_CHANNELS + c;
      lsu_dcache_adapter.at(b)->ReqOut.at(c).bind(&this->dcache_req_out.at(p));
      this->dcache_rsp_in.at(p).bind(&lsu_dcache_adapter.at(b)->RspIn.at(c));
    }
  }

  // initialize dispatchers
  dispatchers_.at((int)FUType::ALU) = SimPlatform::instance().create_object<Dispatcher>(name, this, 2, NUM_ALU_BLOCKS, NUM_ALU_LANES);
  dispatchers_.at((int)FUType::FPU) = SimPlatform::instance().create_object<Dispatcher>(name, this, 2, NUM_FPU_BLOCKS, NUM_FPU_LANES);
  dispatchers_.at((int)FUType::LSU) = SimPlatform::instance().create_object<Dispatcher>(name, this, 2, NUM_LSU_BLOCKS, NUM_LSU_LANES);
  dispatchers_.at((int)FUType::SFU) = SimPlatform::instance().create_object<Dispatcher>(name, this, 2, NUM_SFU_BLOCKS, NUM_SFU_LANES);
  dispatchers_.at((int)FUType::CSR) = SimPlatform::instance().create_object<Dispatcher>(name, this, 2, NUM_SFU_BLOCKS, NUM_SFU_LANES);
#ifdef EXT_TCU_ENABLE
  dispatchers_.at((int)FUType::TCU) = SimPlatform::instance().create_object<Dispatcher>(name, this, 2, NUM_TCU_BLOCKS, NUM_TCU_LANES);
#endif

  // initialize execute units
  snprintf(sname, 100, "%s-alu", name);
  func_units_.at((int)FUType::ALU) = SimPlatform::instance().create_object<AluUnit>(sname, this);
  snprintf(sname, 100, "%s-fpu", name);
  func_units_.at((int)FUType::FPU) = SimPlatform::instance().create_object<FpuUnit>(sname, this);
  snprintf(sname, 100, "%s-lsu", name);
  func_units_.at((int)FUType::LSU) = SimPlatform::instance().create_object<LsuUnit>(sname, this);
  snprintf(sname, 100, "%s-sfu", name);
  func_units_.at((int)FUType::SFU) = SimPlatform::instance().create_object<SfuUnit>(sname, this);
  snprintf(sname, 100, "%s-csr", name);
  csr_unit_ = SimPlatform::instance().create_object<CsrUnit>(sname, this);
  func_units_.at((int)FUType::CSR) = csr_unit_;
#ifdef EXT_TCU_ENABLE
  snprintf(sname, 100, "%s-tcu", name);
  tensor_unit_ = SimPlatform::instance().create_object<TensorUnit>(sname, this);
  func_units_.at((int)FUType::TCU) = tensor_unit_;
#endif

  // bind commit arbiters
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    snprintf(sname, 100, "%s-commit-arb%d", name, iw);
    auto arbiter = TraceArbiter::Create(sname, ArbiterType::RoundRobin, (uint32_t)FUType::Count, 1);
    for (uint32_t fu = 0; fu < (uint32_t)FUType::Count; ++fu) {
      func_units_.at(fu)->Outputs.at(iw).bind(&arbiter->Inputs.at(fu));
    }
    commit_arbs_.at(iw) = arbiter;
  }

  this->on_reset();
}

Core::~Core() {
  //--
}

ProcessorImpl* Core::processor() const {
  return socket_->cluster()->processor();
}

void Core::on_reset() {
  for (auto& arb : ibuffer_arbs_) {
    arb.reset();
  }

  pending_instrs_.clear();
  pending_ifetches_ = 0;

  std::fill(ibuf_inflight_.begin(), ibuf_inflight_.end(), 0);

  perf_stats_ = PerfStats();
}

void Core::on_tick() {
  this->commit();
  this->execute();
  this->issue();
  this->decode();
  this->fetch();
  this->schedule();

  ++perf_stats_.cycles;
  DPN(2, std::flush);
}

void Core::schedule() {
  // profiling
  perf_stats_.active_warps += scheduler_->active_warps().count();
  perf_stats_.stalled_warps += scheduler_->stalled_warps().count();

  // stop when fetch_latch cannot accept
  if (fetch_latch_.full())
    return;

  // eligible warps: ibuffer not yet saturated
  WarpMask warp_mask;
  for (uint32_t w = 0, nw = NUM_WARPS; w < nw; ++w) {
    if (ibuf_inflight_.at(w) < IBUF_SIZE) {
      warp_mask.set(w);
    }
  }

  auto trace = scheduler_->schedule(warp_mask);
  if (trace == nullptr) {
    ++perf_stats_.sched_idle;
    return;
  }

  fetch_latch_.push(trace);
  DT(3, this->name() << "-pipeline schedule: " << *trace);
  pending_instrs_.push_back(trace);
  ++ibuf_inflight_.at(trace->wid);
  perf_stats_.issued_warps += 1;
  perf_stats_.issued_threads += trace->tmask.count();
}

void Core::fetch() {
  perf_stats_.ifetch_latency += pending_ifetches_;

  // handle icache response — extract instruction word from the TLM line payload
  auto& icache_rsp = icache_rsp_in.at(0);
  if (!icache_rsp.empty()){
    auto& mem_rsp = icache_rsp.peek();
    auto trace = pending_icache_.at(mem_rsp.tag);
    if (decode_latch_.try_push(trace)) {
      assert(mem_rsp.data && "icache response must carry line payload");
      uint32_t offset = trace->PC & (MEM_BLOCK_SIZE - 1);
      std::memcpy(&trace->code, mem_rsp.data->data() + offset, sizeof(uint32_t));
      DP(1, "Fetch: code=0x" << std::hex << trace->code << std::dec << ", cid=" << trace->cid
             << ", wid=" << trace->wid << ", tmask=" << trace->tmask
             << ", PC=0x" << std::hex << trace->PC << " (#" << std::dec << trace->uuid << ")");
      DT(3, this->name() << " icache-rsp: addr=0x" << std::hex << trace->PC << ", tag=0x" << mem_rsp.tag << std::dec << ", " << *trace);
      pending_icache_.release(mem_rsp.tag);
      icache_rsp.pop();
      --pending_ifetches_;
    }
  }

  // send icache request
  if (fetch_latch_.empty())
    return;
  auto trace = fetch_latch_.peek();

  // Avoid leaking icache tags when the request port back-pressures.
  if (pending_icache_.full()) {
    ++perf_stats_.fetch_stalls;
    return;
  }

  MemReq mem_req;
  mem_req.addr  = trace->PC;
  mem_req.write = false;
  uint32_t tag = pending_icache_.allocate(trace);
  mem_req.tag   = tag;
  mem_req.cid   = trace->cid;
  mem_req.uuid  = trace->uuid;
  if (this->icache_req_out.at(0).try_send(mem_req)) {
    DT(3, this->name() << " icache-req: addr=0x" << std::hex << mem_req.addr << ", tag=0x" << mem_req.tag << std::dec << ", " << *trace);
    fetch_latch_.pop();
    ++perf_stats_.ifetches;
    ++pending_ifetches_;
  } else {
    pending_icache_.release(tag);
    ++perf_stats_.fetch_stalls;
  }
}

void Core::decode() {
  if (decode_latch_.empty())
    return;

  auto trace = decode_latch_.peek();

  // check ibuffer capacity
  auto& ibuffer = ibuffers_.at(trace->wid);
  if (ibuffer->full()) {
    if (!trace->log_once(true)) {
      DT(4, this->name() << " ibuffer-stall: " << *trace);
    }
    ++perf_stats_.ibuf_stalls;
    return;
  } else {
    trace->log_once(false);
  }

  // Decode: fill trace metadata from instruction bits
  auto instr = decoder_->decode(trace->code, trace->uuid);
  DP(1, "Instr: " << *instr << " (#" << trace->uuid << ")");
  trace->instr_ptr   = instr;
  trace->fu_type     = instr->get_fu_type();
  trace->op_type     = instr->get_op_type();
  trace->dst_reg     = instr->get_dest_reg();
  trace->dst_bytesel = instr->get_dst_bytesel();
  for (uint32_t i = 0; i < NUM_SRC_REGS; ++i) {
    trace->src_regs[i] = instr->get_src_reg(i);
  }
  // Filter x0 for integer dest: writes to x0 are silent in RISC-V, so don't
  // reserve a scoreboard slot — the unit's writeback skips x0 too.
  auto dst = instr->get_dest_reg();
  trace->wb = (dst.type != RegType::None)
           && !(dst.type == RegType::Integer && dst.idx == 0);
  trace->fetch_stall = instr->is_wstall();

  // Resume warp for non-stalling instructions (ALU, FPU);
  // stalling instructions (LSU, SFU, TCU, branches) stay suspended until commit
  if (!trace->fetch_stall) {
    scheduler_->resume(trace->wid);
  }

  DT(3, this->name() << "-pipeline decode: " << *trace);

  // insert to ibuffer
  ibuffer->push(trace);

  decode_latch_.pop();
}

void Core::issue() {
  // dispatch operands
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    auto& operand = operands_.at(iw);
    if (operand->Output.empty())
      continue;
    auto trace = operand->Output.peek();
    if (dispatchers_.at((int)trace->fu_type)->Inputs.at(iw).try_send(trace)) {
      operand->Output.pop();
    }
  }

  // issue ibuffer instructions
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    bool any_scrb_blocked = false;
    BitVector<> ready_set(PER_ISSUE_WARPS);
    BitVector<> suppress_set(PER_ISSUE_WARPS);
    for (uint32_t w = 0; w < PER_ISSUE_WARPS; ++w) {
      uint32_t wid = w * ISSUE_WIDTH + iw;
      auto& ibuffer = ibuffers_.at(wid);
      if (ibuffer->empty())
        continue;

      // check scoreboard against sequencer micro-op (handles uop WAW/RAW)
      auto trace = ibuffer->peek();
      auto seq = sequencers_.at(wid);
      auto uop_trace = seq->get(trace);  // returns cached uop or generates next

      if (scoreboard_->in_use(uop_trace)) {
        auto uses = scoreboard_->get_uses(uop_trace);
        if (!uop_trace->log_once(true)) {
          DTH(4, "*** scoreboard-stall: dependents={");
          for (uint32_t j = 0, n = uses.size(); j < n; ++j) {
            auto& use = uses.at(j);
            __unused (use);
            if (j) DTN(4, ", ");
            DTN(4, use.reg_type << use.reg_id << " (#" << use.uuid << ")");
          }
          DTN(4, "}, " << *uop_trace << std::endl);
        }
        // Mirror RTL |(stg_valid_in & ~operands_ready): count once per cycle
        // per issue slice regardless of how many warps are blocked.
        any_scrb_blocked = true;
      } else {
        uop_trace->log_once(false);
        // FU lock: block warps whose target FU is locked by another warp.
        // fu_lock=1 means acquire request; blocked when FU already locked.
        auto fu = (int)uop_trace->fu_type;
        bool uop_fu_lock = uop_trace->instr_ptr->get_fu_lock();
        if (fu_locked_.at(iw).test(fu) && uop_fu_lock) {
          continue; // blocked by FU lock
        }
        ready_set.set(w); // mark instruction as ready
        // suppress warps whose target FU dispatcher input is full
        if (dispatchers_.at(fu)->Inputs.at(iw).full()) {
          suppress_set.set(w);
        }
      }
    }

    if (ready_set.any()) {
      // Only suppress when at least one warp can issue to a free FU;
      // otherwise let all warps through so the pipeline absorbs transient stalls.
      BitVector<> eff_suppress(PER_ISSUE_WARPS);
      auto unsuppressed = ready_set & ~suppress_set;
      if (unsuppressed.any()) {
        eff_suppress = suppress_set;
      }
      // select one instruction from ready set
      auto w = ibuffer_arbs_.at(iw).grant(ready_set, eff_suppress);
      uint32_t wid = w * ISSUE_WIDTH + iw;
      auto& ibuffer = ibuffers_.at(wid);
      auto trace = ibuffer->peek();

      // Retrieve the (already-generated) micro-op from the sequencer
      auto seq = sequencers_.at(wid);
      auto uop_trace = seq->get(trace);

      // to operand stage
      if (operands_.at(iw)->Input.try_send(uop_trace)) {
        // capture register operands at issue (Operands owns regfile)
        operands_.at(iw)->fetch_operands(uop_trace);
        DT(3, this->name() << "-pipeline issue: " << *uop_trace);
        if (uop_trace->wb) {
          // update scoreboard
          scoreboard_->reserve(uop_trace);
        }
        // Update FU lock state: 10=acquire, 01=release
        {
          auto fui = (int)uop_trace->fu_type;
          bool fl = uop_trace->instr_ptr->get_fu_lock();
          bool ful = uop_trace->instr_ptr->get_fu_unlock();
          if (fl && !ful) {
            fu_locked_.at(iw).set(fui);
          } else if (!fl && ful) {
            fu_locked_.at(iw).reset(fui);
          }
        }
        // Advance sequencer; pop ibuffer only when all micro-ops issued
        if (seq->advance()) {
          // Resume warp for macro instructions that stalled fetch at decode
          if (trace->instr_ptr->is_macro_op()) {
            scheduler_->resume(trace->wid);
            // Macro trace never reaches commit (only micro-ops do),
            // so remove it from pending tracking and deallocate here.
            pending_instrs_.remove(trace);
            trace->~instr_trace_t();
            trace_pool_.deallocate(trace, 1);
          }
          ibuffer->pop();
          // Release ibuffer backpressure slot (matches ++ in schedule()).
          assert(ibuf_inflight_.at(wid) > 0);
          --ibuf_inflight_.at(wid);
        }
      }
    }

    // track scoreboard stalls
    if (any_scrb_blocked) {
      ++perf_stats_.scrb_stalls;
    }
  }
}

void Core::execute() {
  for (uint32_t fu = 0; fu < (uint32_t)FUType::Count; ++fu) {
    auto& dispatch = dispatchers_.at(fu);
    auto& func_unit = func_units_.at(fu);
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      if (dispatch->Outputs.at(iw).empty())
        continue;
      auto trace = dispatch->Outputs.at(iw).peek();
      if (func_unit->Inputs.at(iw).try_send(trace)) {
        dispatch->Outputs.at(iw).pop();
      } else {
        // track functional unit stalls
        switch ((FUType)fu) {
        case FUType::ALU: ++perf_stats_.alu_stalls; break;
        case FUType::FPU: ++perf_stats_.fpu_stalls; break;
        case FUType::LSU: ++perf_stats_.lsu_stalls; break;
        case FUType::SFU: ++perf_stats_.sfu_stalls; break;
        case FUType::CSR: ++perf_stats_.csr_stalls; break;
      #ifdef EXT_TCU_ENABLE
        case FUType::TCU: ++perf_stats_.tcu_stalls; break;
      #endif
        default: assert(false);
        }
      }
    }
  }
}

void Core::commit() {
  // process completed instructions
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    auto& commit_arb = commit_arbs_.at(iw);
    if (commit_arb->Outputs.at(0).empty())
      continue;
    auto trace = commit_arb->Outputs.at(0).peek().data;

    // advance to commit stage
    DT(3, this->name() << "-pipeline commit: " << *trace);
    assert(trace->cid == core_id_);

    // update scoreboard
    if (trace->eop) {
      if (trace->wb) {
        operands_.at(iw)->writeback(trace);
        scoreboard_->release(trace);
      }

      // instruction mix profiling
      switch (trace->fu_type) {
      case FUType::ALU: ++perf_stats_.alu_instrs; break;
      case FUType::FPU: ++perf_stats_.fpu_instrs; break;
      case FUType::LSU: ++perf_stats_.lsu_instrs; break;
      case FUType::SFU: ++perf_stats_.sfu_instrs; break;
      case FUType::CSR: ++perf_stats_.csr_instrs; break;
    #ifdef EXT_TCU_ENABLE
      case FUType::TCU: ++perf_stats_.tcu_instrs; break;
    #endif
      default: assert(false);
      }
      // track committed instructions
      perf_stats_.instrs += 1;
      // Resume warp for FUs that lack explicit resume logic (e.g. LSU)
      if (trace->fetch_stall && trace->fu_type == FUType::LSU) {
        scheduler_->resume(trace->wid);
      }

      // instruction completed
      pending_instrs_.remove(trace);
    }

    // delete the trace
    trace->~instr_trace_t();
    trace_pool_.deallocate(trace, 1);

    commit_arb->Outputs.at(0).pop();
  }
}

int Core::get_exitcode() const {
  return operands_.at(0)->get_exit_code();
}

bool Core::running() const {
  if (scheduler_->running() || !pending_instrs_.empty()) {
    return true;
  }
  return false;
}

bool Core::has_pending_instrs(uint32_t wid) const {
  uint32_t count = 0;
  for (auto trace : pending_instrs_) {
    if (trace->wid == wid)
      ++count;
  }
  return count > 1; // more than 1 because the current instruction is also counted
}

void Core::resume(uint32_t wid) {
  scheduler_->resume(wid);
}

void Core::barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t wid, bool is_sync_bar) {
  scheduler_->barrier_unit().arrive(bar_id, count, wid, is_sync_bar);
}

bool Core::barrier_wait(uint32_t bar_id, uint32_t phase, uint32_t wid) {
  return scheduler_->barrier_unit().wait(bar_id, phase, wid);
}

void Core::global_barrier_resume(uint32_t bar_id) {
  scheduler_->barrier_unit().global_resume(bar_id);
}

void Core::barrier_event_attach(uint32_t bar_id) {
  scheduler_->barrier_unit().event_attach(bar_id);
}

void Core::barrier_event_release(uint32_t bar_id) {
  scheduler_->barrier_unit().event_release(bar_id);
}

bool Core::wspawn(uint32_t num_warps, Word nextPC) {
  return scheduler_->wspawn(num_warps, nextPC);
}

bool Core::setTmask(uint32_t wid, const ThreadMask& tmask) {
  return scheduler_->setTmask(wid, tmask);
}

int Core::dcr_write(uint32_t addr, uint32_t value) {
  __unused(addr);
  __unused(value);
  // KMU DCRs are handled at ProcessorImpl level and never reach here.
  return 0;
}

int Core::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  // tag arrives as (mpm_class << 6) | mpm_tag_idx after socket strips core_id
  switch (addr) {
  case VX_DCR_BASE_CACHE_FLUSH:
    *value = 0;
    break;
  case VX_DCR_BASE_MPM_VALUE: {
    uint32_t mpm_class   = tag >> 6;
    uint32_t mpm_tag_idx = tag & 0x3f;
    bool     is_hi       = (mpm_tag_idx >> 5) & 1;
    uint32_t idx         = mpm_tag_idx & 0x1f;
    uint32_t csr_addr    = is_hi ? (VX_CSR_MPM_BASE_H + idx) : (VX_CSR_MPM_BASE + idx);
    auto saved_class = mpm_class_;
    mpm_class_ = mpm_class;
    *value = static_cast<uint32_t>(csr_unit_->get_csr(csr_addr, 0, 0));
    mpm_class_ = saved_class;
    break;
  }
  default:
    break;
  }
  return 0;
}


Core::PerfStats& Core::perf_stats() {
  perf_stats_.opds_stalls = 0;
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    perf_stats_.opds_stalls += operands_.at(iw)->total_stalls();
  }
  return perf_stats_;
}

const Core::PerfStats& Core::perf_stats() const {
  perf_stats_.opds_stalls = 0;
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    perf_stats_.opds_stalls += operands_.at(iw)->total_stalls();
  }
  return perf_stats_;
}
