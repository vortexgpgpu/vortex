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
#include "arch.h"
#include "mem.h"
#include "core.h"
#include "debug.h"
#include "constants.h"

using namespace vortex;

Core::Core(const SimContext& ctx,
           const char* name,
           uint32_t core_id,
           Socket* socket,
           const Arch &arch,
           const DCRS &dcrs
           )
  : SimObject(ctx, name)
  , icache_req_out(1, this)
  , icache_rsp_in(1, this)
  , dcache_req_out(DCACHE_NUM_REQS, this)
  , dcache_rsp_in(DCACHE_NUM_REQS, this)
  , core_id_(core_id)
  , socket_(socket)
  , arch_(arch)
  , emulator_(arch, dcrs, this)
  , ibuffers_(arch.num_warps())
  , scoreboard_(arch_)
  , operands_(ISSUE_WIDTH)
  , dispatchers_((uint32_t)FUType::Count)
  , func_units_((uint32_t)FUType::Count)
  , lmem_switch_(NUM_LSU_BLOCKS)
  , mem_coalescers_(NUM_LSU_BLOCKS)
  , fetch_latch_(ctx, "fetch_latch", 1, 2)
  , decode_latch_(ctx, "decode_latch", 1, 2)
  , pending_icache_(arch_.num_warps())
  , commit_arbs_(ISSUE_WIDTH)
  , ibuffer_arbs_(ISSUE_WIDTH, {ArbiterType::RoundRobin, PER_ISSUE_WARPS})
{
  char sname[100];

#ifdef EXT_TCU_ENABLE
  snprintf(sname, 100, "%s-tcu", name);
  tensor_unit_ = TensorUnit::Create(sname, arch, this);
#endif
#ifdef EXT_V_ENABLE
  snprintf(sname, 100, "%s-vpu", name);
  vec_unit_ = VecUnit::Create(sname, arch, this);
#endif

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
    LSU_CHANNELS,
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

  // create lmem arbiter
  snprintf(sname, 100, "%s-lmem_arb", name);
  auto lmem_arb = LsuArbiter::Create(sname, ArbiterType::RoundRobin, NUM_LSU_BLOCKS, 1);

  // create lmem adapter
  snprintf(sname, 100, "%s-lsu_lmem_adapter", name);
  auto lsu_lmem_adapter = LsuMemAdapter::Create(sname, LSU_CHANNELS, 1);

  // connect lmem switch
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    lmem_switch_.at(b)->ReqOutLmem.bind(&lmem_arb->ReqIn.at(b));
    lmem_arb->RspOut.at(b).bind(&lmem_switch_.at(b)->RspInLmem);
  }

  // connect lmem arbiter
  lmem_arb->ReqOut.at(0).bind(&lsu_lmem_adapter->ReqIn);
  lsu_lmem_adapter->RspOut.bind(&lmem_arb->RspIn.at(0));

  // connect lmem adapter
  for (uint32_t c = 0; c < LSU_CHANNELS; ++c) {
    lsu_lmem_adapter->ReqOut.at(c).bind(&local_mem_->Inputs.at(c));
    local_mem_->Outputs.at(c).bind(&lsu_lmem_adapter->RspIn.at(c));
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
#ifdef EXT_V_ENABLE
  dispatchers_.at((int)FUType::VPU) = SimPlatform::instance().create_object<Dispatcher>(name, this, 2, NUM_VPU_BLOCKS, NUM_VPU_LANES);
#endif
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
#ifdef EXT_V_ENABLE
  snprintf(sname, 100, "%s-vpu", name);
  func_units_.at((int)FUType::VPU) = SimPlatform::instance().create_object<VpuUnit>(sname, this);
#endif
#ifdef EXT_TCU_ENABLE
  snprintf(sname, 100, "%s-tcu", name);
  func_units_.at((int)FUType::TCU) = SimPlatform::instance().create_object<TcuUnit>(sname, this);
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

  this->reset();
}

Core::~Core() {
  //--
}

void Core::reset() {

  emulator_.reset();

  trace_to_schedule_ = nullptr;

  scoreboard_.reset();

  for (auto& arb : ibuffer_arbs_) {
    arb.reset();
  }

  pending_instrs_.clear();
  pending_ifetches_ = 0;

  perf_stats_ = PerfStats();
}

void Core::tick() {
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
  perf_stats_.active_warps += emulator_.active_warps().count();
  perf_stats_.stalled_warps += emulator_.stalled_warps().count();

  // get next instruction to schedule
  auto trace = trace_to_schedule_;
  if (trace == nullptr) {
    trace = emulator_.step();
    if (trace == nullptr) {
      ++perf_stats_.sched_idle;
      return;
    }
    trace_to_schedule_ = trace;
  }

  // advance to fetch stage
  if (fetch_latch_.try_push(trace)) {
    DT(3, this->name() << "-pipeline schedule: " << *trace);
    // suspend warp until decode
    emulator_.suspend(trace->wid);
    // clear schedule trace
    trace_to_schedule_ = nullptr;
    // track pending instructions
    pending_instrs_.push_back(trace);
    // profiling
    perf_stats_.issued_warps += 1;
    perf_stats_.issued_threads += trace->tmask.count();
  }
}

void Core::fetch() {
  perf_stats_.ifetch_latency += pending_ifetches_;

  // handle icache response
  auto& icache_rsp = icache_rsp_in.at(0);
  if (!icache_rsp.empty()){
    auto& mem_rsp = icache_rsp.peek();
    auto trace = pending_icache_.at(mem_rsp.tag);
    if (decode_latch_.try_push(trace)) {
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

  // release warp
  if (!trace->fetch_stall) {
    emulator_.resume(trace->wid);
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
    bool has_instrs = false;
    BitVector<> ready_set(PER_ISSUE_WARPS);
    for (uint32_t w = 0; w < PER_ISSUE_WARPS; ++w) {
      uint32_t wid = w * ISSUE_WIDTH + iw;
      auto& ibuffer = ibuffers_.at(wid);
      if (ibuffer->empty())
        continue;
      // check scoreboard
      has_instrs = true;
      auto trace = ibuffer->peek();
      if (scoreboard_.in_use(trace)) {
        auto uses = scoreboard_.get_uses(trace);
        if (!trace->log_once(true)) {
          DTH(4, "*** scoreboard-stall: dependents={");
          for (uint32_t j = 0, n = uses.size(); j < n; ++j) {
            auto& use = uses.at(j);
            __unused (use);
            if (j) DTN(4, ", ");
            DTN(4, use.reg_type << use.reg_id << " (#" << use.uuid << ")");
          }
          DTN(4, "}, " << *trace << std::endl);
        }
        ++perf_stats_.scrb_stalls;
      } else {
        trace->log_once(false);
        ready_set.set(w); // mark instruction as ready
      }
    }

    if (ready_set.any()) {
      // select one instruction from ready set
      auto w = ibuffer_arbs_.at(iw).grant(ready_set);
      uint32_t wid = w * ISSUE_WIDTH + iw;
      auto& ibuffer = ibuffers_.at(wid);
      auto trace = ibuffer->peek();
      // to operand stage
      if (operands_.at(iw)->Input.try_send(trace)) {
        DT(3, this->name() << "-pipeline ibuffer: " << *trace);
        if (trace->wb) {
          // update scoreboard
          scoreboard_.reserve(trace);
        }
        ibuffer->pop();
      }
    }

    // track scoreboard stalls
    if (has_instrs && !ready_set.any()) {
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
      #ifdef EXT_TCU_ENABLE
        case FUType::TCU: ++perf_stats_.tcu_stalls; break;
      #endif
      #ifdef EXT_V_ENABLE
        case FUType::VPU: ++perf_stats_.vpu_stalls; break;
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
        scoreboard_.release(trace);
      }

      // instruction mix profiling
      switch (trace->fu_type) {
      case FUType::ALU: ++perf_stats_.alu_instrs; break;
      case FUType::FPU: ++perf_stats_.fpu_instrs; break;
      case FUType::LSU: ++perf_stats_.lsu_instrs; break;
      case FUType::SFU: ++perf_stats_.sfu_instrs; break;
    #ifdef EXT_TCU_ENABLE
      case FUType::TCU: ++perf_stats_.tcu_instrs; break;
    #endif
    #ifdef EXT_V_ENABLE
      case FUType::VPU: ++perf_stats_.vpu_instrs; break;
    #endif
      default: assert(false);
      }
      // track committed instructions
      perf_stats_.instrs += 1;
    #ifdef EXT_V_ENABLE
      if (std::get_if<VsetType>(&trace->op_type)
        || std::get_if<VlsType>(&trace->op_type)
        || std::get_if<VopType>(&trace->op_type)) {
        perf_stats_.vinstrs += 1;
      }
    #endif
    // instruction completed
    pending_instrs_.remove(trace);
    }

    // delete the trace
    trace_pool_.deallocate(trace, 1);

    commit_arb->Outputs.at(0).pop();
  }
}

int Core::get_exitcode() const {
  return emulator_.get_exitcode();
}

bool Core::running() const {
  if (emulator_.running() || !pending_instrs_.empty()) {
    return true;
  }
  return false;
}

void Core::resume(uint32_t wid) {
  emulator_.resume(wid);
}

uint32_t Core::barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t wid, bool is_async_bar) {
  return emulator_.barrier_arrive(bar_id, count, wid, is_async_bar);
}

bool Core::barrier_wait(uint32_t bar_id, uint32_t token, uint32_t wid) {
  return emulator_.barrier_wait(bar_id, token, wid);
}

bool Core::wspawn(uint32_t num_warps, Word nextPC) {
  return emulator_.wspawn(num_warps, nextPC);
}

bool Core::setTmask(uint32_t wid, const ThreadMask& tmask) {
  return emulator_.setTmask(wid, tmask);
}

void Core::attach_ram(RAM* ram) {
  emulator_.attach_ram(ram);
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

#ifdef VM_ENABLE
void Core::set_satp(uint64_t satp) {
  emulator_.set_satp(satp); //JAEWON wit, tid???
  // emulator_.set_csr(VX_CSR_SATP,satp,0,0); //JAEWON wit, tid???
}
#endif
