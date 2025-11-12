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
           uint32_t core_id,
           Socket* socket,
           const Arch &arch,
           const DCRS &dcrs
           )
  : SimObject(ctx, StrFormat("core%d", core_id))
  , icache_req_ports(1, this)
  , icache_rsp_ports(1, this)
  , dcache_req_ports(DCACHE_NUM_REQS, this)
  , dcache_rsp_ports(DCACHE_NUM_REQS, this)
  , core_id_(core_id)
  , socket_(socket)
  , arch_(arch)
#ifdef EXT_TCU_ENABLE
  , tensor_unit_(TensorUnit::Create("tcu", arch, this))
#endif
#ifdef EXT_V_ENABLE
  , vec_unit_(VecUnit::Create("vpu", arch, this))
#endif
  , emulator_(arch, dcrs, this)
  , ibuffers_(arch.num_warps(), IBUF_SIZE)
  , scoreboard_(arch_)
  , operands_(ISSUE_WIDTH)
  , dispatchers_((uint32_t)FUType::Count)
  , func_units_((uint32_t)FUType::Count)
  , lmem_switch_(NUM_LSU_BLOCKS)
  , mem_coalescers_(NUM_LSU_BLOCKS)
  , pending_icache_(arch_.num_warps())
  , commit_arbs_(ISSUE_WIDTH)
  , ibuffer_arbs_(ISSUE_WIDTH, {ArbiterType::RoundRobin, PER_ISSUE_WARPS})
{
  char sname[100];

  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    operands_.at(iw) = Operands::Create(this);
  }

  // create the memory coalescer
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    snprintf(sname, 100, "%s-coalescer%d", this->name().c_str(), b);
    mem_coalescers_.at(b) = MemCoalescer::Create(sname, LSU_CHANNELS, DCACHE_CHANNELS, DCACHE_WORD_SIZE, LSUQ_OUT_SIZE, 1);
  }

  // create local memory
  snprintf(sname, 100, "%s-lmem", this->name().c_str());
  local_mem_ = LocalMem::Create(sname, LocalMem::Config{
    (1 << LMEM_LOG_SIZE),
    LSU_WORD_SIZE,
    LSU_CHANNELS,
    log2ceil(LMEM_NUM_BANKS),
    false
  });

  // create lmem switch
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    snprintf(sname, 100, "%s-lmem_switch%d", this->name().c_str(), b);
    lmem_switch_.at(b) = LocalMemSwitch::Create(sname, 1);
  }

  // create dcache adapter
  std::vector<LsuMemAdapter::Ptr> lsu_dcache_adapter(NUM_LSU_BLOCKS);
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    snprintf(sname, 100, "%s-lsu_dcache_adapter%d", this->name().c_str(), b);
    lsu_dcache_adapter.at(b) = LsuMemAdapter::Create(sname, DCACHE_CHANNELS, 1);
  }

  // create lmem arbiter
  snprintf(sname, 100, "%s-lmem_arb", this->name().c_str());
  auto lmem_arb = LsuArbiter::Create(sname, ArbiterType::RoundRobin, NUM_LSU_BLOCKS, 1);

  // create lmem adapter
  snprintf(sname, 100, "%s-lsu_lmem_adapter", this->name().c_str());
  auto lsu_lmem_adapter = LsuMemAdapter::Create(sname, LSU_CHANNELS, 1);

  // connect lmem switch
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    lmem_switch_.at(b)->ReqDC.bind(&mem_coalescers_.at(b)->ReqIn);
    lmem_switch_.at(b)->ReqLmem.bind(&lmem_arb->ReqIn.at(b));

    mem_coalescers_.at(b)->RspIn.bind(&lmem_switch_.at(b)->RspDC);
    lmem_arb->RspIn.at(b).bind(&lmem_switch_.at(b)->RspLmem);
  }

  // connect lmem arbiter
  lmem_arb->ReqOut.at(0).bind(&lsu_lmem_adapter->ReqIn);
  lsu_lmem_adapter->RspIn.bind(&lmem_arb->RspOut.at(0));

  // connect lmem adapter
  for (uint32_t c = 0; c < LSU_CHANNELS; ++c) {
    lsu_lmem_adapter->ReqOut.at(c).bind(&local_mem_->Inputs.at(c));
    local_mem_->Outputs.at(c).bind(&lsu_lmem_adapter->RspOut.at(c));
  }

  // connect dcache coalescer
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    mem_coalescers_.at(b)->ReqOut.bind(&lsu_dcache_adapter.at(b)->ReqIn);
    lsu_dcache_adapter.at(b)->RspIn.bind(&mem_coalescers_.at(b)->RspOut);
  }

  // connect dcache adapter
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    for (uint32_t c = 0; c < DCACHE_CHANNELS; ++c) {
      uint32_t p = b * DCACHE_CHANNELS + c;
      lsu_dcache_adapter.at(b)->ReqOut.at(c).bind(&dcache_req_ports.at(p));
      dcache_rsp_ports.at(p).bind(&lsu_dcache_adapter.at(b)->RspOut.at(c));
    }
  }

  // initialize dispatchers
  dispatchers_.at((int)FUType::ALU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_ALU_BLOCKS, NUM_ALU_LANES);
  dispatchers_.at((int)FUType::FPU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_FPU_BLOCKS, NUM_FPU_LANES);
  dispatchers_.at((int)FUType::LSU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_LSU_BLOCKS, NUM_LSU_LANES);
  dispatchers_.at((int)FUType::SFU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_SFU_BLOCKS, NUM_SFU_LANES);
#ifdef EXT_V_ENABLE
  dispatchers_.at((int)FUType::VPU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_VPU_BLOCKS, NUM_VPU_LANES);
#endif
#ifdef EXT_TCU_ENABLE
  dispatchers_.at((int)FUType::TCU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_TCU_BLOCKS, NUM_TCU_LANES);
#endif

  // initialize execute units
  func_units_.at((int)FUType::ALU) = SimPlatform::instance().create_object<AluUnit>(this);
  func_units_.at((int)FUType::FPU) = SimPlatform::instance().create_object<FpuUnit>(this);
  func_units_.at((int)FUType::LSU) = SimPlatform::instance().create_object<LsuUnit>(this);
  func_units_.at((int)FUType::SFU) = SimPlatform::instance().create_object<SfuUnit>(this);
#ifdef EXT_V_ENABLE
  func_units_.at((int)FUType::VPU) = SimPlatform::instance().create_object<VpuUnit>(this);
#endif
#ifdef EXT_TCU_ENABLE
  func_units_.at((int)FUType::TCU) = SimPlatform::instance().create_object<TcuUnit>(this);
#endif

  // bind commit arbiters
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    snprintf(sname, 100, "%s-commit-arb%d", this->name().c_str(), iw);
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

  for (auto& commit_arb : commit_arbs_) {
    commit_arb->reset();
  }

  for (auto& ibuf : ibuffers_) {
    ibuf.reset();
  }

  scoreboard_.reset();
  fetch_latch_.reset();
  decode_latch_.reset();
  pending_icache_.clear();

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
  auto trace = emulator_.step();
  if (trace == nullptr) {
    ++perf_stats_.sched_idle;
    return;
  }

  // suspend warp until decode
  emulator_.suspend(trace->wid);

  DT(3, "pipeline-schedule: " << *trace);

  // advance to fetch stage
  fetch_latch_.push(trace);
  pending_instrs_.push_back(trace);
}

void Core::fetch() {
  perf_stats_.ifetch_latency += pending_ifetches_;

  // handle icache response
  auto& icache_rsp_port = icache_rsp_ports.at(0);
  if (!icache_rsp_port.empty()){
    auto& mem_rsp = icache_rsp_port.front();
    auto trace = pending_icache_.at(mem_rsp.tag);
    decode_latch_.push(trace);
    DT(3, "icache-rsp: addr=0x" << std::hex << trace->PC << ", tag=0x" << mem_rsp.tag << std::dec << ", " << *trace);
    pending_icache_.release(mem_rsp.tag);
    icache_rsp_port.pop();
    --pending_ifetches_;
  }

  // send icache request
  if (fetch_latch_.empty())
    return;
  auto trace = fetch_latch_.front();
  MemReq mem_req;
  mem_req.addr  = trace->PC;
  mem_req.write = false;
  mem_req.tag   = pending_icache_.allocate(trace);
  mem_req.cid   = trace->cid;
  mem_req.uuid  = trace->uuid;
  icache_req_ports.at(0).push(mem_req, 2);
  DT(3, "icache-req: addr=0x" << std::hex << mem_req.addr << ", tag=0x" << mem_req.tag << std::dec << ", " << *trace);
  fetch_latch_.pop();
  ++perf_stats_.ifetches;
  ++pending_ifetches_;
}

void Core::decode() {
  if (decode_latch_.empty())
    return;

  auto trace = decode_latch_.front();

  // check ibuffer capacity
  auto& ibuffer = ibuffers_.at(trace->wid);
  if (ibuffer.full()) {
    if (!trace->log_once(true)) {
      DT(4, "*** ibuffer-stall: " << *trace);
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

  DT(3, "pipeline-decode: " << *trace);

  // insert to ibuffer
  ibuffer.push(trace);

  decode_latch_.pop();
}

void Core::issue() {
  // dispatch operands
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    auto& operand = operands_.at(iw);
    if (operand->Output.empty())
      continue;
    auto trace = operand->Output.front();
    dispatchers_.at((int)trace->fu_type)->Inputs.at(iw).push(trace);
    operand->Output.pop();
  }

  // issue ibuffer instructions
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    bool has_instrs = false;
    BitVector<> ready_set(PER_ISSUE_WARPS);
    for (uint32_t w = 0; w < PER_ISSUE_WARPS; ++w) {
      uint32_t wid = w * ISSUE_WIDTH + iw;
      auto& ibuffer = ibuffers_.at(wid);
      if (ibuffer.empty())
        continue;
      // check scoreboard
      has_instrs = true;
      auto trace = ibuffer.top();
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
        for (uint32_t j = 0, n = uses.size(); j < n; ++j) {
          auto& use = uses.at(j);
          switch (use.fu_type) {
          case FUType::ALU: ++perf_stats_.scrb_alu; break;
          case FUType::FPU: ++perf_stats_.scrb_fpu; break;
          case FUType::LSU: ++perf_stats_.scrb_lsu; break;
          case FUType::SFU: {
            ++perf_stats_.scrb_sfu;
            if (std::get_if<WctlType>(&use.op_type)) {
              ++perf_stats_.scrb_wctl;
            } else if (std::get_if<CsrType>(&use.op_type)) {
              ++perf_stats_.scrb_csrs;
            }
          } break;
        #ifdef EXT_V_ENABLE
          case FUType::VPU: ++perf_stats_.scrb_vpu; break;
        #endif
        #ifdef EXT_TCU_ENABLE
          case FUType::TCU: ++perf_stats_.scrb_tcu; break;
        #endif
          default: assert(false);
          }
        }
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
      auto trace = ibuffer.top();
      // update scoreboard
      DT(3, "pipeline-ibuffer: " << *trace);
      if (trace->wb) {
        scoreboard_.reserve(trace);
      }
      // to operand stage
      operands_.at(iw)->Input.push(trace, 1);
      ibuffer.pop();
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
      auto trace = dispatch->Outputs.at(iw).front();
      func_unit->Inputs.at(iw).push(trace, 2);
      dispatch->Outputs.at(iw).pop();
    }
  }
}

void Core::commit() {
  // process completed instructions
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    auto& commit_arb = commit_arbs_.at(iw);
    if (commit_arb->Outputs.at(0).empty())
      continue;
    auto trace = commit_arb->Outputs.at(0).front().data;

    // advance to commit stage
    DT(3, "pipeline-commit: " << *trace);
    assert(trace->cid == core_id_);

    // update scoreboard
    if (trace->eop) {
      if (trace->wb) {
        operands_.at(iw)->writeback(trace);
        scoreboard_.release(trace);
      }
      auto orig_size = pending_instrs_.size();
      pending_instrs_.remove(trace);
      if (pending_instrs_.size() != orig_size) {
        perf_stats_.instrs += trace->tmask.count();
      #ifdef EXT_V_ENABLE
        if (std::get_if<VsetType>(&trace->op_type)
         || std::get_if<VlsType>(&trace->op_type)
         || std::get_if<VopType>(&trace->op_type)) {
          perf_stats_.vinstrs += trace->tmask.count();
        }
      #endif
      }
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
  #ifndef NDEBUG
    for (auto& trace : pending_instrs_) {
      DT(5, "pipeline-pending: " << *trace);
    }
  #endif
    return true;
  }
  return false;
}

void Core::resume(uint32_t wid) {
  emulator_.resume(wid);
}

bool Core::barrier(uint32_t bar_id, uint32_t count, uint32_t wid) {
  return emulator_.barrier(bar_id, count, wid);
}

bool Core::wspawn(uint32_t num_warps, Word nextPC) {
  return emulator_.wspawn(num_warps, nextPC);
}

void Core::attach_ram(RAM* ram) {
  emulator_.attach_ram(ram);
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