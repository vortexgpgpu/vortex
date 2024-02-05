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
#include "decode.h"
#include "core.h"
#include "socket.h"
#include "debug.h"
#include "constants.h"
#include "processor_impl.h"

using namespace vortex;

Core::Core(const SimContext& ctx, 
           uint32_t core_id, 
           Socket* socket,
           const Arch &arch, 
           const DCRS &dcrs)
    : SimObject(ctx, "core")
    , icache_req_ports(1, this)
    , icache_rsp_ports(1, this)
    , dcache_req_ports(NUM_LSU_LANES, this)
    , dcache_rsp_ports(NUM_LSU_LANES, this)
    , core_id_(core_id)
    , socket_(socket)
    , arch_(arch)
    , dcrs_(dcrs)
    , decoder_(arch)
    , warps_(arch.num_warps())
    , barriers_(arch.num_barriers(), 0)
    , fcsrs_(arch.num_warps(), 0)
    , ibuffers_(arch.num_warps(), IBUF_SIZE)
    , scoreboard_(arch_)
    , operands_(ISSUE_WIDTH)
    , dispatchers_((uint32_t)ExeType::ExeTypeCount)
    , exe_units_((uint32_t)ExeType::ExeTypeCount)
    , smem_demuxs_(NUM_LSU_LANES)
    , fetch_latch_("fetch")
    , decode_latch_("decode")
    , pending_icache_(arch_.num_warps())
    , csrs_(arch.num_warps())  
    , commit_arbs_(ISSUE_WIDTH)
{
  char sname[100];

  for (uint32_t i = 0; i < arch_.num_warps(); ++i) {
    csrs_.at(i).resize(arch.num_threads());
  }

  for (uint32_t i = 0; i < arch_.num_warps(); ++i) {
    warps_.at(i) = std::make_shared<Warp>(this, i);
  }

  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    operands_.at(i) = SimPlatform::instance().create_object<Operand>();
  }

  // initialize shared memory
  snprintf(sname, 100, "core%d-shared_mem", core_id);
  shared_mem_ = SharedMem::Create(sname, SharedMem::Config{
    (1 << SMEM_LOG_SIZE),
    sizeof(Word),
    NUM_LSU_LANES, 
    NUM_LSU_LANES,
    false
  });
  for (uint32_t i = 0; i < NUM_LSU_LANES; ++i) {
    snprintf(sname, 100, "core%d-smem_demux%d", core_id, i);
    auto smem_demux = SMemDemux::Create(sname);
    
    smem_demux->ReqDC.bind(&dcache_req_ports.at(i));
    dcache_rsp_ports.at(i).bind(&smem_demux->RspDC);

    smem_demux->ReqSM.bind(&shared_mem_->Inputs.at(i));
    shared_mem_->Outputs.at(i).bind(&smem_demux->RspSM);

    smem_demuxs_.at(i) = smem_demux;
  }

  // initialize dispatchers
  dispatchers_.at((int)ExeType::ALU) = SimPlatform::instance().create_object<Dispatcher>(arch, 2, NUM_ALU_BLOCKS, NUM_ALU_LANES);
  dispatchers_.at((int)ExeType::FPU) = SimPlatform::instance().create_object<Dispatcher>(arch, 2, NUM_FPU_BLOCKS, NUM_FPU_LANES);
  dispatchers_.at((int)ExeType::LSU) = SimPlatform::instance().create_object<Dispatcher>(arch, 2, 1, NUM_LSU_LANES);
  dispatchers_.at((int)ExeType::SFU) = SimPlatform::instance().create_object<Dispatcher>(arch, 2, 1, NUM_SFU_LANES);
  
  // initialize execute units
  exe_units_.at((int)ExeType::ALU) = SimPlatform::instance().create_object<AluUnit>(this);
  exe_units_.at((int)ExeType::FPU) = SimPlatform::instance().create_object<FpuUnit>(this);
  exe_units_.at((int)ExeType::LSU) = SimPlatform::instance().create_object<LsuUnit>(this);
  exe_units_.at((int)ExeType::SFU) = SimPlatform::instance().create_object<SfuUnit>(this);

  // bind commit arbiters
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {    
    snprintf(sname, 100, "core%d-commit-arb%d", core_id, i);
    auto arbiter = TraceSwitch::Create(sname, ArbiterType::RoundRobin, (uint32_t)ExeType::ExeTypeCount, 1);
    for (uint32_t j = 0; j < (uint32_t)ExeType::ExeTypeCount; ++j) {
      exe_units_.at(j)->Outputs.at(i).bind(&arbiter->Inputs.at(j));
    }
    commit_arbs_.at(i) = arbiter;
  }

  this->reset();
}

Core::~Core() {
  this->cout_flush();
}

void Core::reset() {
  for (auto& warp : warps_) {
    warp->reset();
  }
  warps_.at(0)->setTmask(0, true);
  active_warps_ = 1;

  for (auto& exe_unit : exe_units_) {
    exe_unit->reset();
  }
 
  for (auto& commit_arb : commit_arbs_) {
    commit_arb->reset();
  }
  
  for (auto& barrier : barriers_) {
    barrier.reset();
  }
  
  for (auto& fcsr : fcsrs_) {
    fcsr = 0;
  }
  
  for (auto& ibuf : ibuffers_) {
    ibuf.clear();
  }

  ibuffer_idx_ = 0;

  scoreboard_.clear();
  fetch_latch_.clear();
  decode_latch_.clear();
  pending_icache_.clear();
  stalled_warps_.reset();
  issued_instrs_ = 0;
  committed_instrs_ = 0;
  exited_ = false;
  perf_stats_ = PerfStats();
  pending_ifetches_ = 0;
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
  int scheduled_warp = -1;

  // find next ready warp
  for (size_t wid = 0, nw = arch_.num_warps(); wid < nw; ++wid) {  
    bool warp_active = active_warps_.test(wid);
    bool warp_stalled = stalled_warps_.test(wid); 
    if (warp_active && !warp_stalled) {      
      scheduled_warp = wid;
      break;
    }
  }
  if (scheduled_warp == -1) {
    ++perf_stats_.sched_idle;
    return;
  }

  // suspend warp until decode
  stalled_warps_.set(scheduled_warp);

  // evaluate scheduled warp
  auto& warp = warps_.at(scheduled_warp);
  auto trace = warp->eval();

  DT(3, "pipeline-schedule: " << *trace);

  // advance to fetch stage
  fetch_latch_.push(trace);
  ++issued_instrs_;
}

void Core::fetch() {
  perf_stats_.ifetch_latency += pending_ifetches_;

  // handle icache response
  auto& icache_rsp_port = icache_rsp_ports.at(0);
  if (!icache_rsp_port.empty()){
    auto& mem_rsp = icache_rsp_port.front();
    auto trace = pending_icache_.at(mem_rsp.tag);
    decode_latch_.push(trace);
    DT(3, "icache-rsp: addr=0x" << std::hex << trace->PC << ", tag=" << mem_rsp.tag << ", " << *trace);
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
  icache_req_ports.at(0).send(mem_req, 2);    
  DT(3, "icache-req: addr=0x" << std::hex << mem_req.addr << ", tag=" << mem_req.tag << ", " << *trace);    
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
      DT(3, "*** ibuffer-stall: " << *trace);
    }
    ++perf_stats_.ibuf_stalls;
    return;
  } else {
    trace->log_once(false);
  }
  
  // release warp
  if (!trace->fetch_stall) {
    assert(stalled_warps_.test(trace->wid));
    stalled_warps_.reset(trace->wid);
  }

  DT(3, "pipeline-decode: " << *trace);

  // insert to ibuffer 
  ibuffer.push(trace);

  decode_latch_.pop();
}

void Core::issue() {   
  // operands to dispatchers
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    auto& operand = operands_.at(i);    
    if (operand->Output.empty())
      continue;
    auto trace = operand->Output.front();
    if (dispatchers_.at((int)trace->exe_type)->push(i, trace)) {
      operand->Output.pop();
      trace->log_once(false);
    } else {
      if (!trace->log_once(true)) {
        DT(3, "*** dispatch-stall: " << *trace);
      }
    }
  }

  // issue ibuffer instructions
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    uint32_t ii = (ibuffer_idx_ + i) % ibuffers_.size();
    auto& ibuffer = ibuffers_.at(ii);
    if (ibuffer.empty())
      continue;

    auto trace = ibuffer.top();

    // check scoreboard
    if (scoreboard_.in_use(trace)) {
      auto uses = scoreboard_.get_uses(trace);
      if (!trace->log_once(true)) {
        DTH(3, "*** scoreboard-stall: dependents={");        
        for (uint32_t j = 0, n = uses.size(); j < n; ++j) {
          auto& use = uses.at(j);
          __unused (use);
          if (j) DTN(3, ", ");
          DTN(3, use.reg_type << use.reg_id << "(#" << use.uuid << ")");
        }
        DTN(3, "}, " << *trace << std::endl);
      }
      for (uint32_t j = 0, n = uses.size(); j < n; ++j) {
        auto& use = uses.at(j);
        switch (use.exe_type) {        
        case ExeType::ALU: ++perf_stats_.scrb_alu; break;
        case ExeType::FPU: ++perf_stats_.scrb_fpu; break;
        case ExeType::LSU: ++perf_stats_.scrb_lsu; break;        
        case ExeType::SFU: {
          ++perf_stats_.scrb_sfu;
          switch (use.sfu_type) {
          case SfuType::TMC:
          case SfuType::WSPAWN:
          case SfuType::SPLIT:
          case SfuType::JOIN:
          case SfuType::BAR:
          case SfuType::PRED: ++perf_stats_.scrb_wctl; break;
          case SfuType::CSRRW:
          case SfuType::CSRRS:
          case SfuType::CSRRC: ++perf_stats_.scrb_csrs; break;
          default: assert(false);
          }
        } break;
        default: assert(false);
        }        
      }
      ++perf_stats_.scrb_stalls;
      continue;
    } else {
      trace->log_once(false);
    }

    // update scoreboard
    if (trace->wb) {
      scoreboard_.reserve(trace);
    }

    DT(3, "pipeline-scoreboard: " << *trace);

    // to operand stage
    operands_.at(i)->Input.send(trace, 1);

    ibuffer.pop();
  }
  ibuffer_idx_ += ISSUE_WIDTH;
}

void Core::execute() {
  for (uint32_t i = 0; i < (uint32_t)ExeType::ExeTypeCount; ++i) {
    auto& dispatch = dispatchers_.at(i);
    auto& exe_unit = exe_units_.at(i);
    for (uint32_t j = 0; j < ISSUE_WIDTH; ++j) {
      if (dispatch->Outputs.at(j).empty())
        continue;
      auto trace = dispatch->Outputs.at(j).front();
      exe_unit->Inputs.at(j).send(trace, 1);
      dispatch->Outputs.at(j).pop();
    }
  }
}

void Core::commit() {
  // process completed instructions 
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    auto& commit_arb = commit_arbs_.at(i);
    if (commit_arb->Outputs.at(0).empty())
      continue;
    auto trace = commit_arb->Outputs.at(0).front();

    // advance to commit stage
    DT(3, "pipeline-commit: " << *trace);
    assert(trace->cid == core_id_);

    // update scoreboard
    if (trace->eop) {
      if (trace->wb) {
        scoreboard_.release(trace);
      }

      assert(committed_instrs_ <= issued_instrs_);
      ++committed_instrs_;

      perf_stats_.instrs += trace->tmask.count();
    }

    commit_arb->Outputs.at(0).pop();

    // delete the trace
    delete trace;
  }
}

void Core::wspawn(uint32_t num_warps, Word nextPC) {
  uint32_t active_warps = std::min<uint32_t>(num_warps, arch_.num_warps());
  DP(3, "*** Activate " << (active_warps-1) << " warps at PC: " << std::hex << nextPC);
  for (uint32_t i = 1; i < active_warps; ++i) {
    auto warp = warps_.at(i);
    warp->setPC(nextPC);
    warp->setTmask(0, true);
    active_warps_.set(i);
  }
}

void Core::barrier(uint32_t bar_id, uint32_t count, uint32_t warp_id) {
  uint32_t bar_idx = bar_id & 0x7fffffff;
  bool is_global = (bar_id >> 31);

  auto& barrier = barriers_.at(bar_idx);
  barrier.set(warp_id);
  DP(3, "*** Suspend core #" << core_id_ << ", warp #" << warp_id << " at barrier #" << bar_idx);

  if (is_global) {
    // global barrier handling
    if (barrier.count() == active_warps_.count()) {
      socket_->barrier(bar_idx, count, core_id_);
      barrier.reset();
    }    
  } else {
    // local barrier handling
    if (barrier.count() == (size_t)count) {
      // resume suspended warps
      for (uint32_t i = 0; i < arch_.num_warps(); ++i) {
        if (barrier.test(i)) {
          DP(3, "*** Resume core #" << core_id_ << ", warp #" << i << " at barrier #" << bar_idx);
          stalled_warps_.reset(i);
        }
      }
      barrier.reset();
    }
  }
}

void Core::icache_read(void *data, uint64_t addr, uint32_t size) {
  mmu_.read(data, addr, size, 0);
}

AddrType Core::get_addr_type(uint64_t addr) {
  if (SM_ENABLED) {
    if (addr >= SMEM_BASE_ADDR && addr < (SMEM_BASE_ADDR + (1 << SMEM_LOG_SIZE))) {
        return AddrType::Shared;
    }
  }
  if (addr >= IO_BASE_ADDR) {
     return AddrType::IO;
  }
  return AddrType::Global;
}

void Core::dcache_read(void *data, uint64_t addr, uint32_t size) {  
  auto type = this->get_addr_type(addr);
  if (type == AddrType::Shared) {
    shared_mem_->read(data, addr, size);
  } else {  
    mmu_.read(data, addr, size, 0);
  }

  DPH(2, "Mem Read: addr=0x" << std::hex << addr << ", data=0x" << ByteStream(data, size) << " (size=" << size << ", type=" << type << ")" << std::endl);
}

void Core::dcache_write(const void* data, uint64_t addr, uint32_t size) {  
  auto type = this->get_addr_type(addr);
  if (addr >= uint64_t(IO_COUT_ADDR)
   && addr < (uint64_t(IO_COUT_ADDR) + IO_COUT_SIZE)) {
     this->writeToStdOut(data, addr, size);
  } else {
    if (type == AddrType::Shared) {
      shared_mem_->write(data, addr, size);
    } else {
      mmu_.write(data, addr, size, 0);
    }
  }
  DPH(2, "Mem Write: addr=0x" << std::hex << addr << ", data=0x" << ByteStream(data, size) << " (size=" << size << ", type=" << type << ")" << std::endl);  
}

void Core::dcache_amo_reserve(uint64_t addr) {
  auto type = this->get_addr_type(addr);
  if (type == AddrType::Global) {
    mmu_.amo_reserve(addr);
  }
}

bool Core::dcache_amo_check(uint64_t addr) {
  auto type = this->get_addr_type(addr);
  if (type == AddrType::Global) {
    return mmu_.amo_check(addr);
  }
  return false;
}

void Core::writeToStdOut(const void* data, uint64_t addr, uint32_t size) {
  if (size != 1)
    std::abort();
  uint32_t tid = (addr - IO_COUT_ADDR) & (IO_COUT_SIZE-1);
  auto& ss_buf = print_bufs_[tid];
  char c = *(char*)data;
  ss_buf << c;
  if (c == '\n') {
    std::cout << std::dec << "#" << tid << ": " << ss_buf.str() << std::flush;
    ss_buf.str("");
  }
}

void Core::cout_flush() {
  for (auto& buf : print_bufs_) {
    auto str = buf.second.str();
    if (!str.empty()) {
      std::cout << "#" << buf.first << ": " << str << std::endl;
    }
  }
}

uint32_t Core::get_csr(uint32_t addr, uint32_t tid, uint32_t wid) {
  switch (addr) {
  case VX_CSR_SATP:
  case VX_CSR_PMPCFG0:
  case VX_CSR_PMPADDR0:
  case VX_CSR_MSTATUS:
  case VX_CSR_MISA:
  case VX_CSR_MEDELEG:
  case VX_CSR_MIDELEG:
  case VX_CSR_MIE:
  case VX_CSR_MTVEC:
  case VX_CSR_MEPC:
  case VX_CSR_MNSTATUS:
    return 0;

  case VX_CSR_FFLAGS:
    return fcsrs_.at(wid) & 0x1F;
  case VX_CSR_FRM:
    return (fcsrs_.at(wid) >> 5);
  case VX_CSR_FCSR:
    return fcsrs_.at(wid);
  case VX_CSR_MHARTID: // global thread ID
    return (core_id_ * arch_.num_warps() + wid) * arch_.num_threads() + tid;
  case VX_CSR_THREAD_ID: // thread ID
    return tid;
  case VX_CSR_WARP_ID: // warp ID
    return wid;
  case VX_CSR_CORE_ID: // core ID
    return core_id_;
  case VX_CSR_THREAD_MASK: // thread mask
    return warps_.at(wid)->getTmask();
  case VX_CSR_WARP_MASK: // active warps
    return active_warps_.to_ulong();
  case VX_CSR_NUM_THREADS: // Number of threads per warp
    return arch_.num_threads();
  case VX_CSR_NUM_WARPS: // Number of warps per core
    return arch_.num_warps();
  case VX_CSR_NUM_CORES: // Number of cores per cluster
    return uint32_t(arch_.num_cores()) * arch_.num_clusters();
  case VX_CSR_MCYCLE: // NumCycles
    return perf_stats_.cycles & 0xffffffff;
  case VX_CSR_MCYCLE_H: // NumCycles
    return (uint32_t)(perf_stats_.cycles >> 32);
  case VX_CSR_MINSTRET: // NumInsts
    return perf_stats_.instrs & 0xffffffff;
  case VX_CSR_MINSTRET_H: // NumInsts
    return (uint32_t)(perf_stats_.instrs >> 32);
  default:
    if ((addr >= VX_CSR_MPM_BASE && addr < (VX_CSR_MPM_BASE + 32))
     || (addr >= VX_CSR_MPM_BASE_H && addr < (VX_CSR_MPM_BASE_H + 32))) {
      // user-defined MPM CSRs
      auto perf_class = dcrs_.base_dcrs.read(VX_DCR_BASE_MPM_CLASS);
      switch (perf_class) {                
      case VX_DCR_MPM_CLASS_NONE: 
        break;    
      case VX_DCR_MPM_CLASS_CORE: {
        switch (addr) {
        case VX_CSR_MPM_SCHED_ID:  return perf_stats_.sched_idle & 0xffffffff; 
        case VX_CSR_MPM_SCHED_ID_H:return perf_stats_.sched_idle >> 32;
        case VX_CSR_MPM_SCHED_ST:  return perf_stats_.sched_stalls & 0xffffffff; 
        case VX_CSR_MPM_SCHED_ST_H:return perf_stats_.sched_stalls >> 32;
        case VX_CSR_MPM_IBUF_ST:   return perf_stats_.ibuf_stalls & 0xffffffff; 
        case VX_CSR_MPM_IBUF_ST_H: return perf_stats_.ibuf_stalls >> 32; 
        case VX_CSR_MPM_SCRB_ST:   return perf_stats_.scrb_stalls & 0xffffffff;
        case VX_CSR_MPM_SCRB_ST_H: return perf_stats_.scrb_stalls >> 32;
        case VX_CSR_MPM_SCRB_ALU:  return perf_stats_.scrb_alu & 0xffffffff;
        case VX_CSR_MPM_SCRB_ALU_H:return perf_stats_.scrb_alu >> 32;
        case VX_CSR_MPM_SCRB_FPU:  return perf_stats_.scrb_fpu & 0xffffffff;
        case VX_CSR_MPM_SCRB_FPU_H:return perf_stats_.scrb_fpu >> 32;
        case VX_CSR_MPM_SCRB_LSU:  return perf_stats_.scrb_lsu & 0xffffffff;
        case VX_CSR_MPM_SCRB_LSU_H:return perf_stats_.scrb_lsu >> 32;
        case VX_CSR_MPM_SCRB_SFU:  return perf_stats_.scrb_sfu & 0xffffffff;
        case VX_CSR_MPM_SCRB_SFU_H:return perf_stats_.scrb_sfu >> 32;
        case VX_CSR_MPM_SCRB_WCTL: return perf_stats_.scrb_wctl & 0xffffffff;
        case VX_CSR_MPM_SCRB_WCTL_H: return perf_stats_.scrb_wctl >> 32;
        case VX_CSR_MPM_SCRB_CSRS: return perf_stats_.scrb_csrs & 0xffffffff;
        case VX_CSR_MPM_SCRB_CSRS_H: return perf_stats_.scrb_csrs >> 32;
        case VX_CSR_MPM_IFETCHES:  return perf_stats_.ifetches & 0xffffffff; 
        case VX_CSR_MPM_IFETCHES_H: return perf_stats_.ifetches >> 32; 
        case VX_CSR_MPM_LOADS:     return perf_stats_.loads & 0xffffffff; 
        case VX_CSR_MPM_LOADS_H:   return perf_stats_.loads >> 32; 
        case VX_CSR_MPM_STORES:    return perf_stats_.stores & 0xffffffff; 
        case VX_CSR_MPM_STORES_H:  return perf_stats_.stores >> 32;
        case VX_CSR_MPM_IFETCH_LT: return perf_stats_.ifetch_latency & 0xffffffff; 
        case VX_CSR_MPM_IFETCH_LT_H: return perf_stats_.ifetch_latency >> 32; 
        case VX_CSR_MPM_LOAD_LT:   return perf_stats_.load_latency & 0xffffffff; 
        case VX_CSR_MPM_LOAD_LT_H: return perf_stats_.load_latency >> 32;
       }
      } break; 
      case VX_DCR_MPM_CLASS_MEM: {
        auto proc_perf = socket_->cluster()->processor()->perf_stats();
        auto cluster_perf = socket_->cluster()->perf_stats();
        auto socket_perf = socket_->perf_stats();
        auto smem_perf = shared_mem_->perf_stats();
        switch (addr) {
        case VX_CSR_MPM_ICACHE_READS:     return socket_perf.icache.reads & 0xffffffff; 
        case VX_CSR_MPM_ICACHE_READS_H:   return socket_perf.icache.reads >> 32; 
        case VX_CSR_MPM_ICACHE_MISS_R:    return socket_perf.icache.read_misses & 0xffffffff;
        case VX_CSR_MPM_ICACHE_MISS_R_H:  return socket_perf.icache.read_misses >> 32;
        case VX_CSR_MPM_ICACHE_MSHR_ST:   return socket_perf.icache.mshr_stalls & 0xffffffff; 
        case VX_CSR_MPM_ICACHE_MSHR_ST_H: return socket_perf.icache.mshr_stalls >> 32;
        
        case VX_CSR_MPM_DCACHE_READS:     return socket_perf.dcache.reads & 0xffffffff; 
        case VX_CSR_MPM_DCACHE_READS_H:   return socket_perf.dcache.reads >> 32; 
        case VX_CSR_MPM_DCACHE_WRITES:    return socket_perf.dcache.writes & 0xffffffff; 
        case VX_CSR_MPM_DCACHE_WRITES_H:  return socket_perf.dcache.writes >> 32; 
        case VX_CSR_MPM_DCACHE_MISS_R:    return socket_perf.dcache.read_misses & 0xffffffff; 
        case VX_CSR_MPM_DCACHE_MISS_R_H:  return socket_perf.dcache.read_misses >> 32; 
        case VX_CSR_MPM_DCACHE_MISS_W:    return socket_perf.dcache.write_misses & 0xffffffff; 
        case VX_CSR_MPM_DCACHE_MISS_W_H:  return socket_perf.dcache.write_misses >> 32; 
        case VX_CSR_MPM_DCACHE_BANK_ST:   return socket_perf.dcache.bank_stalls & 0xffffffff; 
        case VX_CSR_MPM_DCACHE_BANK_ST_H: return socket_perf.dcache.bank_stalls >> 32;
        case VX_CSR_MPM_DCACHE_MSHR_ST:   return socket_perf.dcache.mshr_stalls & 0xffffffff; 
        case VX_CSR_MPM_DCACHE_MSHR_ST_H: return socket_perf.dcache.mshr_stalls >> 32;

        case VX_CSR_MPM_L2CACHE_READS:    return cluster_perf.l2cache.reads & 0xffffffff; 
        case VX_CSR_MPM_L2CACHE_READS_H:  return cluster_perf.l2cache.reads >> 32; 
        case VX_CSR_MPM_L2CACHE_WRITES:   return cluster_perf.l2cache.writes & 0xffffffff; 
        case VX_CSR_MPM_L2CACHE_WRITES_H: return cluster_perf.l2cache.writes >> 32; 
        case VX_CSR_MPM_L2CACHE_MISS_R:   return cluster_perf.l2cache.read_misses & 0xffffffff; 
        case VX_CSR_MPM_L2CACHE_MISS_R_H: return cluster_perf.l2cache.read_misses >> 32; 
        case VX_CSR_MPM_L2CACHE_MISS_W:   return cluster_perf.l2cache.write_misses & 0xffffffff; 
        case VX_CSR_MPM_L2CACHE_MISS_W_H: return cluster_perf.l2cache.write_misses >> 32; 
        case VX_CSR_MPM_L2CACHE_BANK_ST:  return cluster_perf.l2cache.bank_stalls & 0xffffffff; 
        case VX_CSR_MPM_L2CACHE_BANK_ST_H:return cluster_perf.l2cache.bank_stalls >> 32;
        case VX_CSR_MPM_L2CACHE_MSHR_ST:  return cluster_perf.l2cache.mshr_stalls & 0xffffffff; 
        case VX_CSR_MPM_L2CACHE_MSHR_ST_H:return cluster_perf.l2cache.mshr_stalls >> 32;

        case VX_CSR_MPM_L3CACHE_READS:    return proc_perf.l3cache.reads & 0xffffffff; 
        case VX_CSR_MPM_L3CACHE_READS_H:  return proc_perf.l3cache.reads >> 32; 
        case VX_CSR_MPM_L3CACHE_WRITES:   return proc_perf.l3cache.writes & 0xffffffff; 
        case VX_CSR_MPM_L3CACHE_WRITES_H: return proc_perf.l3cache.writes >> 32; 
        case VX_CSR_MPM_L3CACHE_MISS_R:   return proc_perf.l3cache.read_misses & 0xffffffff; 
        case VX_CSR_MPM_L3CACHE_MISS_R_H: return proc_perf.l3cache.read_misses >> 32; 
        case VX_CSR_MPM_L3CACHE_MISS_W:   return proc_perf.l3cache.write_misses & 0xffffffff; 
        case VX_CSR_MPM_L3CACHE_MISS_W_H: return proc_perf.l3cache.write_misses >> 32; 
        case VX_CSR_MPM_L3CACHE_BANK_ST:  return proc_perf.l3cache.bank_stalls & 0xffffffff; 
        case VX_CSR_MPM_L3CACHE_BANK_ST_H:return proc_perf.l3cache.bank_stalls >> 32;
        case VX_CSR_MPM_L3CACHE_MSHR_ST:  return proc_perf.l3cache.mshr_stalls & 0xffffffff; 
        case VX_CSR_MPM_L3CACHE_MSHR_ST_H:return proc_perf.l3cache.mshr_stalls >> 32;

        case VX_CSR_MPM_MEM_READS:        return proc_perf.mem_reads & 0xffffffff; 
        case VX_CSR_MPM_MEM_READS_H:      return proc_perf.mem_reads >> 32;
        case VX_CSR_MPM_MEM_WRITES:       return proc_perf.mem_writes & 0xffffffff; 
        case VX_CSR_MPM_MEM_WRITES_H:     return proc_perf.mem_writes >> 32; 
        case VX_CSR_MPM_MEM_LT:           return proc_perf.mem_latency & 0xffffffff; 
        case VX_CSR_MPM_MEM_LT_H :        return proc_perf.mem_latency >> 32;
         
        case VX_CSR_MPM_SMEM_READS:       return smem_perf.reads & 0xffffffff;
        case VX_CSR_MPM_SMEM_READS_H:     return smem_perf.reads >> 32;
        case VX_CSR_MPM_SMEM_WRITES:      return smem_perf.writes & 0xffffffff;
        case VX_CSR_MPM_SMEM_WRITES_H:    return smem_perf.writes >> 32;
        case VX_CSR_MPM_SMEM_BANK_ST:     return smem_perf.bank_stalls & 0xffffffff; 
        case VX_CSR_MPM_SMEM_BANK_ST_H:   return smem_perf.bank_stalls >> 32; 
        }
      } break;
      default: {
        std::cout << std::dec << "Error: invalid MPM CLASS: value=" << perf_class << std::endl;
        std::abort();
      } break;
      }
    } else {
      std::cout << std::hex << "Error: invalid CSR read addr=0x" << addr << std::endl;
      std::abort();
    }
  }
  return 0;
}

void Core::set_csr(uint32_t addr, uint32_t value, uint32_t tid, uint32_t wid) {
  __unused (tid);
  switch (addr) {
  case VX_CSR_FFLAGS:
    fcsrs_.at(wid) = (fcsrs_.at(wid) & ~0x1F) | (value & 0x1F);
    break;
  case VX_CSR_FRM:
    fcsrs_.at(wid) = (fcsrs_.at(wid) & ~0xE0) | (value << 5);
    break;
  case VX_CSR_FCSR:
    fcsrs_.at(wid) = value & 0xff;
    break;
  case VX_CSR_SATP:
  case VX_CSR_MSTATUS:
  case VX_CSR_MEDELEG:
  case VX_CSR_MIDELEG:
  case VX_CSR_MIE:
  case VX_CSR_MTVEC:
  case VX_CSR_MEPC:
  case VX_CSR_PMPCFG0:
  case VX_CSR_PMPADDR0:
  case VX_CSR_MNSTATUS:
    break;
  default:
    {
      std::cout << std::hex << "Error: invalid CSR write addr=0x" << addr << ", value=0x" << value << std::endl;
      std::abort();
    }
  }
}

void Core::trigger_ecall() {
  active_warps_.reset();
  exited_ = true;
}

void Core::trigger_ebreak() {
  active_warps_.reset();
  exited_ = true;
}

bool Core::check_exit(Word* exitcode, bool riscv_test) const {
  if (exited_) {
    Word ec = warps_.at(0)->getIRegValue(3);
    if (riscv_test) {
      *exitcode = (1 - ec);
    } else {
      *exitcode = ec;
    }
    return true;
  }
  return false;
}

bool Core::running() const {
  return (committed_instrs_ != issued_instrs_);
}

void Core::resume() {
  stalled_warps_.reset();
}

void Core::attach_ram(RAM* ram) {
  // bind RAM to memory unit
#if (XLEN == 64)
  mmu_.attach(*ram, 0, 0xFFFFFFFFFFFFFFFF);
#else
  mmu_.attach(*ram, 0, 0xFFFFFFFF);
#endif
}
