#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "types.h"
#include "archdef.h"
#include "mem.h"
#include "decode.h"
#include "core.h"
#include "debug.h"

using namespace vortex;

Core::Core(const SimContext& ctx, const ArchDef &arch, Word id)
    : SimObject(ctx, "Core")
    , id_(id)
    , arch_(arch)
    , decoder_(arch)
    , mmu_(0, arch.wsize(), true)
    , shared_mem_(4096)
    , warps_(arch.num_warps())
    , barriers_(arch.num_barriers(), 0)
    , csrs_(arch.num_csrs(), 0)
    , fcsrs_(arch.num_warps(), 0)
    , ibuffers_(arch.num_warps(), IBUF_SIZE)
    , scoreboard_(arch_) 
    , exe_units_((int)ExeType::MAX)
    , icache_(Cache::Create("Icache", CacheConfig{
        log2ceil(ICACHE_SIZE),  // C
        log2ceil(L1_BLOCK_SIZE),// B
        2,                      // W
        0,                      // A
        32,                     // address bits    
        1,                      // number of banks
        1,                      // number of ports
        1,                      // request size   
        true,                   // write-throught
        0,                      // victim size
        NUM_WARPS,              // mshr
        2,                      // pipeline latency
      }))
    , dcache_(Cache::Create("Dcache", CacheConfig{
        log2ceil(DCACHE_SIZE),  // C
        log2ceil(L1_BLOCK_SIZE),// B
        2,                      // W
        0,                      // A
        32,                     // address bits    
        DCACHE_NUM_BANKS,       // number of banks
        DCACHE_NUM_PORTS,       // number of ports
        (uint8_t)arch.num_threads(), // request size   
        true,                   // write-throught
        0,                      // victim size
        DCACHE_MSHR_SIZE,       // mshr
        2,                      // pipeline latency
      }))
    , l1_mem_switch_(Switch<MemReq, MemRsp>::Create("l1_arb", ArbiterType::Priority, 2))
    , fetch_stage_("fetch")
    , decode_stage_("decode")
    , issue_stage_("issue")
    , execute_stage_("execute")
    , commit_stage_("writeback")
    , pending_icache_(arch_.num_warps())
    , stalled_warps_(0)
    , last_schedule_wid_(0)
    , issued_instrs_(0)
    , committed_instrs_(0)
    , ebreak_(false)   
    , stats_insts_(0)
    , stats_loads_(0)
    , stats_stores_(0)
    , MemRspPort(this)
    , MemReqPort(this)    
{  
  for (int i = 0; i < arch_.num_warps(); ++i) {
    warps_.at(i) = std::make_shared<Warp>(this, i);
  }

  // register execute units
  exe_units_.at((int)ExeType::NOP) = std::make_shared<NopUnit>(this);
  exe_units_.at((int)ExeType::ALU) = std::make_shared<AluUnit>(this);
  exe_units_.at((int)ExeType::LSU) = std::make_shared<LsuUnit>(this);
  exe_units_.at((int)ExeType::CSR) = std::make_shared<CsrUnit>(this);
  exe_units_.at((int)ExeType::FPU) = std::make_shared<FpuUnit>(this);  
  exe_units_.at((int)ExeType::GPU) = std::make_shared<GpuUnit>(this);

  // connect l1 switch
  icache_->MemReqPort.bind(&l1_mem_switch_->ReqIn[0]);
  dcache_->MemReqPort.bind(&l1_mem_switch_->ReqIn[1]);
  l1_mem_switch_->RspOut[0].bind(&icache_->MemRspPort);  
  l1_mem_switch_->RspOut[1].bind(&dcache_->MemRspPort);
  this->MemRspPort.bind(&l1_mem_switch_->RspIn);
  l1_mem_switch_->ReqOut.bind(&this->MemReqPort);

  // activate warp0
  warps_.at(0)->setTmask(0, true);
}

Core::~Core() {
  for (auto& buf : print_bufs_) {
    auto str = buf.second.str();
    if (!str.empty()) {
      std::cout << "#" << buf.first << ": " << str << std::endl;
    }
  }
}

void Core::attach_ram(RAM* ram) {
  // bind RAM to memory unit
  mmu_.attach(*ram, 0, 0xFFFFFFFF);    
}

void Core::step(uint64_t cycle) {
  this->commit(cycle);
  this->execute(cycle);
  this->issue(cycle);
  this->decode(cycle);
  this->fetch(cycle);

  DPN(2, std::flush);
}

void Core::warp_scheduler(uint64_t cycle) {
  __unused (cycle);

  bool foundSchedule = false;
  int scheduled_warp = last_schedule_wid_;

  // round robin scheduling
  for (size_t wid = 0; wid < warps_.size(); ++wid) {    
    scheduled_warp = (scheduled_warp + 1) % warps_.size();
    bool warp_active  = warps_.at(scheduled_warp)->active();
    bool warp_stalled = stalled_warps_.test(scheduled_warp); 
    if (warp_active && !warp_stalled) {      
      last_schedule_wid_ = scheduled_warp;
      foundSchedule = true;
      break;
    }
  }

  if (!foundSchedule)
    return;  

  // suspend warp until decode
  stalled_warps_.set(scheduled_warp);

  auto& warp = warps_.at(scheduled_warp);  
  stats_insts_ += warp->getActiveThreads();
  
  pipeline_state_t state;
  state.clear();
  state.id = (issued_instrs_++ * arch_.num_cores()) + id_;

  warp->eval(&state);

  DT(3, cycle, "pipeline-schedule: " << state);

  // advance to fetch stage  
  fetch_stage_.push(state);
}

void Core::fetch(uint64_t cycle) {
  // handle icache reponse
  {
    MemRsp mem_rsp;
    if (icache_->CoreRspPorts.at(0).read(&mem_rsp)){
      pipeline_state_t state;
      pending_icache_.remove(mem_rsp.tag, &state);
      auto latency = (SimPlatform::instance().cycles() - state.icache_latency);
      state.icache_latency = latency;
      decode_stage_.push(state);
      DT(3, cycle, "icache-rsp: addr=" << std::hex << state.PC << ", tag=" << mem_rsp.tag << ", " << state);
    }
  }

  // send icache request
  {
    pipeline_state_t state;
    if (fetch_stage_.try_pop(&state)) {
      state.icache_latency = SimPlatform::instance().cycles();
      MemReq mem_req;
      mem_req.addr  = state.PC;
      mem_req.write = false;
      mem_req.tag   = pending_icache_.allocate(state);    
      icache_->CoreReqPorts.at(0).send(mem_req, 1);
      DT(3, cycle, "icache-req: addr=" << std::hex << mem_req.addr << ", tag=" << mem_req.tag << ", " << state);
    }
  }  

  // schedule next warp
  this->warp_scheduler(cycle);  
}

void Core::decode(uint64_t cycle) {
  __unused (cycle);

  pipeline_state_t state;
  if (!decode_stage_.try_pop(&state))
    return;    
  
  // release warp
  if (!state.stall_warp) {
    stalled_warps_.reset(state.wid);
  }

  DT(3, cycle, "pipeline-decode: " << state);
  
  // advance to issue stage
  issue_stage_.push(state);
}

void Core::issue(uint64_t cycle) {
  __unused (cycle);

  if (!issue_stage_.empty()) {
    // insert to ibuffer 
    auto& state = issue_stage_.top();
    auto& ibuffer = ibuffers_.at(state.wid);
    if (ibuffer.full()) {
      DT(3, cycle, "*** ibuffer-stall: " << state);
    } else {
      ibuffer.push(state);
      issue_stage_.pop();
    }
  }
    
  // issue ibuffer instructions
  for (auto& ibuffer : ibuffers_) {
    if (ibuffer.empty())
      continue;

    auto& state = ibuffer.top();

    // check scoreboard
    if (scoreboard_.in_use(state)) {
      DTH(3, cycle, "*** scoreboard-stall: dependents={");
      auto owners = scoreboard_.owners(state);
      for (uint32_t i = 0, n = owners.size(); i < n; ++i) {
        if (i) DTN(3, ", ");
        DTN(3, "#" << owners.at(i));  
      }
      DTN(3, "}, " << state << std::endl);
      continue;
    }

    DT(3, cycle, "pipeline-issue: " << state);

    // update scoreboard
    scoreboard_.reserve(state);

    // advance to execute stage
    execute_stage_.push(state);

    ibuffer.pop();
    break;
  }
}

void Core::execute(uint64_t cycle) {
  // process stage inputs
  if (!execute_stage_.empty()) {
    auto& state = execute_stage_.top();
    auto& exe_unit = exe_units_.at((int)state.exe_type);
    exe_unit->push_input(state);
    execute_stage_.pop();
    DT(3, cycle, "pipeline-execute: " << state);
  }

  // advance execute units
  for (auto& exe_unit : exe_units_) {
    exe_unit->step(cycle);
  }  
  
  // commit completed instructions
  for (auto& exe_unit : exe_units_) {
    pipeline_state_t state;
    if (exe_unit->pop_output(&state)) {
      if (state.stall_warp) {
        stalled_warps_.reset(state.wid);
      }
      // advance to commit stage
      commit_stage_.push(state);   
    }
  }
}

void Core::commit(uint64_t cycle) {
  __unused (cycle);
  
  pipeline_state_t state;
  if (!commit_stage_.try_pop(&state))
    return;

  DT(3, cycle, "pipeline-commit: " << state);

  // update scoreboard
  scoreboard_.release(state);

  assert(committed_instrs_ <= issued_instrs_);
  ++committed_instrs_;
}

bool Core::running() const {
  return (committed_instrs_ != issued_instrs_);
}

Word Core::get_csr(Addr addr, int tid, int wid) {
  if (addr == CSR_FFLAGS) {
    return fcsrs_.at(wid) & 0x1F;
  } else if (addr == CSR_FRM) {
    return (fcsrs_.at(wid) >> 5);
  } else if (addr == CSR_FCSR) {
    return fcsrs_.at(wid);
  } else if (addr == CSR_WTID) {
    // Warp threadID
    return tid;
  } else if (addr == CSR_LTID) {
    // Core threadID
    return tid + (wid * arch_.num_threads());
  } else if (addr == CSR_GTID) {
    // Processor threadID
    return tid + (wid * arch_.num_threads()) + 
              (arch_.num_threads() * arch_.num_warps() * id_);
  } else if (addr == CSR_LWID) {
    // Core warpID
    return wid;
  } else if (addr == CSR_GWID) {
    // Processor warpID        
    return wid + (arch_.num_warps() * id_);
  } else if (addr == CSR_GCID) {
    // Processor coreID
    return id_;
  } else if (addr == CSR_TMASK) {
    // Processor coreID
    return warps_.at(wid)->getTmask();
  } else if (addr == CSR_NT) {
    // Number of threads per warp
    return arch_.num_threads();
  } else if (addr == CSR_NW) {
    // Number of warps per core
    return arch_.num_warps();
  } else if (addr == CSR_NC) {
    // Number of cores
    return arch_.num_cores();
  } else if (addr == CSR_MINSTRET) {
    // NumInsts
    return stats_insts_;
  } else if (addr == CSR_MINSTRET_H) {
    // NumInsts
    return (Word)(stats_insts_ >> 32);
  } else if (addr == CSR_MCYCLE) {
    // NumCycles
    return (Word)SimPlatform::instance().cycles();
  } else if (addr == CSR_MCYCLE_H) {
    // NumCycles
    return (Word)(SimPlatform::instance().cycles() >> 32);
  } else {
    return csrs_.at(addr);
  }
}

void Core::set_csr(Addr addr, Word value, int /*tid*/, int wid) {
  if (addr == CSR_FFLAGS) {
    fcsrs_.at(wid) = (fcsrs_.at(wid) & ~0x1F) | (value & 0x1F);
  } else if (addr == CSR_FRM) {
    fcsrs_.at(wid) = (fcsrs_.at(wid) & ~0xE0) | (value << 5);
  } else if (addr == CSR_FCSR) {
    fcsrs_.at(wid) = value & 0xff;
  } else {
    csrs_.at(addr) = value;
  }
}

void Core::barrier(int bar_id, int count, int warp_id) {
  auto& barrier = barriers_.at(bar_id);
  barrier.set(warp_id);
  if (barrier.count() < (size_t)count)    
    return;
  for (int i = 0; i < arch_.num_warps(); ++i) {
    if (barrier.test(i)) {
      warps_.at(i)->activate();
    }
  }
  barrier.reset();
}

Word Core::icache_read(Addr addr, Size size) {
  Word data;
  mmu_.read(&data, addr, size, 0);
  return data;
}

Word Core::dcache_read(Addr addr, Size size) {
  ++stats_loads_;
  Word data = 0;
#ifdef SM_ENABLE
  if ((addr >= (SMEM_BASE_ADDR - SMEM_SIZE))
   && ((addr + 3) < SMEM_BASE_ADDR)) {
     shared_mem_.read(&data, addr & (SMEM_SIZE-1), size);
     return data;
  }
#endif
  mmu_.read(&data, addr, size, 0);
  return data;
}

void Core::dcache_write(Addr addr, Word data, Size size) {
  ++stats_stores_;
#ifdef SM_ENABLE
  if ((addr >= (SMEM_BASE_ADDR - SMEM_SIZE))
   && ((addr + 3) < SMEM_BASE_ADDR)) {
     shared_mem_.write(&data, addr & (SMEM_SIZE-1), size);
     return;
  }
#endif
  if (addr >= IO_COUT_ADDR 
   && addr <= (IO_COUT_ADDR + IO_COUT_SIZE - 1)) {
     this->writeToStdOut(addr, data);
     return;
  }
  mmu_.write(&data, addr, size, 0);
}

void Core::printStats() const {
  std::cout << "Cycles: " << SimPlatform::instance().cycles() << std::endl
            << "Insts : " << stats_insts_ << std::endl
            << "Loads : " << stats_loads_ << std::endl
            << "Stores: " << stats_stores_ << std::endl;
}

void Core::writeToStdOut(Addr addr, Word data) {
  uint32_t tid = (addr - IO_COUT_ADDR) & (IO_COUT_SIZE-1);
  auto& ss_buf = print_bufs_[tid];
  char c = (char)data;
  ss_buf << c;
  if (c == '\n') {
    std::cout << std::dec << "#" << tid << ": " << ss_buf.str() << std::flush;
    ss_buf.str("");
  }
}

void Core::trigger_ebreak() {
  ebreak_ = true;
}

bool Core::check_ebreak() const {
  return ebreak_;
}