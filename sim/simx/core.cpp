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
#include "debug.h"
#include "constants.h"

using namespace vortex;

Core::Core(const SimContext& ctx, 
           uint32_t id, 
           const Arch &arch, 
           const DCRS &dcrs,
           RasterUnit::Ptr raster_unit,
           RopUnit::Ptr rop_unit)
    : SimObject(ctx, "core")
    , MemRspPort(this)
    , MemReqPort(this)
    , id_(id)
    , arch_(arch)
    , dcrs_(dcrs)
    , decoder_(arch)
    , mmu_(0, arch.wsize(), true)
    , warps_(arch.num_warps())
    , barriers_(arch.num_barriers(), 0)
    , fcsrs_(arch.num_warps(), 0)
    , ibuffers_(arch.num_warps(), IBUF_SIZE)
    , scoreboard_(arch_) 
    , exe_units_((int)ExeType::MAX)
    , icache_(CacheSim::Create("icache", CacheSim::Config{
        log2ceil(ICACHE_SIZE),  // C
        log2ceil(L1_BLOCK_SIZE),// B
        log2ceil(sizeof(uint32_t)), // W
        0,                      // A
        32,                     // address bits    
        1,                      // number of banks
        1,                      // number of ports
        1,                      // request size   
        true,                   // write-through
        false,                  // write response
        0,                      // victim size
        (uint8_t)arch.num_warps(), // mshr
        2,                      // pipeline latency
      }))
    , dcache_(CacheSim::Create("dcache", CacheSim::Config{
        log2ceil(DCACHE_SIZE),  // C
        log2ceil(L1_BLOCK_SIZE),// B
        log2ceil(sizeof(Word)), // W
        0,                      // A
        32,                     // address bits    
        DCACHE_NUM_BANKS,       // number of banks
        DCACHE_NUM_PORTS,       // number of ports
        (uint8_t)arch.num_threads(), // request size   
        true,                   // write-through
        false,                  // write response
        0,                      // victim size
        DCACHE_MSHR_SIZE,       // mshr
        4,                      // pipeline latency
      }))
    , tcache_(CacheSim::Create("tcache", CacheSim::Config{
        log2ceil(TCACHE_SIZE),  // C
        log2ceil(L1_BLOCK_SIZE),// B
        log2ceil(sizeof(uint32_t)), // W
        2,                      // A
        32,                     // address bits    
        (uint8_t)arch.num_threads(), // number of banks
        1,                      // number of ports
        (uint8_t)arch.num_threads(), // request size   
        false,                  // write-through
        false,                  // write response
        0,                      // victim size
        DCACHE_MSHR_SIZE,       // mshr
        4,                      // pipeline latency
      }))
    , sharedmem_(SharedMem::Create("shared_mem", SharedMem::Config{
        uint32_t(SMEM_LOCAL_SIZE) * arch.num_warps() * arch.num_threads(),
        SMEM_LOCAL_SIZE,
        arch.num_threads(), 
        arch.num_threads(),
        log2ceil(STACK_SIZE),
        1,
        false
      }))
    , l1_mem_switch_(Switch<MemReq, MemRsp>::Create("l1_arb", ArbiterType::Priority, 3))
    , fetch_latch_("fetch")
    , decode_latch_("decode")
    , pending_icache_(arch_.num_warps())
{  
  for (uint32_t i = 0; i < arch_.num_warps(); ++i) {
    warps_.at(i) = std::make_shared<Warp>(this, i);
  }

  tex_unit_ = TexUnit::Create("tex_unit", TexUnit::Config{
    2, // address latency
    6, // sampler latency
  }, this);
  raster_svc_ = RasterSvc::Create("raster_svc", this, raster_unit);
  rop_svc_ = RopSvc::Create("rop_svc", this, rop_unit);

  // register execute units
  exe_units_.at((int)ExeType::NOP) = SimPlatform::instance().create_object<NopUnit>(this);
  exe_units_.at((int)ExeType::ALU) = SimPlatform::instance().create_object<AluUnit>(this);
  exe_units_.at((int)ExeType::LSU) = SimPlatform::instance().create_object<LsuUnit>(this);
  exe_units_.at((int)ExeType::CSR) = SimPlatform::instance().create_object<CsrUnit>(this);
  exe_units_.at((int)ExeType::FPU) = SimPlatform::instance().create_object<FpuUnit>(this);  
  exe_units_.at((int)ExeType::GPU) = SimPlatform::instance().create_object<GpuUnit>(this);

  // connect l1 switch
  icache_->MemReqPort.bind(&l1_mem_switch_->ReqIn[0]);
  dcache_->MemReqPort.bind(&l1_mem_switch_->ReqIn[1]);
  tcache_->MemReqPort.bind(&l1_mem_switch_->ReqIn[2]);
  l1_mem_switch_->RspOut[0].bind(&icache_->MemRspPort);  
  l1_mem_switch_->RspOut[1].bind(&dcache_->MemRspPort);
  l1_mem_switch_->RspOut[2].bind(&tcache_->MemRspPort);
  this->MemRspPort.bind(&l1_mem_switch_->RspIn);
  l1_mem_switch_->ReqOut.bind(&this->MemReqPort);

  // memory perf callbacks
  MemReqPort.tx_callback([&](const MemReq& req, uint64_t cycle){
    __unused (cycle);
    perf_stats_.mem_reads   += !req.write;
    perf_stats_.mem_writes  += req.write;
    perf_mem_pending_reads_ += !req.write;    
  });
  MemRspPort.tx_callback([&](const MemRsp&, uint64_t cycle){
    __unused (cycle);
    --perf_mem_pending_reads_;
  });

  this->reset();
}

Core::~Core() {
  this->cout_flush();
}

void Core::reset() {
  for (auto& warp : warps_) {
    warp->clear();
  }
  warps_.at(0)->setTmask(0, true);
  active_warps_ = 1;

  for (auto& exe_unit : exe_units_) {
    exe_unit->reset();
  }

  tex_unit_->reset();
  
  for ( auto& barrier : barriers_) {
    barrier.reset();
  }
  
  for (auto& fcsr : fcsrs_) {
    fcsr = 0;
  }
  
  for (auto& ibuf : ibuffers_) {
    ibuf.clear();
  }

  scoreboard_.clear();
  fetch_latch_.clear();
  decode_latch_.clear();
  pending_icache_.clear();
  stalled_warps_.reset();
  issued_instrs_ = 0;
  committed_instrs_ = 0;
  ecall_ = false;
  ebreak_ = false;
  perf_mem_pending_reads_ = 0;
  perf_stats_ = PerfStats();
}

void Core::attach_ram(RAM* ram) {
  // bind RAM to memory unit
  mmu_.attach(*ram, 0, 0xFFFFFFFF);    
}

void Core::cout_flush() {
  for (auto& buf : print_bufs_) {
    auto str = buf.second.str();
    if (!str.empty()) {
      std::cout << "#" << buf.first << ": " << str << std::endl;
    }
  }
}

void Core::tick() {
  this->commit();
  this->execute();
  this->decode();
  this->fetch();
  this->schedule();

  // update perf counter  
  perf_stats_.mem_latency += perf_mem_pending_reads_;

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

  if (scheduled_warp == -1)
    return;

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
  // handle icache reponse
  auto& icache_rsp_port = icache_->CoreRspPorts.at(0);      
  if (!icache_rsp_port.empty()){
    auto& mem_rsp = icache_rsp_port.front();
    auto trace = pending_icache_.at(mem_rsp.tag);
    decode_latch_.push(trace);
    DT(3, "icache-rsp: addr=" << std::hex << trace->PC << ", tag=" << mem_rsp.tag << ", " << *trace);
    pending_icache_.release(mem_rsp.tag);
    icache_rsp_port.pop();
  }

  // send icache request
  if (!fetch_latch_.empty()) {
    auto trace = fetch_latch_.front();
    MemReq mem_req;
    mem_req.addr  = trace->PC;
    mem_req.write = false;
    mem_req.tag   = pending_icache_.allocate(trace);    
    mem_req.core_id = trace->cid;
    mem_req.uuid = trace->uuid;
    icache_->CoreReqPorts.at(0).send(mem_req, 1);    
    DT(3, "icache-req: addr=" << std::hex << mem_req.addr << ", tag=" << mem_req.tag << ", " << *trace);
    fetch_latch_.pop();
  }    
}

void Core::decode() {
  if (decode_latch_.empty())
    return;

  auto trace = decode_latch_.front();

  // check ibuffer capacity
  auto& ibuffer = ibuffers_.at(trace->wid);
  if (ibuffer.full()) {
    if (!trace->suspend()) {
      DT(3, "*** ibuffer-stall: " << *trace);
    }
    ++perf_stats_.ibuf_stalls;
    return;
  } else {
    trace->resume();
  }
  
  // release warp
  if (!trace->fetch_stall) {
    stalled_warps_.reset(trace->wid);
  }

  // update perf counters
  uint32_t active_threads = trace->tmask.count();
  if (trace->exe_type == ExeType::LSU && trace->lsu_type == LsuType::LOAD)
    perf_stats_.loads += active_threads;
  if (trace->exe_type == ExeType::LSU && trace->lsu_type == LsuType::STORE) 
    perf_stats_.stores += active_threads;
  if (trace->exe_type == ExeType::ALU && trace->alu_type == AluType::BRANCH) 
    perf_stats_.branches += active_threads;

  DT(3, "pipeline-decode: " << *trace);

  // insert to ibuffer 
  ibuffer.push(trace);

  decode_latch_.pop();
}

void Core::execute() {    
  // issue ibuffer instructions
  for (auto& ibuffer : ibuffers_) {
    if (ibuffer.empty())
      continue;

    auto trace = ibuffer.top();

    // check scoreboard
    if (scoreboard_.in_use(trace)) {
      if (!trace->suspend()) {
        DTH(3, "*** scoreboard-stall: dependents={");
        auto uses = scoreboard_.get_uses(trace);
        for (uint32_t i = 0, n = uses.size(); i < n; ++i) {
          auto& use = uses.at(i);
          __unused (use);
          if (i) DTN(3, ", ");      
          DTN(3, use.type << use.reg << "(#" << use.owner << ")");
        }
        DTN(3, "}, " << *trace << std::endl);
      }
      ++perf_stats_.scrb_stalls;
      continue;
    } else {
      trace->resume();
    }

    // update scoreboard
    scoreboard_.reserve(trace);

    DT(3, "pipeline-issue: " << *trace);

    // push to execute units
    auto& exe_unit = exe_units_.at((int)trace->exe_type);
    exe_unit->Input.send(trace, 1);

    ibuffer.pop();
    break;
  }
}

void Core::commit() {  
  // commit completed instructions
  bool wb = false;
  for (auto& exe_unit : exe_units_) {
    if (!exe_unit->Output.empty()) {
      auto trace = exe_unit->Output.front();    

      // allow only one commit that updates registers
      if (trace->wb && wb)
        continue;        
      wb |= trace->wb;

      // advance to commit stage
      DT(3, "pipeline-commit: " << *trace);

      // update scoreboard
      scoreboard_.release(trace);

      assert(committed_instrs_ <= issued_instrs_);
      ++committed_instrs_;

      perf_stats_.instrs += trace->tmask.count();

      // delete the trace
      delete trace;

      exe_unit->Output.pop();
    }
  }
}

WarpMask Core::wspawn(uint32_t num_warps, uint32_t nextPC) {
  WarpMask ret(1);
  uint32_t active_warps = std::min<uint32_t>(num_warps, arch_.num_warps());
  DP(3, "*** Activate " << (active_warps-1) << " warps at PC: " << std::hex << nextPC);
  for (uint32_t i = 1; i < active_warps; ++i) {
    auto warp = warps_.at(i);
    warp->setPC(nextPC);
    warp->setTmask(0, true);
    ret.set(i); 
  }
  return ret;
}

WarpMask Core::barrier(uint32_t bar_id, uint32_t count, uint32_t warp_id) {
  WarpMask ret(0);
  auto& barrier = barriers_.at(bar_id);
  barrier.set(warp_id);
  if (barrier.count() < (size_t)count) {
    warps_.at(warp_id)->suspend();
    DP(3, "*** Suspend warp #" << warp_id << " at barrier #" << bar_id);
    return ret;
  }
  for (uint32_t i = 0; i < arch_.num_warps(); ++i) {
    if (barrier.test(i)) {
      DP(3, "*** Resume warp #" << i << " at barrier #" << bar_id);
      warps_.at(i)->activate();
      ret.set(i);
    }
  }
  barrier.reset();
  return ret;
}

void Core::icache_read(void *data, uint64_t addr, uint32_t size) {
  mmu_.read(data, addr, size, 0);
}

AddrType Core::get_addr_type(uint64_t addr) {  
  if (addr >= IO_BASE_ADDR) {
     return AddrType::IO;
  }
  if (SM_ENABLED) {
    // check if address is a stack address
    uint32_t total_threads    = arch_.num_cores() * arch_.num_warps() * arch_.num_threads();
    uint64_t total_stack_size = STACK_SIZE * total_threads;
    uint64_t stack_end        = STACK_BASE_ADDR - total_stack_size;
    if (addr >= stack_end && addr < STACK_BASE_ADDR) {     
      // check if address is within shared memory region
      uint32_t offset = addr % STACK_SIZE;
      if (offset >= (STACK_SIZE - SMEM_LOCAL_SIZE)) {
        return AddrType::Shared;
      }
    }
  }
  return AddrType::Global;
}

void Core::dcache_read(void *data, uint64_t addr, uint32_t size) {  
  auto type = this->get_addr_type(addr);
  if (type == AddrType::Shared) {
    sharedmem_->read(data, addr, size);
  } else {  
    mmu_.read(data, addr, size, 0);
  }

  DPH(2, "Mem Read: addr=0x" << std::hex << addr << ", data=0x" << ByteStream(data, size) << " (size=" << size << ", type=" << type << ")" << std::endl);
}

void Core::dcache_write(const void* data, uint64_t addr, uint32_t size) {  
  auto type = this->get_addr_type(addr);
  if (addr >= IO_COUT_ADDR 
   && addr <= (IO_COUT_ADDR + IO_COUT_SIZE - 1)) {
     this->writeToStdOut(data, addr, size);
  } else {
    if (type == AddrType::Shared) {
      sharedmem_->write(data, addr, size);
    } else {
      mmu_.write(data, addr, size, 0);
    }
  }

  DPH(2, "Mem Write: addr=0x" << std::hex << addr << ", data=0x" << ByteStream(data, size) << " (size=" << size << ", type=" << type << ")" << std::endl);
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

uint32_t Core::get_csr(uint32_t addr, uint32_t tid, uint32_t wid) {
  switch (addr) {
  case CSR_SATP:
  case CSR_PMPCFG0:
  case CSR_PMPADDR0:
  case CSR_MSTATUS:
  case CSR_MISA:
  case CSR_MEDELEG:
  case CSR_MIDELEG:
  case CSR_MIE:
  case CSR_MTVEC:
  case CSR_MEPC:
    return 0;

  case CSR_FFLAGS:
    return fcsrs_.at(wid) & 0x1F;
  case CSR_FRM:
    return (fcsrs_.at(wid) >> 5);
  case CSR_FCSR:
    return fcsrs_.at(wid);
  case CSR_WTID:
    // Warp threadID
    return tid;
  case CSR_LTID:
    // Core threadID
    return tid + (wid * arch_.num_threads());
  case CSR_GTID:
    // Processor threadID
    return tid + (wid * arch_.num_threads()) + 
        (arch_.num_threads() * arch_.num_warps() * id_);
  case CSR_LWID:
    // Core warpID
    return wid;
  case CSR_GWID:
    // Processor warpID        
    return wid + (arch_.num_warps() * id_);
  case CSR_GCID:
    // Processor coreID
    return id_;
  case CSR_TMASK:
    // Processor coreID
    return warps_.at(wid)->getTmask();
  case CSR_NT:
    // Number of threads per warp
    return arch_.num_threads();
  case CSR_NW:
    // Number of warps per core
    return arch_.num_warps();
  case CSR_NC:
    // Number of cores
    return arch_.num_cores();
  case CSR_MINSTRET:
    // NumInsts
    return perf_stats_.instrs & 0xffffffff;
  case CSR_MINSTRET_H:
    // NumInsts
    return (uint32_t)(perf_stats_.instrs >> 32);
  case CSR_MCYCLE:
    // NumCycles
    return (uint32_t)SimPlatform::instance().cycles();
  case CSR_MCYCLE_H:
    // NumCycles
    return (uint32_t)(SimPlatform::instance().cycles() >> 32);
  default:
    if ((addr >= CSR_MPM_BASE && addr < (CSR_MPM_BASE + 32))
     || (addr >= CSR_MPM_BASE_H && addr < (CSR_MPM_BASE_H + 32))) {
      // user-defined MPM CSRs
      auto perf_class = dcrs_.base_dcrs.read(DCR_MPM_CLASS);
      switch (perf_class) {
      case DCR_MPM_CLASS_CORE: {
        switch (addr) {
        case CSR_MPM_IBUF_ST:
          return perf_stats_.ibuf_stalls & 0xffffffff; 
        case CSR_MPM_IBUF_ST_H:
          return perf_stats_.ibuf_stalls >> 32; 
        case CSR_MPM_SCRB_ST:
          return perf_stats_.scrb_stalls & 0xffffffff; 
        case CSR_MPM_SCRB_ST_H:
          return perf_stats_.scrb_stalls >> 32; 
        case CSR_MPM_ALU_ST:
          return perf_stats_.alu_stalls & 0xffffffff; 
        case CSR_MPM_ALU_ST_H:
          return perf_stats_.alu_stalls >> 32; 
        case CSR_MPM_LSU_ST:
          return perf_stats_.lsu_stalls & 0xffffffff; 
        case CSR_MPM_LSU_ST_H:
          return perf_stats_.lsu_stalls >> 32; 
        case CSR_MPM_CSR_ST:
          return perf_stats_.csr_stalls & 0xffffffff; 
        case CSR_MPM_CSR_ST_H:
          return perf_stats_.csr_stalls >> 32; 
        case CSR_MPM_FPU_ST:
          return perf_stats_.fpu_stalls & 0xffffffff; 
        case CSR_MPM_FPU_ST_H:
          return perf_stats_.fpu_stalls >> 32; 
        case CSR_MPM_GPU_ST:
          return perf_stats_.gpu_stalls & 0xffffffff; 
        case CSR_MPM_GPU_ST_H:
          return perf_stats_.gpu_stalls >> 32; 
        
        case CSR_MPM_LOADS:
          return perf_stats_.loads & 0xffffffff; 
        case CSR_MPM_LOADS_H:
          return perf_stats_.loads >> 32; 
        case CSR_MPM_STORES:
          return perf_stats_.stores & 0xffffffff; 
        case CSR_MPM_STORES_H:
          return perf_stats_.stores >> 32;
        case CSR_MPM_BRANCHES:
          return perf_stats_.branches & 0xffffffff; 
        case CSR_MPM_BRANCHES_H:
          return perf_stats_.branches >> 32; 

        case CSR_MPM_ICACHE_READS:
          return icache_->perf_stats().reads & 0xffffffff; 
        case CSR_MPM_ICACHE_READS_H:
          return icache_->perf_stats().reads >> 32; 
        case CSR_MPM_ICACHE_MISS_R:
          return icache_->perf_stats().read_misses & 0xffffffff;
        case CSR_MPM_ICACHE_MISS_R_H:
          return icache_->perf_stats().read_misses >> 32;
        
        case CSR_MPM_DCACHE_READS:
          return dcache_->perf_stats().reads & 0xffffffff; 
        case CSR_MPM_DCACHE_READS_H:
          return dcache_->perf_stats().reads >> 32; 
        case CSR_MPM_DCACHE_WRITES:
          return dcache_->perf_stats().writes & 0xffffffff; 
        case CSR_MPM_DCACHE_WRITES_H:
          return dcache_->perf_stats().writes >> 32; 
        case CSR_MPM_DCACHE_MISS_R:
          return dcache_->perf_stats().read_misses & 0xffffffff; 
        case CSR_MPM_DCACHE_MISS_R_H:
          return dcache_->perf_stats().read_misses >> 32; 
        case CSR_MPM_DCACHE_MISS_W:
          return dcache_->perf_stats().write_misses & 0xffffffff; 
        case CSR_MPM_DCACHE_MISS_W_H:
          return dcache_->perf_stats().write_misses >> 32; 
        case CSR_MPM_DCACHE_BANK_ST:
          return dcache_->perf_stats().bank_stalls & 0xffffffff; 
        case CSR_MPM_DCACHE_BANK_ST_H:
          return dcache_->perf_stats().bank_stalls >> 32;
        case CSR_MPM_DCACHE_MSHR_ST:
          return dcache_->perf_stats().mshr_stalls & 0xffffffff; 
        case CSR_MPM_DCACHE_MSHR_ST_H:
          return dcache_->perf_stats().mshr_stalls >> 32;
        
        case CSR_MPM_SMEM_READS:
          return sharedmem_->perf_stats().reads & 0xffffffff;
        case CSR_MPM_SMEM_READS_H:
          return sharedmem_->perf_stats().reads >> 32;
        case CSR_MPM_SMEM_WRITES:
          return sharedmem_->perf_stats().writes & 0xffffffff;
        case CSR_MPM_SMEM_WRITES_H:
          return sharedmem_->perf_stats().writes >> 32;
        case CSR_MPM_SMEM_BANK_ST:
          return sharedmem_->perf_stats().bank_stalls & 0xffffffff; 
        case CSR_MPM_SMEM_BANK_ST_H:
          return sharedmem_->perf_stats().bank_stalls >> 32; 

        case CSR_MPM_MEM_READS:
          return perf_stats_.mem_reads & 0xffffffff; 
        case CSR_MPM_MEM_READS_H:
          return perf_stats_.mem_reads >> 32; 
        case CSR_MPM_MEM_WRITES:
          return perf_stats_.mem_writes & 0xffffffff; 
        case CSR_MPM_MEM_WRITES_H:
          return perf_stats_.mem_writes >> 32; 
        case CSR_MPM_MEM_LAT:
          return perf_stats_.mem_latency & 0xffffffff; 
        case CSR_MPM_MEM_LAT_H:
          return perf_stats_.mem_latency >> 32;
        }
      } break;
      case DCR_MPM_CLASS_TEX: {
        switch (addr) {
        case CSR_MPM_TEX_READS:
          return tex_unit_->perf_stats().reads & 0xffffffff;
        case CSR_MPM_TEX_READS_H:
          return tex_unit_->perf_stats().reads >> 32;
        case CSR_MPM_TEX_LAT:
          return tex_unit_->perf_stats().latency & 0xffffffff;
        case CSR_MPM_TEX_LAT_H:
          return tex_unit_->perf_stats().latency >> 32;
        }
      } break;
      case DCR_MPM_CLASS_RASTER: {
        switch (addr) {        
        default:
          return 0;
        }
      } break;
      case DCR_MPM_CLASS_ROP: {
        switch (addr) {        
        default:
          return 0;
        }
      } break;
      default: {
        std::cout << std::dec << "Error: invalid MPM CLASS: value=" << perf_class << std::endl;
        std::abort();
      } break;
      }
    } else
  #ifdef EXT_RASTER_ENABLE
    if (addr >= CSR_RASTER_BEGIN
     && addr < CSR_RASTER_END) {
      return raster_svc_->csr_read(wid, tid, addr);
    } else
  #endif
  #ifdef EXT_ROP_ENABLE
    if (addr >= CSR_ROP_BEGIN
     && addr < CSR_ROP_END) {
      return rop_svc_->csr_read(wid, tid, addr);
    } else
  #endif  
    {
      std::cout << std::hex << "Error: invalid CSR read addr=0x" << addr << std::endl;
      std::abort();
    }
  }
  return 0;
}

void Core::set_csr(uint32_t addr, uint32_t value, uint32_t tid, uint32_t wid) {
  __unused (tid);
  switch (addr) {
  case CSR_FFLAGS:
    fcsrs_.at(wid) = (fcsrs_.at(wid) & ~0x1F) | (value & 0x1F);
    break;
  case CSR_FRM:
    fcsrs_.at(wid) = (fcsrs_.at(wid) & ~0xE0) | (value << 5);
    break;
  case CSR_FCSR:
    fcsrs_.at(wid) = value & 0xff;
    break;
  case CSR_SATP:
  case CSR_MSTATUS:
  case CSR_MEDELEG:
  case CSR_MIDELEG:
  case CSR_MIE:
  case CSR_MTVEC:
  case CSR_MEPC:
  case CSR_PMPCFG0:
  case CSR_PMPADDR0:
    break;
  default:
  #ifdef EXT_RASTER_ENABLE
    if (addr >= CSR_RASTER_BEGIN
     && addr < CSR_RASTER_END) {
      raster_svc_->csr_write(wid, tid, addr, value);
    } else
  #endif
  #ifdef EXT_ROP_ENABLE
    if (addr >= CSR_ROP_BEGIN
     && addr < CSR_ROP_END) {
      rop_svc_->csr_write(wid, tid, addr, value);
    } else
  #endif
    {
      std::cout << std::hex << "Error: invalid CSR write addr=0x" << addr << ", value=0x" << value << std::endl;
      std::abort();
    }
  }
}

void Core::trigger_ecall() {
  ecall_ = true;
}

void Core::trigger_ebreak() {
  ebreak_ = true;
}

bool Core::check_exit() const {
  return ebreak_ || ecall_;
}

bool Core::running() const {
  bool is_running = (committed_instrs_ != issued_instrs_);
  return is_running;
}