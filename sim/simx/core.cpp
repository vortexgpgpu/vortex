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
#include "processor_impl.h"

using namespace vortex;

Core::Core(const SimContext& ctx, 
           uint32_t core_id, 
           Cluster* cluster,
           const Arch &arch, 
           const DCRS &dcrs,
           SharedMem::Ptr  sharedmem,
           std::vector<RasterUnit::Ptr>& raster_units,
           std::vector<RopUnit::Ptr>& rop_units,
           std::vector<TexUnit::Ptr>& tex_units)
    : SimObject(ctx, "core")
    , icache_req_ports(1, this)
    , icache_rsp_ports(1, this)
    , dcache_req_ports(arch.num_threads(), this)
    , dcache_rsp_ports(arch.num_threads(), this)
    , core_id_(core_id)
    , arch_(arch)
    , dcrs_(dcrs)
    , decoder_(arch)
    , warps_(arch.num_warps())
    , barriers_(arch.num_barriers(), 0)
    , fcsrs_(arch.num_warps(), 0)
    , ibuffers_(arch.num_warps(), IBUF_SIZE)
    , scoreboard_(arch_) 
    , exe_units_((int)ExeType::MAX)    
    , raster_units_(raster_units)
    , rop_units_(rop_units)
    , tex_units_(tex_units)
    , sharedmem_(sharedmem)
    , fetch_latch_("fetch")
    , decode_latch_("decode")
    , pending_icache_(arch_.num_warps())
    , csrs_(arch.num_warps())
    , cluster_(cluster)
    , raster_idx_(0)
    , rop_idx_(0)
    , tex_idx_(0)
{  
  for (uint32_t i = 0; i < arch_.num_warps(); ++i) {
    csrs_.at(i).resize(arch.num_threads());
  }

  for (uint32_t i = 0; i < arch_.num_warps(); ++i) {
    warps_.at(i) = std::make_shared<Warp>(this, i);
  }

  // register execute units
  exe_units_.at((int)ExeType::ALU) = SimPlatform::instance().create_object<AluUnit>(this);
  exe_units_.at((int)ExeType::LSU) = SimPlatform::instance().create_object<LsuUnit>(this);
  exe_units_.at((int)ExeType::CSR) = SimPlatform::instance().create_object<CsrUnit>(this);
  exe_units_.at((int)ExeType::FPU) = SimPlatform::instance().create_object<FpuUnit>(this);  
  exe_units_.at((int)ExeType::GPU) = SimPlatform::instance().create_object<GpuUnit>(this);

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

  for (auto& raster_unit : raster_units_) {
    raster_unit->reset();
  }

  for (auto& rop_unit : rop_units_) {
    rop_unit->reset();
  }

  for (auto& tex_unit : tex_units_) {
    tex_unit->reset();
  }
  
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
  exited_ = false;
  perf_stats_ = PerfStats();
  pending_ifetches_ = 0;
}

void Core::tick() {
  this->commit();
  this->execute();
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
  perf_stats_.ifetch_latency += pending_ifetches_;

  // handle icache reponse
  auto& icache_rsp_port = icache_rsp_ports.at(0);
  if (!icache_rsp_port.empty()){
    auto& mem_rsp = icache_rsp_port.front();
    auto trace = pending_icache_.at(mem_rsp.tag);
    decode_latch_.push(trace);
    DT(3, "icache-rsp: addr=" << std::hex << trace->PC << ", tag=" << mem_rsp.tag << ", " << *trace);
    pending_icache_.release(mem_rsp.tag);
    icache_rsp_port.pop();
    --pending_ifetches_;
  }

  // send icache request
  if (!fetch_latch_.empty()) {
    auto trace = fetch_latch_.front();
    MemReq mem_req;
    mem_req.addr  = trace->PC;
    mem_req.write = false;
    mem_req.tag   = pending_icache_.allocate(trace);    
    mem_req.cid   = trace->cid;
    mem_req.uuid  = trace->uuid;
    icache_req_ports.at(0).send(mem_req, 1);    
    DT(3, "icache-req: addr=" << std::hex << mem_req.addr << ", tag=" << mem_req.tag << ", " << *trace);    
    fetch_latch_.pop();    
    ++pending_ifetches_;   
    ++perf_stats_.ifetches;
  }  
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
    stalled_warps_.reset(trace->wid);
  }

  // update perf counters
  uint32_t active_threads = trace->tmask.count();
  if (trace->exe_type == ExeType::LSU && trace->lsu_type == LsuType::LOAD)
    perf_stats_.loads += active_threads;
  if (trace->exe_type == ExeType::LSU && trace->lsu_type == LsuType::STORE) 
    perf_stats_.stores += active_threads;

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
      if (!trace->log_once(true)) {
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
      trace->log_once(false);
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
      assert(trace->cid == core_id_);

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
      cluster_->processor()->barrier(bar_idx, count, core_id_);
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
  if (addr >= uint64_t(IO_COUT_ADDR)
   && addr < (uint64_t(IO_COUT_ADDR) + IO_COUT_SIZE)) {
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
  case CSR_MNSTATUS:
    return 0;

  case CSR_FFLAGS:
    return fcsrs_.at(wid) & 0x1F;
  case CSR_FRM:
    return (fcsrs_.at(wid) >> 5);
  case CSR_FCSR:
    return fcsrs_.at(wid);
  case CSR_WTID: // Warp threadID
    return tid;
  case CSR_LTID: // Core threadID
    return tid + (wid * arch_.num_threads());
  case CSR_GTID: // Processor threadID
    return (core_id_ * arch_.num_warps() + wid) * arch_.num_threads() + tid;
  case CSR_LWID: // Core warpID
    return wid;
  case CSR_GWID: // Processor warpID        
    return core_id_ * arch_.num_warps() + wid;
  case CSR_GCID: // Processor coreID
    return core_id_;
  case CSR_TMASK: // Processor coreID
    return warps_.at(wid)->getTmask();
  case CSR_NT: // Number of threads per warp
    return arch_.num_threads();
  case CSR_NW: // Number of warps per core
    return arch_.num_warps();
  case CSR_NC: // Number of cores
    return arch_.num_cores();
  case CSR_MCYCLE: // NumCycles
    return perf_stats_.cycles & 0xffffffff;
  case CSR_MCYCLE_H: // NumCycles
    return (uint32_t)(perf_stats_.cycles >> 32);
  case CSR_MINSTRET: // NumInsts
    return perf_stats_.instrs & 0xffffffff;
  case CSR_MINSTRET_H: // NumInsts
    return (uint32_t)(perf_stats_.instrs >> 32);
  default:
    if ((addr >= CSR_MPM_BASE && addr < (CSR_MPM_BASE + 32))
     || (addr >= CSR_MPM_BASE_H && addr < (CSR_MPM_BASE_H + 32))) {
      // user-defined MPM CSRs
      auto perf_class = dcrs_.base_dcrs.read(DCR_BASE_MPM_CLASS);
      switch (perf_class) {                
      case DCR_MPM_CLASS_NONE: 
        break;    
      case DCR_MPM_CLASS_CORE: {
        switch (addr) {
        case CSR_MPM_IBUF_ST:   return perf_stats_.ibuf_stalls & 0xffffffff; 
        case CSR_MPM_IBUF_ST_H: return perf_stats_.ibuf_stalls >> 32; 
        case CSR_MPM_SCRB_ST:   return perf_stats_.scrb_stalls & 0xffffffff; 
        case CSR_MPM_SCRB_ST_H: return perf_stats_.scrb_stalls >> 32; 
        case CSR_MPM_ALU_ST:    return perf_stats_.alu_stalls & 0xffffffff; 
        case CSR_MPM_ALU_ST_H:  return perf_stats_.alu_stalls >> 32; 
        case CSR_MPM_LSU_ST:    return perf_stats_.lsu_stalls & 0xffffffff; 
        case CSR_MPM_LSU_ST_H:  return perf_stats_.lsu_stalls >> 32; 
        case CSR_MPM_CSR_ST:    return perf_stats_.csr_stalls & 0xffffffff; 
        case CSR_MPM_CSR_ST_H:  return perf_stats_.csr_stalls >> 32; 
        case CSR_MPM_FPU_ST:    return perf_stats_.fpu_stalls & 0xffffffff; 
        case CSR_MPM_FPU_ST_H:  return perf_stats_.fpu_stalls >> 32; 
        case CSR_MPM_GPU_ST:    return perf_stats_.gpu_stalls & 0xffffffff; 
        case CSR_MPM_GPU_ST_H:  return perf_stats_.gpu_stalls >> 32; 
        
        case CSR_MPM_IFETCHES:  return perf_stats_.ifetches & 0xffffffff; 
        case CSR_MPM_IFETCHES_H: return perf_stats_.ifetches >> 32; 
        case CSR_MPM_LOADS:     return perf_stats_.loads & 0xffffffff; 
        case CSR_MPM_LOADS_H:   return perf_stats_.loads >> 32; 
        case CSR_MPM_STORES:    return perf_stats_.stores & 0xffffffff; 
        case CSR_MPM_STORES_H:  return perf_stats_.stores >> 32;
        case CSR_MPM_IFETCH_LAT: return perf_stats_.ifetch_latency & 0xffffffff; 
        case CSR_MPM_IFETCH_LAT_H: return perf_stats_.ifetch_latency >> 32; 
        case CSR_MPM_LOAD_LAT:  return perf_stats_.load_latency & 0xffffffff; 
        case CSR_MPM_LOAD_LAT_H: return perf_stats_.load_latency >> 32;
       }
      } break; 
      case DCR_MPM_CLASS_MEM: {
        auto proc_perf = cluster_->processor()->perf_stats();
        switch (addr) {
        case CSR_MPM_ICACHE_READS:    return proc_perf.clusters.icache.reads & 0xffffffff; 
        case CSR_MPM_ICACHE_READS_H:  return proc_perf.clusters.icache.reads >> 32; 
        case CSR_MPM_ICACHE_MISS_R:   return proc_perf.clusters.icache.read_misses & 0xffffffff;
        case CSR_MPM_ICACHE_MISS_R_H: return proc_perf.clusters.icache.read_misses >> 32;
        
        case CSR_MPM_DCACHE_READS:    return proc_perf.clusters.dcache.reads & 0xffffffff; 
        case CSR_MPM_DCACHE_READS_H:  return proc_perf.clusters.dcache.reads >> 32; 
        case CSR_MPM_DCACHE_WRITES:   return proc_perf.clusters.dcache.writes & 0xffffffff; 
        case CSR_MPM_DCACHE_WRITES_H: return proc_perf.clusters.dcache.writes >> 32; 
        case CSR_MPM_DCACHE_MISS_R:   return proc_perf.clusters.dcache.read_misses & 0xffffffff; 
        case CSR_MPM_DCACHE_MISS_R_H: return proc_perf.clusters.dcache.read_misses >> 32; 
        case CSR_MPM_DCACHE_MISS_W:   return proc_perf.clusters.dcache.write_misses & 0xffffffff; 
        case CSR_MPM_DCACHE_MISS_W_H: return proc_perf.clusters.dcache.write_misses >> 32; 
        case CSR_MPM_DCACHE_BANK_ST:  return proc_perf.clusters.dcache.bank_stalls & 0xffffffff; 
        case CSR_MPM_DCACHE_BANK_ST_H:return proc_perf.clusters.dcache.bank_stalls >> 32;
        case CSR_MPM_DCACHE_MSHR_ST:  return proc_perf.clusters.dcache.mshr_stalls & 0xffffffff; 
        case CSR_MPM_DCACHE_MSHR_ST_H:return proc_perf.clusters.dcache.mshr_stalls >> 32;
        
        case CSR_MPM_SMEM_READS:    return proc_perf.clusters.sharedmem.reads & 0xffffffff;
        case CSR_MPM_SMEM_READS_H:  return proc_perf.clusters.sharedmem.reads >> 32;
        case CSR_MPM_SMEM_WRITES:   return proc_perf.clusters.sharedmem.writes & 0xffffffff;
        case CSR_MPM_SMEM_WRITES_H: return proc_perf.clusters.sharedmem.writes >> 32;
        case CSR_MPM_SMEM_BANK_ST:  return proc_perf.clusters.sharedmem.bank_stalls & 0xffffffff; 
        case CSR_MPM_SMEM_BANK_ST_H:return proc_perf.clusters.sharedmem.bank_stalls >> 32; 

        case CSR_MPM_L2CACHE_READS:    return proc_perf.clusters.l2cache.reads & 0xffffffff; 
        case CSR_MPM_L2CACHE_READS_H:  return proc_perf.clusters.l2cache.reads >> 32; 
        case CSR_MPM_L2CACHE_WRITES:   return proc_perf.clusters.l2cache.writes & 0xffffffff; 
        case CSR_MPM_L2CACHE_WRITES_H: return proc_perf.clusters.l2cache.writes >> 32; 
        case CSR_MPM_L2CACHE_MISS_R:   return proc_perf.clusters.l2cache.read_misses & 0xffffffff; 
        case CSR_MPM_L2CACHE_MISS_R_H: return proc_perf.clusters.l2cache.read_misses >> 32; 
        case CSR_MPM_L2CACHE_MISS_W:   return proc_perf.clusters.l2cache.write_misses & 0xffffffff; 
        case CSR_MPM_L2CACHE_MISS_W_H: return proc_perf.clusters.l2cache.write_misses >> 32; 
        case CSR_MPM_L2CACHE_BANK_ST:  return proc_perf.clusters.l2cache.bank_stalls & 0xffffffff; 
        case CSR_MPM_L2CACHE_BANK_ST_H:return proc_perf.clusters.l2cache.bank_stalls >> 32;
        case CSR_MPM_L2CACHE_MSHR_ST:  return proc_perf.clusters.l2cache.mshr_stalls & 0xffffffff; 
        case CSR_MPM_L2CACHE_MSHR_ST_H:return proc_perf.clusters.l2cache.mshr_stalls >> 32;

        case CSR_MPM_L3CACHE_READS:    return proc_perf.l3cache.reads & 0xffffffff; 
        case CSR_MPM_L3CACHE_READS_H:  return proc_perf.l3cache.reads >> 32; 
        case CSR_MPM_L3CACHE_WRITES:   return proc_perf.l3cache.writes & 0xffffffff; 
        case CSR_MPM_L3CACHE_WRITES_H: return proc_perf.l3cache.writes >> 32; 
        case CSR_MPM_L3CACHE_MISS_R:   return proc_perf.l3cache.read_misses & 0xffffffff; 
        case CSR_MPM_L3CACHE_MISS_R_H: return proc_perf.l3cache.read_misses >> 32; 
        case CSR_MPM_L3CACHE_MISS_W:   return proc_perf.l3cache.write_misses & 0xffffffff; 
        case CSR_MPM_L3CACHE_MISS_W_H: return proc_perf.l3cache.write_misses >> 32; 
        case CSR_MPM_L3CACHE_BANK_ST:  return proc_perf.l3cache.bank_stalls & 0xffffffff; 
        case CSR_MPM_L3CACHE_BANK_ST_H:return proc_perf.l3cache.bank_stalls >> 32;
        case CSR_MPM_L3CACHE_MSHR_ST:  return proc_perf.l3cache.mshr_stalls & 0xffffffff; 
        case CSR_MPM_L3CACHE_MSHR_ST_H:return proc_perf.l3cache.mshr_stalls >> 32;

        case CSR_MPM_MEM_READS:   return proc_perf.mem_reads & 0xffffffff; 
        case CSR_MPM_MEM_READS_H: return proc_perf.mem_reads >> 32; 
        case CSR_MPM_MEM_WRITES:  return proc_perf.mem_writes & 0xffffffff; 
        case CSR_MPM_MEM_WRITES_H:return proc_perf.mem_writes >> 32; 
        case CSR_MPM_MEM_LAT:     return proc_perf.mem_latency & 0xffffffff; 
        case CSR_MPM_MEM_LAT_H:   return proc_perf.mem_latency >> 32;
        }
      } break;
      case DCR_MPM_CLASS_TEX: {
        auto proc_perf = cluster_->processor()->perf_stats();
        switch (addr) {
        case CSR_MPM_TEX_READS:   return proc_perf.clusters.tex_unit.reads & 0xffffffff;
        case CSR_MPM_TEX_READS_H: return proc_perf.clusters.tex_unit.reads >> 32;
        case CSR_MPM_TEX_LAT:     return proc_perf.clusters.tex_unit.latency & 0xffffffff;
        case CSR_MPM_TEX_LAT_H:   return proc_perf.clusters.tex_unit.latency >> 32;
        case CSR_MPM_TEX_STALL:   return proc_perf.clusters.tex_unit.stalls & 0xffffffff;
        case CSR_MPM_TEX_STALL_H: return proc_perf.clusters.tex_unit.stalls >> 32;

        case CSR_MPM_TCACHE_READS:    return proc_perf.clusters.tcache.reads & 0xffffffff; 
        case CSR_MPM_TCACHE_READS_H:  return proc_perf.clusters.tcache.reads >> 32;
        case CSR_MPM_TCACHE_MISS_R:   return proc_perf.clusters.tcache.read_misses & 0xffffffff; 
        case CSR_MPM_TCACHE_MISS_R_H: return proc_perf.clusters.tcache.read_misses >> 32;
        case CSR_MPM_TCACHE_BANK_ST:  return proc_perf.clusters.tcache.bank_stalls & 0xffffffff; 
        case CSR_MPM_TCACHE_BANK_ST_H:return proc_perf.clusters.tcache.bank_stalls >> 32;
        case CSR_MPM_TCACHE_MSHR_ST:  return proc_perf.clusters.tcache.mshr_stalls & 0xffffffff; 
        case CSR_MPM_TCACHE_MSHR_ST_H:return proc_perf.clusters.tcache.mshr_stalls >> 32;

        case CSR_MPM_TEX_ISSUE_ST:    return perf_stats_.tex_issue_stalls & 0xffffffff;
        case CSR_MPM_TEX_ISSUE_ST_H:  return perf_stats_.tex_issue_stalls >> 32;
        }
      } break;
      case DCR_MPM_CLASS_RASTER: {
        auto proc_perf = cluster_->processor()->perf_stats();
        switch (addr) {        
        case CSR_MPM_RASTER_READS:   return proc_perf.clusters.raster_unit.reads & 0xffffffff;
        case CSR_MPM_RASTER_READS_H: return proc_perf.clusters.raster_unit.reads >> 32;
        case CSR_MPM_RASTER_LAT:     return proc_perf.clusters.raster_unit.latency & 0xffffffff;
        case CSR_MPM_RASTER_LAT_H:   return proc_perf.clusters.raster_unit.latency >> 32;
        case CSR_MPM_RASTER_STALL:   return proc_perf.clusters.raster_unit.stalls & 0xffffffff;
        case CSR_MPM_RASTER_STALL_H: return proc_perf.clusters.raster_unit.stalls >> 32;

        case CSR_MPM_RCACHE_READS:    return proc_perf.clusters.rcache.reads & 0xffffffff; 
        case CSR_MPM_RCACHE_READS_H:  return proc_perf.clusters.rcache.reads >> 32; 
        case CSR_MPM_RCACHE_MISS_R:   return proc_perf.clusters.rcache.read_misses & 0xffffffff; 
        case CSR_MPM_RCACHE_MISS_R_H: return proc_perf.clusters.rcache.read_misses >> 32;  
        case CSR_MPM_RCACHE_BANK_ST:  return proc_perf.clusters.rcache.bank_stalls & 0xffffffff; 
        case CSR_MPM_RCACHE_BANK_ST_H:return proc_perf.clusters.rcache.bank_stalls >> 32;
        case CSR_MPM_RCACHE_MSHR_ST:  return proc_perf.clusters.rcache.mshr_stalls & 0xffffffff; 
        case CSR_MPM_RCACHE_MSHR_ST_H:return proc_perf.clusters.rcache.mshr_stalls >> 32;

        case CSR_MPM_RASTER_ISSUE_ST: return perf_stats_.raster_issue_stalls & 0xffffffff;
        case CSR_MPM_RASTER_ISSUE_ST_H: return perf_stats_.raster_issue_stalls >> 32;
        default:
          return 0;
        }
      } break;
      case DCR_MPM_CLASS_ROP: {
        auto proc_perf = cluster_->processor()->perf_stats();
        switch (addr) { 
        case CSR_MPM_ROP_READS:   return proc_perf.clusters.rop_unit.reads & 0xffffffff;
        case CSR_MPM_ROP_READS_H: return proc_perf.clusters.rop_unit.reads >> 32;
        case CSR_MPM_ROP_WRITES:  return proc_perf.clusters.rop_unit.writes & 0xffffffff;
        case CSR_MPM_ROP_WRITES_H:return proc_perf.clusters.rop_unit.writes >> 32;
        case CSR_MPM_ROP_LAT:     return proc_perf.clusters.rop_unit.latency & 0xffffffff;
        case CSR_MPM_ROP_LAT_H:   return proc_perf.clusters.rop_unit.latency >> 32;
        case CSR_MPM_ROP_STALL:   return proc_perf.clusters.rop_unit.stalls & 0xffffffff;
        case CSR_MPM_ROP_STALL_H: return proc_perf.clusters.rop_unit.stalls >> 32;

        case CSR_MPM_OCACHE_READS:    return proc_perf.clusters.ocache.reads & 0xffffffff; 
        case CSR_MPM_OCACHE_READS_H:  return proc_perf.clusters.ocache.reads >> 32; 
        case CSR_MPM_OCACHE_WRITES:   return proc_perf.clusters.ocache.writes & 0xffffffff; 
        case CSR_MPM_OCACHE_WRITES_H: return proc_perf.clusters.ocache.writes >> 32; 
        case CSR_MPM_OCACHE_MISS_R:   return proc_perf.clusters.ocache.read_misses & 0xffffffff; 
        case CSR_MPM_OCACHE_MISS_R_H: return proc_perf.clusters.ocache.read_misses >> 32; 
        case CSR_MPM_OCACHE_MISS_W:   return proc_perf.clusters.ocache.write_misses & 0xffffffff; 
        case CSR_MPM_OCACHE_MISS_W_H: return proc_perf.clusters.ocache.write_misses >> 32; 
        case CSR_MPM_OCACHE_BANK_ST:  return proc_perf.clusters.ocache.bank_stalls & 0xffffffff; 
        case CSR_MPM_OCACHE_BANK_ST_H:return proc_perf.clusters.ocache.bank_stalls >> 32;
        case CSR_MPM_OCACHE_MSHR_ST:  return proc_perf.clusters.ocache.mshr_stalls & 0xffffffff; 
        case CSR_MPM_OCACHE_MSHR_ST_H:return proc_perf.clusters.ocache.mshr_stalls >> 32;

        case CSR_MPM_ROP_ISSUE_ST:    return perf_stats_.rop_issue_stalls & 0xffffffff;
        case CSR_MPM_ROP_ISSUE_ST_H:  return perf_stats_.rop_issue_stalls >> 32;
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
      return csrs_.at(wid).at(tid).at(addr);
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
  case CSR_MNSTATUS:
    break;
  default:
  #ifdef EXT_ROP_ENABLE
    if (addr >= CSR_ROP_BEGIN
     && addr < CSR_ROP_END) {
      csrs_.at(wid).at(tid)[addr] = value;
    } else
  #endif
  #ifdef EXT_TEX_ENABLE
    if (addr >= CSR_TEX_BEGIN
     && addr < CSR_TEX_END) {
      csrs_.at(wid).at(tid)[addr] = value;
    } else
  #endif
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

bool Core::check_exit(Word* exitcode, int reg) const {
  if (exited_) {
    *exitcode = warps_.at(0)->getIRegValue(reg);
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

uint32_t Core::raster_idx() {
  auto ret = raster_idx_++;
  raster_idx_ %= raster_units_.size();
  return ret;
}

uint32_t Core::rop_idx() {
  auto ret = rop_idx_++;
  rop_idx_ %= rop_units_.size();
  return ret;
}

uint32_t Core::tex_idx() {
  auto ret = tex_idx_++;
  tex_idx_ %= tex_units_.size();
  return ret;
}