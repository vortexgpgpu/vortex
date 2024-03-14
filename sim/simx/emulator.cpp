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

#include "emulator.h"
#include "instr_trace.h"
#include "instr.h"
#include "dcrs.h"
#include "core.h"
#include "socket.h"
#include "cluster.h"
#include "processor_impl.h"
#include "local_mem.h"

using namespace vortex;

Emulator::ipdom_entry_t::ipdom_entry_t(const ThreadMask &tmask, Word PC) 
  : tmask(tmask)
  , PC(PC)
  , fallthrough(false)
{}

Emulator::ipdom_entry_t::ipdom_entry_t(const ThreadMask &tmask) 
  : tmask(tmask)
  , fallthrough(true)
{}

Emulator::warp_t::warp_t(const Arch& arch)
  : ireg_file(arch.num_threads(), std::vector<Word>(arch.num_regs()))
  , freg_file(arch.num_threads(), std::vector<uint64_t>(arch.num_regs()))
{}

void Emulator::warp_t::clear(const Arch& arch, const DCRS &dcrs) {
  this->PC = dcrs.base_dcrs.read(VX_DCR_BASE_STARTUP_ADDR0);
#if (XLEN == 64)
  this->PC = (uint64_t(dcrs.base_dcrs.read(VX_DCR_BASE_STARTUP_ADDR1)) << 32) | this->PC;
#endif
  this->tmask.reset();
  for (uint32_t i = 0, n = arch.num_threads(); i < n; ++i) {
    for (auto& reg : this->ireg_file.at(i)) {
      reg = 0;
    }
    for (auto& reg : this->freg_file.at(i)) {
      reg = 0;
    }
  }
  this->fcsr = 0;
  this->uui_gen.reset();
}

///////////////////////////////////////////////////////////////////////////////

Emulator::Emulator(const Arch &arch, const DCRS &dcrs, Core* core)
    : arch_(arch)
    , dcrs_(dcrs)
    , core_(core)
    , warps_(arch.num_warps(), arch)    
    , barriers_(arch.num_barriers(), 0)
{
  this->clear();
}

Emulator::~Emulator() {
  this->cout_flush();
}

void Emulator::clear() {  
  for (auto& warp : warps_) {
    warp.clear(arch_, dcrs_);
  }
  
  for (auto& barrier : barriers_) {
    barrier.reset();
  }

  stalled_warps_.reset();
  active_warps_.reset();

  // activate first warp and thread
  active_warps_.set(0);
  warps_[0].tmask.set(0);
}

void Emulator::attach_ram(RAM* ram) {
  // bind RAM to memory unit
#if (XLEN == 64)
  mmu_.attach(*ram, 0, 0xFFFFFFFFFFFFFFFF);
#else
  mmu_.attach(*ram, 0, 0xFFFFFFFF);
#endif
}

instr_trace_t* Emulator::step() {  
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
    return nullptr;

  // suspend warp until decode
  auto& warp = warps_.at(scheduled_warp);
  assert(warp.tmask.any());

#ifndef NDEBUG
  uint32_t instr_uuid = warp.uui_gen.get_uuid(warp.PC);
  uint32_t g_wid = core_->id() * arch_.num_warps() + scheduled_warp;
  uint32_t instr_id  = instr_uuid & 0xffff;
  uint32_t instr_ref = instr_uuid >> 16;
  uint64_t uuid = (uint64_t(instr_ref) << 32) | (g_wid << 16) | instr_id;
#else
  uint64_t uuid = 0;
#endif
  
  DPH(1, "Fetch: cid=" << core_->id() << ", wid=" << scheduled_warp << ", tmask=");
  for (uint32_t i = 0, n = arch_.num_threads(); i < n; ++i)
    DPN(1, warp.tmask.test(i));
  DPN(1, ", PC=0x" << std::hex << warp.PC << " (#" << std::dec << uuid << ")" << std::endl);

  // Fetch
  uint32_t instr_code = 0;
  this->icache_read(&instr_code, warp.PC, sizeof(uint32_t));

  // Decode
  auto instr = this->decode(instr_code);
  if (!instr) {
    std::cout << std::hex << "Error: invalid instruction 0x" << instr_code << ", at PC=0x" << warp.PC << " (#" << std::dec << uuid << ")" << std::endl;
    std::abort();
  }  

  DP(1, "Instr 0x" << std::hex << instr_code << ": " << *instr);

  // Create trace
  auto trace = new instr_trace_t(uuid, arch_);
    
  // Execute
  this->execute(*instr, scheduled_warp, trace);

  DP(5, "Register state:");
  for (uint32_t i = 0; i < arch_.num_regs(); ++i) {
    DPN(5, "  %r" << std::setfill('0') << std::setw(2) << std::dec << i << ':');
    // Integer register file
    for (uint32_t j = 0; j < arch_.num_threads(); ++j) {
      DPN(5, ' ' << std::setfill('0') << std::setw(XLEN/4) << std::hex << warp.ireg_file.at(j).at(i) << std::setfill(' ') << ' ');
    }
    DPN(5, '|');
    // Floating point register file
    for (uint32_t j = 0; j < arch_.num_threads(); ++j) {
      DPN(5, ' ' << std::setfill('0') << std::setw(16) << std::hex << warp.freg_file.at(j).at(i) << std::setfill(' ') << ' ');
    }
    DPN(5, std::endl);
  }  

  return trace;
}

bool Emulator::running() const {
  return active_warps_.any();
}

int Emulator::get_exitcode() const {
  return warps_.at(0).ireg_file.at(0).at(3);
}

void Emulator::suspend(uint32_t wid) {
  assert(!stalled_warps_.test(wid));
  stalled_warps_.set(wid);
}

void Emulator::resume(uint32_t wid) {
  if (wid != 0xffffffff) {
    assert(stalled_warps_.test(wid));
    stalled_warps_.reset(wid);
  } else {
    stalled_warps_.reset();
  }
}

void Emulator::wspawn(uint32_t num_warps, Word nextPC) {
  uint32_t active_warps = std::min<uint32_t>(num_warps, arch_.num_warps());
  DP(3, "*** Activate " << (active_warps-1) << " warps at PC: " << std::hex << nextPC);
  for (uint32_t i = 1; i < active_warps; ++i) {
    auto& warp = warps_.at(i);
    warp.PC = nextPC;
    warp.tmask.set(0);
    active_warps_.set(i);
  }
}

void Emulator::barrier(uint32_t bar_id, uint32_t count, uint32_t wid) {
  uint32_t bar_idx = bar_id & 0x7fffffff;
  bool is_global = (bar_id >> 31);

  auto& barrier = barriers_.at(bar_idx);
  barrier.set(wid);
  DP(3, "*** Suspend core #" << core_->id() << ", warp #" << wid << " at barrier #" << bar_idx);

  if (is_global) {
    // global barrier handling
    if (barrier.count() == active_warps_.count()) {
      core_->socket()->barrier(bar_idx, count, core_->id());
      barrier.reset();
    }    
  } else {
    // local barrier handling
    if (barrier.count() == (size_t)count) {
      // resume suspended warps
      for (uint32_t i = 0; i < arch_.num_warps(); ++i) {
        if (barrier.test(i)) {
          DP(3, "*** Resume core #" << core_->id() << ", warp #" << i << " at barrier #" << bar_idx);
          stalled_warps_.reset(i);
        }
      }
      barrier.reset();
    }
  }
}

void Emulator::icache_read(void *data, uint64_t addr, uint32_t size) {
  mmu_.read(data, addr, size, 0);
}

void Emulator::dcache_read(void *data, uint64_t addr, uint32_t size) {  
  auto type = get_addr_type(addr);
  if (type == AddrType::Shared) {
    core_->local_mem()->read(data, addr, size);
  } else {  
    mmu_.read(data, addr, size, 0);
  }

  DPH(2, "Mem Read: addr=0x" << std::hex << addr << ", data=0x" << ByteStream(data, size) << " (size=" << size << ", type=" << type << ")" << std::endl);
}

void Emulator::dcache_write(const void* data, uint64_t addr, uint32_t size) {  
  auto type = get_addr_type(addr);
  if (addr >= uint64_t(IO_COUT_ADDR)
   && addr < (uint64_t(IO_COUT_ADDR) + IO_COUT_SIZE)) {
     this->writeToStdOut(data, addr, size);
  } else {
    if (type == AddrType::Shared) {
      core_->local_mem()->write(data, addr, size);
    } else {
      mmu_.write(data, addr, size, 0);
    }
  }
  DPH(2, "Mem Write: addr=0x" << std::hex << addr << ", data=0x" << ByteStream(data, size) << " (size=" << size << ", type=" << type << ")" << std::endl);  
}

void Emulator::dcache_amo_reserve(uint64_t addr) {
  auto type = get_addr_type(addr);
  if (type == AddrType::Global) {
    mmu_.amo_reserve(addr);
  }
}

bool Emulator::dcache_amo_check(uint64_t addr) {
  auto type = get_addr_type(addr);
  if (type == AddrType::Global) {
    return mmu_.amo_check(addr);
  }
  return false;
}

void Emulator::writeToStdOut(const void* data, uint64_t addr, uint32_t size) {
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

void Emulator::cout_flush() {
  for (auto& buf : print_bufs_) {
    auto str = buf.second.str();
    if (!str.empty()) {
      std::cout << "#" << buf.first << ": " << str << std::endl;
    }
  }
}

uint32_t Emulator::get_csr(uint32_t addr, uint32_t tid, uint32_t wid) {  
  auto core_perf = core_->perf_stats();  
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
    return warps_.at(wid).fcsr & 0x1F;
  case VX_CSR_FRM:
    return (warps_.at(wid).fcsr >> 5);
  case VX_CSR_FCSR:
    return warps_.at(wid).fcsr;
  case VX_CSR_MHARTID: // global thread ID
    return (core_->id() * arch_.num_warps() + wid) * arch_.num_threads() + tid;
  case VX_CSR_THREAD_ID: // thread ID
    return tid;
  case VX_CSR_WARP_ID: // warp ID
    return wid;
  case VX_CSR_CORE_ID: // core ID
    return core_->id();
  case VX_CSR_THREAD_MASK: // thread mask
    return warps_.at(wid).tmask.to_ulong();
  case VX_CSR_WARP_MASK: // active warps
    return active_warps_.to_ulong();
  case VX_CSR_NUM_THREADS: // Number of threads per warp
    return arch_.num_threads();
  case VX_CSR_NUM_WARPS: // Number of warps per core
    return arch_.num_warps();
  case VX_CSR_NUM_CORES: // Number of cores per cluster
    return uint32_t(arch_.num_cores()) * arch_.num_clusters();
  case VX_CSR_MCYCLE: // NumCycles
    return core_perf.cycles & 0xffffffff;
  case VX_CSR_MCYCLE_H: // NumCycles
    return (uint32_t)(core_perf.cycles >> 32);
  case VX_CSR_MINSTRET: // NumInsts
    return core_perf.instrs & 0xffffffff;
  case VX_CSR_MINSTRET_H: // NumInsts
    return (uint32_t)(core_perf.instrs >> 32);
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
        case VX_CSR_MPM_SCHED_ID:  return core_perf.sched_idle & 0xffffffff; 
        case VX_CSR_MPM_SCHED_ID_H:return core_perf.sched_idle >> 32;
        case VX_CSR_MPM_SCHED_ST:  return core_perf.sched_stalls & 0xffffffff; 
        case VX_CSR_MPM_SCHED_ST_H:return core_perf.sched_stalls >> 32;
        case VX_CSR_MPM_IBUF_ST:   return core_perf.ibuf_stalls & 0xffffffff; 
        case VX_CSR_MPM_IBUF_ST_H: return core_perf.ibuf_stalls >> 32; 
        case VX_CSR_MPM_SCRB_ST:   return core_perf.scrb_stalls & 0xffffffff;
        case VX_CSR_MPM_SCRB_ST_H: return core_perf.scrb_stalls >> 32;
        case VX_CSR_MPM_SCRB_ALU:  return core_perf.scrb_alu & 0xffffffff;
        case VX_CSR_MPM_SCRB_ALU_H:return core_perf.scrb_alu >> 32;
        case VX_CSR_MPM_SCRB_FPU:  return core_perf.scrb_fpu & 0xffffffff;
        case VX_CSR_MPM_SCRB_FPU_H:return core_perf.scrb_fpu >> 32;
        case VX_CSR_MPM_SCRB_LSU:  return core_perf.scrb_lsu & 0xffffffff;
        case VX_CSR_MPM_SCRB_LSU_H:return core_perf.scrb_lsu >> 32;
        case VX_CSR_MPM_SCRB_SFU:  return core_perf.scrb_sfu & 0xffffffff;
        case VX_CSR_MPM_SCRB_SFU_H:return core_perf.scrb_sfu >> 32;
        case VX_CSR_MPM_SCRB_WCTL: return core_perf.scrb_wctl & 0xffffffff;
        case VX_CSR_MPM_SCRB_WCTL_H: return core_perf.scrb_wctl >> 32;
        case VX_CSR_MPM_SCRB_CSRS: return core_perf.scrb_csrs & 0xffffffff;
        case VX_CSR_MPM_SCRB_CSRS_H: return core_perf.scrb_csrs >> 32;
        case VX_CSR_MPM_IFETCHES:  return core_perf.ifetches & 0xffffffff; 
        case VX_CSR_MPM_IFETCHES_H: return core_perf.ifetches >> 32; 
        case VX_CSR_MPM_LOADS:     return core_perf.loads & 0xffffffff; 
        case VX_CSR_MPM_LOADS_H:   return core_perf.loads >> 32; 
        case VX_CSR_MPM_STORES:    return core_perf.stores & 0xffffffff; 
        case VX_CSR_MPM_STORES_H:  return core_perf.stores >> 32;
        case VX_CSR_MPM_IFETCH_LT: return core_perf.ifetch_latency & 0xffffffff; 
        case VX_CSR_MPM_IFETCH_LT_H: return core_perf.ifetch_latency >> 32; 
        case VX_CSR_MPM_LOAD_LT:   return core_perf.load_latency & 0xffffffff; 
        case VX_CSR_MPM_LOAD_LT_H: return core_perf.load_latency >> 32;
       }
      } break; 
      case VX_DCR_MPM_CLASS_MEM: {
        auto proc_perf = core_->socket()->cluster()->processor()->perf_stats();
        auto cluster_perf = core_->socket()->cluster()->perf_stats();
        auto socket_perf = core_->socket()->perf_stats();
        auto lmem_perf = core_->local_mem()->perf_stats();
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
         
        case VX_CSR_MPM_LMEM_READS:       return lmem_perf.reads & 0xffffffff;
        case VX_CSR_MPM_LMEM_READS_H:     return lmem_perf.reads >> 32;
        case VX_CSR_MPM_LMEM_WRITES:      return lmem_perf.writes & 0xffffffff;
        case VX_CSR_MPM_LMEM_WRITES_H:    return lmem_perf.writes >> 32;
        case VX_CSR_MPM_LMEM_BANK_ST:     return lmem_perf.bank_stalls & 0xffffffff; 
        case VX_CSR_MPM_LMEM_BANK_ST_H:   return lmem_perf.bank_stalls >> 32; 
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

void Emulator::set_csr(uint32_t addr, uint32_t value, uint32_t tid, uint32_t wid) {
  __unused (tid);
  switch (addr) {
  case VX_CSR_FFLAGS:
    warps_.at(wid).fcsr = (warps_.at(wid).fcsr & ~0x1F) | (value & 0x1F);
    break;
  case VX_CSR_FRM:
    warps_.at(wid).fcsr = (warps_.at(wid).fcsr & ~0xE0) | (value << 5);
    break;
  case VX_CSR_FCSR:
    warps_.at(wid).fcsr = value & 0xff;
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

uint32_t Emulator::get_fpu_rm(uint32_t func3, uint32_t tid, uint32_t wid) {
  return (func3 == 0x7) ? this->get_csr(VX_CSR_FRM, tid, wid) : func3;
}

void Emulator::update_fcrs(uint32_t fflags, uint32_t tid, uint32_t wid) {
  if (fflags) {
    this->set_csr(VX_CSR_FCSR, this->get_csr(VX_CSR_FCSR, tid, wid) | fflags, tid, wid);
    this->set_csr(VX_CSR_FFLAGS, this->get_csr(VX_CSR_FFLAGS, tid, wid) | fflags, tid, wid);
  }
}

void Emulator::trigger_ecall() {
  active_warps_.reset();
}

void Emulator::trigger_ebreak() {
  active_warps_.reset();
}