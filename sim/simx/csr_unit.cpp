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

#include "csr_unit.h"
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <util.h>
#include "debug.h"
#include "core.h"
#include "scheduler.h"
#include "socket.h"
#include "cluster.h"
#include "processor_impl.h"
#include "processor_impl.h"
#include "local_mem.h"
#include "constants.h"
#include "VX_types.h"
#ifdef EXT_TCU_ENABLE
#include "tensor_unit.h"
#endif

using namespace vortex;

#ifdef XLEN_64
  #define CSR_READ_64(addr, value) \
    case addr: return value
#else
  #define CSR_READ_64(addr, value) \
    case addr : return (uint32_t)value; \
    case (addr + (VX_CSR_MPM_BASE_H-VX_CSR_MPM_BASE)) : return ((value >> 32) & 0xFFFFFFFF)
#endif

CsrUnit::CsrUnit(const SimContext& ctx, const char* name, Core* core)
  : FuncUnit(ctx, name, core)
{}

uint32_t CsrUnit::latency_of(const instr_trace_t* /*trace*/) const {
  return 4;
}

Word CsrUnit::get_csr(uint32_t addr, uint32_t wid, uint32_t tid) {
  auto& sched     = core_->scheduler();
  auto& warp      = sched.warp(wid);
  auto core_perf  = core_->perf_stats();
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
  case VX_CSR_MCAUSE:
    return 0;

  case VX_CSR_FFLAGS: return warp.fcsr & 0x1F;
  case VX_CSR_FRM:    return (warp.fcsr >> 5);
  case VX_CSR_FCSR:   return warp.fcsr;

  case VX_CSR_MHARTID:    return (core_->id() * NUM_WARPS + wid) * NUM_THREADS + tid;
  case VX_CSR_THREAD_ID:  return tid;
  case VX_CSR_WARP_ID:    return wid;
  case VX_CSR_CORE_ID:    return core_->id();
  case VX_CSR_ACTIVE_THREADS:return warp.tmask.to_ulong();
  case VX_CSR_ACTIVE_WARPS:return sched.active_warps().to_ulong();
  case VX_CSR_NUM_THREADS:return NUM_THREADS;
  case VX_CSR_NUM_WARPS:  return NUM_WARPS;
  case VX_CSR_NUM_CORES:  return uint32_t(NUM_CORES) * NUM_CLUSTERS;
  case VX_CSR_LOCAL_MEM_BASE: return LMEM_BASE_ADDR;
  case VX_CSR_NUM_BARRIERS: return NUM_BARRIERS;
  case VX_CSR_MSCRATCH:   return warp.mscratch;

  case VX_CSR_CTA_ID:       return warp.cta_csrs.cta_id;
  case VX_CSR_CTA_RANK:     return warp.cta_csrs.cta_rank;
  case VX_CSR_CTA_SIZE:     return warp.cta_csrs.cta_size;
  case VX_CSR_CTA_THREAD_ID_X:
  case VX_CSR_CTA_THREAD_ID_Y:
  case VX_CSR_CTA_THREAD_ID_Z: {
    auto& cta = warp.cta_csrs;
    uint32_t x = cta.thread_idx[0] + tid;
    uint32_t y = cta.thread_idx[1] + x / cta.block_dim[0];
    uint32_t z = cta.thread_idx[2] + y / cta.block_dim[1];
    x %= cta.block_dim[0];
    y %= cta.block_dim[1];
    if (addr == VX_CSR_CTA_THREAD_ID_X) return x;
    if (addr == VX_CSR_CTA_THREAD_ID_Y) return y;
    return z;
  }
  case VX_CSR_CTA_BLOCK_ID_X:  return warp.cta_csrs.block_idx[0];
  case VX_CSR_CTA_BLOCK_ID_Y:  return warp.cta_csrs.block_idx[1];
  case VX_CSR_CTA_BLOCK_ID_Z:  return warp.cta_csrs.block_idx[2];
  case VX_CSR_CTA_BLOCK_DIM_X: return warp.cta_csrs.block_dim[0];
  case VX_CSR_CTA_BLOCK_DIM_Y: return warp.cta_csrs.block_dim[1];
  case VX_CSR_CTA_BLOCK_DIM_Z: return warp.cta_csrs.block_dim[2];
  case VX_CSR_CTA_GRID_DIM_X:  return warp.cta_csrs.grid_dim[0];
  case VX_CSR_CTA_GRID_DIM_Y:  return warp.cta_csrs.grid_dim[1];
  case VX_CSR_CTA_GRID_DIM_Z:  return warp.cta_csrs.grid_dim[2];
  case VX_CSR_CTA_LMEM_ADDR:   return warp.cta_csrs.lmem_addr;

  CSR_READ_64(VX_CSR_MCYCLE, core_perf.cycles);
  CSR_READ_64(VX_CSR_MINSTRET, core_perf.instrs);
  default:
    if ((addr >= VX_CSR_MPM_BASE && addr < (VX_CSR_MPM_BASE + 32))
     || (addr >= VX_CSR_MPM_BASE_H && addr < (VX_CSR_MPM_BASE_H + 32))) {
      // user-defined MPM CSRs
      auto proc_perf = core_->socket()->cluster()->processor()->perf_stats();
      auto perf_class = core_->mpm_class();
      switch (perf_class) {
      case VX_DCR_MPM_CLASS_BASE:
        break;
      case VX_DCR_MPM_CLASS_CORE: {
        switch (addr) {
        CSR_READ_64(VX_CSR_MPM_SCHED_IDLE, core_perf.sched_idle);
        CSR_READ_64(VX_CSR_MPM_ACTIVE_WARPS, core_perf.active_warps);
        CSR_READ_64(VX_CSR_MPM_STALLED_WARPS, core_perf.stalled_warps);
        CSR_READ_64(VX_CSR_MPM_ISSUED_WARPS, core_perf.issued_warps);
        CSR_READ_64(VX_CSR_MPM_ISSUED_THREADS, core_perf.issued_threads);
        CSR_READ_64(VX_CSR_MPM_STALL_FETCH, core_perf.fetch_stalls);
        CSR_READ_64(VX_CSR_MPM_STALL_IBUF, core_perf.ibuf_stalls);
        CSR_READ_64(VX_CSR_MPM_STALL_SCRB, core_perf.scrb_stalls);
        CSR_READ_64(VX_CSR_MPM_STALL_OPDS, core_perf.opds_stalls);
        CSR_READ_64(VX_CSR_MPM_STALL_ALU, core_perf.alu_stalls);
        CSR_READ_64(VX_CSR_MPM_STALL_FPU, core_perf.fpu_stalls);
        CSR_READ_64(VX_CSR_MPM_STALL_LSU, core_perf.lsu_stalls);
        CSR_READ_64(VX_CSR_MPM_STALL_SFU, core_perf.sfu_stalls);
      #ifdef EXT_TCU_ENABLE
        CSR_READ_64(VX_CSR_MPM_STALL_TCU, core_perf.tcu_stalls);
      #endif
        CSR_READ_64(VX_CSR_MPM_BRANCHES, core_perf.branches);
        CSR_READ_64(VX_CSR_MPM_DIVERGENCE, core_perf.divergence);
        CSR_READ_64(VX_CSR_MPM_INSTR_ALU, core_perf.alu_instrs);
        CSR_READ_64(VX_CSR_MPM_INSTR_FPU, core_perf.fpu_instrs);
        CSR_READ_64(VX_CSR_MPM_INSTR_LSU, core_perf.lsu_instrs);
        CSR_READ_64(VX_CSR_MPM_INSTR_SFU, core_perf.sfu_instrs);
      #ifdef EXT_TCU_ENABLE
        CSR_READ_64(VX_CSR_MPM_INSTR_TCU, core_perf.tcu_instrs);
      #endif
        CSR_READ_64(VX_CSR_MPM_MEM_READS, proc_perf.mem_reads);
        CSR_READ_64(VX_CSR_MPM_MEM_WRITES, proc_perf.mem_writes);
        CSR_READ_64(VX_CSR_MPM_IFETCHES, core_perf.ifetches);
        CSR_READ_64(VX_CSR_MPM_IFETCH_LT, core_perf.ifetch_latency);
        CSR_READ_64(VX_CSR_MPM_LOADS, core_perf.loads);
        CSR_READ_64(VX_CSR_MPM_STORES, core_perf.stores);
        CSR_READ_64(VX_CSR_MPM_LOAD_LT, core_perf.load_latency);
        }
      } break;
      case VX_DCR_MPM_CLASS_MEM: {
        auto cluster_perf = core_->socket()->cluster()->perf_stats();
        auto socket_perf = core_->socket()->perf_stats();
        auto lmem_perf = core_->local_mem()->perf_stats();

        uint64_t coalescer_misses = 0;
        for (uint i = 0; i < NUM_LSU_BLOCKS; ++i) {
          coalescer_misses += core_->mem_coalescer(i)->perf_stats().misses;
        }

        switch (addr) {
        CSR_READ_64(VX_CSR_MPM_ICACHE_READS, socket_perf.icache.reads);
        CSR_READ_64(VX_CSR_MPM_ICACHE_MISS_R, socket_perf.icache.read_misses);
        CSR_READ_64(VX_CSR_MPM_ICACHE_MSHR_ST, socket_perf.icache.mshr_stalls);

        CSR_READ_64(VX_CSR_MPM_DCACHE_READS, socket_perf.dcache.reads);
        CSR_READ_64(VX_CSR_MPM_DCACHE_WRITES, socket_perf.dcache.writes);
        CSR_READ_64(VX_CSR_MPM_DCACHE_MISS_R, socket_perf.dcache.read_misses);
        CSR_READ_64(VX_CSR_MPM_DCACHE_MISS_W, socket_perf.dcache.write_misses);
        CSR_READ_64(VX_CSR_MPM_DCACHE_BANK_ST, socket_perf.dcache.bank_stalls);
        CSR_READ_64(VX_CSR_MPM_DCACHE_MSHR_ST, socket_perf.dcache.mshr_stalls);

        CSR_READ_64(VX_CSR_MPM_L2CACHE_READS, cluster_perf.l2cache.reads);
        CSR_READ_64(VX_CSR_MPM_L2CACHE_WRITES, cluster_perf.l2cache.writes);
        CSR_READ_64(VX_CSR_MPM_L2CACHE_MISS_R, cluster_perf.l2cache.read_misses);
        CSR_READ_64(VX_CSR_MPM_L2CACHE_MISS_W, cluster_perf.l2cache.write_misses);
        CSR_READ_64(VX_CSR_MPM_L2CACHE_BANK_ST, cluster_perf.l2cache.bank_stalls);
        CSR_READ_64(VX_CSR_MPM_L2CACHE_MSHR_ST, cluster_perf.l2cache.mshr_stalls);

        CSR_READ_64(VX_CSR_MPM_L3CACHE_READS, proc_perf.l3cache.reads);
        CSR_READ_64(VX_CSR_MPM_L3CACHE_WRITES, proc_perf.l3cache.writes);
        CSR_READ_64(VX_CSR_MPM_L3CACHE_MISS_R, proc_perf.l3cache.read_misses);
        CSR_READ_64(VX_CSR_MPM_L3CACHE_MISS_W, proc_perf.l3cache.write_misses);
        CSR_READ_64(VX_CSR_MPM_L3CACHE_BANK_ST, proc_perf.l3cache.bank_stalls);
        CSR_READ_64(VX_CSR_MPM_L3CACHE_MSHR_ST, proc_perf.l3cache.mshr_stalls);

        CSR_READ_64(VX_CSR_MPM_MEM_READS, proc_perf.mem_reads);
        CSR_READ_64(VX_CSR_MPM_MEM_WRITES, proc_perf.mem_writes);
        CSR_READ_64(VX_CSR_MPM_MEM_LT, proc_perf.mem_latency);
        CSR_READ_64(VX_CSR_MPM_MEM_BANK_ST, proc_perf.memsim.bank_stalls);

        CSR_READ_64(VX_CSR_MPM_COALESCER_MISS, coalescer_misses);

        CSR_READ_64(VX_CSR_MPM_LMEM_READS, lmem_perf.reads);
        CSR_READ_64(VX_CSR_MPM_LMEM_WRITES, lmem_perf.writes);
        CSR_READ_64(VX_CSR_MPM_LMEM_BANK_ST, lmem_perf.bank_stalls);
        }
      } break;
    #ifdef EXT_TCU_ENABLE
      case VX_DCR_MPM_CLASS_TCU: {
        auto tcu_perf = core_->tensor_unit()->perf_stats();
        switch (addr) {
        CSR_READ_64(VX_CSR_MPM_TCU_TBUF_STALLS,     tcu_perf.tbuf_stalls);
        CSR_READ_64(VX_CSR_MPM_TCU_TBUF_CACHE_HITS, tcu_perf.tbuf_cache_hits);
        CSR_READ_64(VX_CSR_MPM_TCU_LMEM_READS,    tcu_perf.lmem_reads);
        }
      } break;
    #endif
    #ifdef EXT_DXA_ENABLE
      case VX_DCR_MPM_CLASS_DXA: {
        auto cluster_perf = core_->socket()->cluster()->perf_stats();
        switch (addr) {
        CSR_READ_64(VX_CSR_MPM_DXA_TRANSFERS,  cluster_perf.dxa.transfers);
        CSR_READ_64(VX_CSR_MPM_DXA_GMEM_READS, cluster_perf.dxa.gmem_reads);
        CSR_READ_64(VX_CSR_MPM_DXA_GMEM_DEDUP, cluster_perf.dxa.gmem_dedup);
        CSR_READ_64(VX_CSR_MPM_DXA_LMEM_WRITES,cluster_perf.dxa.lmem_writes);
        CSR_READ_64(VX_CSR_MPM_DXA_GMEM_LT,    cluster_perf.dxa.total_latency);
        }
      } break;
    #endif
      default:
        std::cerr << "Error: invalid MPM CLASS: value=" << perf_class << std::endl;
        std::abort();
        break;
      }
    } else {
      std::cerr << "Error: invalid CSR read addr=0x"<< std::hex << addr << std::dec << std::endl;
      std::abort();
    }
  }
  return 0;
}

void CsrUnit::set_csr(uint32_t addr, Word value, uint32_t wid, uint32_t tid) {
  __unused(tid);
  auto& warp = core_->scheduler().warp(wid);
  switch (addr) {
  case VX_CSR_FFLAGS:
    warp.fcsr = (warp.fcsr & ~0x1F) | (value & 0x1F);
    break;
  case VX_CSR_FRM:
    warp.fcsr = (warp.fcsr & ~0xE0) | (value << 5);
    break;
  case VX_CSR_FCSR:
    warp.fcsr = value & 0xff;
    break;
  case VX_CSR_MSCRATCH:
    warp.mscratch = value;
    break;
  case VX_CSR_SATP:
    break;
  case VX_CSR_MSTATUS:
  case VX_CSR_MEDELEG:
  case VX_CSR_MIDELEG:
  case VX_CSR_MIE:
  case VX_CSR_MTVEC:
  case VX_CSR_MEPC:
  case VX_CSR_PMPCFG0:
  case VX_CSR_PMPADDR0:
  case VX_CSR_MNSTATUS:
  case VX_CSR_MCAUSE:
    break;
  default: {
      std::cerr << "Error: invalid CSR write addr=0x" << std::hex << addr << ", value=0x" << value << std::dec << std::endl;
      std::flush(std::cout);
      std::abort();
    }
  }
}

uint32_t CsrUnit::get_fpu_rm(uint32_t funct3, uint32_t wid, uint32_t tid) {
  return (funct3 == 0x7) ? this->get_csr(VX_CSR_FRM, wid, tid) : funct3;
}

void CsrUnit::update_fcrs(uint32_t fflags, uint32_t wid, uint32_t tid) {
  if (fflags) {
    this->set_csr(VX_CSR_FCSR, this->get_csr(VX_CSR_FCSR, wid, tid) | fflags, wid, tid);
    this->set_csr(VX_CSR_FFLAGS, this->get_csr(VX_CSR_FFLAGS, wid, tid) | fflags, wid, tid);
  }
}

void CsrUnit::execute(instr_trace_t* trace) {
  // Use trace->tmask captured at issue (matches what commit/writeback uses).
  auto& tmask = trace->tmask;
  auto& instr = *trace->instr_ptr;
  auto instrArgs = instr.get_args();
  auto csrArgs   = std::get<IntrCsrArgs>(instrArgs);
  auto csr_type  = std::get<CsrType>(trace->op_type);
  uint32_t wid   = trace->wid;
  uint32_t csr_addr = csrArgs.csr;
  uint32_t num_threads = NUM_THREADS;
  auto& rs1_data = trace->src_data[0];

  uint32_t thread_start = 0;
  for (; thread_start < num_threads; ++thread_start) {
    if (tmask.test(thread_start)) break;
  }

  trace->dst_data.assign(num_threads, reg_data_t{});
  auto& rd_data = trace->dst_data;

  switch (csr_type) {
  case CsrType::CSRRW:
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask.test(t)) continue;
      Word csr_value = this->get_csr(csr_addr, wid, t);
      auto src_v = csrArgs.is_imm ? csrArgs.imm : rs1_data[t].i;
      this->set_csr(csr_addr, src_v, wid, t);
      rd_data[t].i = csr_value;
    }
    break;
  case CsrType::CSRRS:
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask.test(t)) continue;
      Word csr_value = this->get_csr(csr_addr, wid, t);
      auto src_v = csrArgs.is_imm ? csrArgs.imm : rs1_data[t].i;
      if (src_v != 0) {
        this->set_csr(csr_addr, csr_value | src_v, wid, t);
      }
      rd_data[t].i = csr_value;
    }
    break;
  case CsrType::CSRRC:
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask.test(t)) continue;
      Word csr_value = this->get_csr(csr_addr, wid, t);
      auto src_v = csrArgs.is_imm ? csrArgs.imm : rs1_data[t].i;
      if (src_v != 0) {
        this->set_csr(csr_addr, csr_value & ~src_v, wid, t);
      }
      rd_data[t].i = csr_value;
    }
    break;
  default:
    std::abort();
  }
  DT(3, this->name() << " execute: op=" << csr_type << ", " << *trace);
}

void CsrUnit::on_tick() {
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    auto& input = Inputs.at(iw);
    if (input.empty())
      continue;
    auto& output = Outputs.at(iw);
    if (output.full())
      continue; // stall
    auto trace = input.peek();
    uint32_t delay = this->latency_of(trace);
    this->execute(trace);
    output.send(trace, delay);
    input.pop();
  }
}
