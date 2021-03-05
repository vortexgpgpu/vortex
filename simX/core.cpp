#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include "types.h"
#include "util.h"
#include "archdef.h"
#include "mem.h"
#include "decode.h"
#include "core.h"
#include "debug.h"

#define INIT_TRACE(trace_inst)                                          \
  trace_inst.valid = false;                                             \
  trace_inst.PC = 0;                                                    \
  trace_inst.wid = schedule_w_;                                         \
  trace_inst.irs1 = -1;                                                 \
  trace_inst.irs2 = -1;                                                 \
  trace_inst.frs1 = -1;                                                 \
  trace_inst.frs2 = -1;                                                 \
  trace_inst.frs3 = -1;                                                 \
  trace_inst.frd = -1;                                                  \
  trace_inst.ird = -1;                                                  \
  trace_inst.vrs1 = -1;                                                 \
  trace_inst.vrs2 = -1;                                                 \
  trace_inst.vrd = -1;                                                  \
  trace_inst.is_lw = false;                                             \
  trace_inst.is_sw = false;                                             \
  if (trace_inst.mem_addresses != NULL)                                 \
    free(trace_inst.mem_addresses);                                     \
  trace_inst.mem_addresses = (unsigned *)malloc(32 * sizeof(unsigned)); \
  for (int tid = 0; tid < arch_.num_threads(); tid++)                   \
    trace_inst.mem_addresses[tid] = 0xdeadbeef;                         \
  trace_inst.mem_stall_cycles = 0;                                      \
  trace_inst.fetch_stall_cycles = 0;                                    \
  trace_inst.stall_warp = false;                                        \
  trace_inst.wspawn = false;                                            \
  trace_inst.stalled = false;

#define CPY_TRACE(drain, source)                          \
  drain.valid = source.valid;                             \
  drain.PC = source.PC;                                   \
  drain.wid = source.wid;                                 \
  drain.irs1 = source.irs1;                               \
  drain.irs2 = source.irs2;                               \
  drain.ird = source.ird;                                 \
  drain.frs1 = source.frs1;                               \
  drain.frs2 = source.frs2;                               \
  drain.frs3 = source.frs3;                               \
  drain.frd = source.frd;                                 \
  drain.vrs1 = source.vrs1;                               \
  drain.vrs2 = source.vrs2;                               \
  drain.vrd = source.vrd;                                 \
  drain.is_lw = source.is_lw;                             \
  drain.is_sw = source.is_sw;                             \
  for (int tid = 0; tid < arch_.num_threads(); tid++)     \
    drain.mem_addresses[tid] = source.mem_addresses[tid]; \
  drain.mem_stall_cycles = source.mem_stall_cycles;       \
  drain.fetch_stall_cycles = source.fetch_stall_cycles;   \
  drain.stall_warp = source.stall_warp;                   \
  drain.wspawn = source.wspawn;                           \
  drain.stalled = false;

using namespace vortex;

void printTrace(trace_inst_t *trace, const char *stage_name) {
  __unused(trace, stage_name);
  D(4, stage_name << ": valid=" << trace->valid);
  D(4, stage_name << ": PC=" << std::hex << trace->PC << std::dec);
  D(4, stage_name << ": wid=" << trace->wid);
  D(4, stage_name << ": rd=" << trace->ird << ", rs1=" << trace->irs1 << ", trs2=" << trace->irs2);
  D(4, stage_name << ": is_lw=" << trace->is_lw);
  D(4, stage_name << ": is_sw=" << trace->is_sw);
  D(4, stage_name << ": fetch_stall_cycles=" << trace->fetch_stall_cycles);
  D(4, stage_name << ": mem_stall_cycles=" << trace->mem_stall_cycles);
  D(4, stage_name << ": stall_warp=" << trace->stall_warp);
  D(4, stage_name << ": wspawn=" << trace->wspawn);
  D(4, stage_name << ": stalled=" << trace->stalled);
}

Core::Core(const ArchDef &arch, Decoder &decoder, MemoryUnit &mem, Word id)
    : id_(id)
    , arch_(arch)
    , decoder_(decoder)
    , mem_(mem)
    , shared_mem_(1, SMEM_SIZE)
    , steps_(0)
    , num_insts_(0) {  

  foundSchedule_ = true;
  schedule_w_ = 0;

  memset(&inst_in_fetch_, 0, sizeof(inst_in_fetch_));
  memset(&inst_in_decode_, 0, sizeof(inst_in_decode_));
  memset(&inst_in_scheduler_, 0, sizeof(inst_in_scheduler_));
  memset(&inst_in_exe_, 0, sizeof(inst_in_exe_));
  memset(&inst_in_lsu_, 0, sizeof(inst_in_lsu_));
  memset(&inst_in_wb_, 0, sizeof(inst_in_wb_));

  INIT_TRACE(inst_in_fetch_);
  INIT_TRACE(inst_in_decode_);
  INIT_TRACE(inst_in_scheduler_);
  INIT_TRACE(inst_in_exe_);
  INIT_TRACE(inst_in_lsu_);
  INIT_TRACE(inst_in_wb_);

  iRenameTable_.resize(arch.num_warps(), std::vector<bool>(arch.num_regs(), false));
  fRenameTable_.resize(arch.num_warps(), std::vector<bool>(arch.num_regs(), false));
  vRenameTable_.resize(arch.num_regs(), false);

  csrs_.resize(arch_.num_csrs());

  barriers_.resize(arch_.num_barriers(), 0);

  stalled_warps_.resize(arch.num_warps(), false);

  for (int i = 0; i < arch_.num_warps(); ++i) {
    warps_.emplace_back(this, i);
  }

  warps_[0].setTmask(0, true);
}

Core::~Core() {
  //--
}

void Core::step() {
  D(3, "###########################################################");

  steps_++;
  D(3, std::dec << "Core" << id_ << ": cycle: " << steps_);

  DPH(3, "stalled warps:");
  for (int i = 0; i < arch_.num_warps(); i++) {
    DPN(3, " " << stalled_warps_[i]);
  }
  DPN(3, "\n");

  this->writeback();
  this->load_store();
  this->execute_unit();
  this->scheduler();
  this->decode();
  this->fetch();

  DPN(3, std::flush);
}

void Core::warpScheduler() {
  foundSchedule_ = false;
  int next_warp = schedule_w_;
  for (size_t wid = 0; wid < warps_.size(); ++wid) {
    // round robin scheduling
    next_warp = (next_warp + 1) % warps_.size();
    bool is_active = warps_[next_warp].active();
    bool stalled = stalled_warps_[next_warp];
    if (is_active && !stalled) {
      foundSchedule_ = true;
      break;
    }
  }
  schedule_w_ = next_warp;
}

void Core::fetch() {
  if ((!inst_in_scheduler_.stalled) 
   && (inst_in_fetch_.fetch_stall_cycles == 0)) {
    INIT_TRACE(inst_in_fetch_);

    if (foundSchedule_) {
      auto active_threads_b = warps_[schedule_w_].getActiveThreads();
      num_insts_ = num_insts_ + warps_[schedule_w_].getActiveThreads();

      warps_[schedule_w_].step(&inst_in_fetch_);

      auto active_threads_a = warps_[schedule_w_].getActiveThreads();
      if (active_threads_b != active_threads_a) {
        D(3, "** warp #" << schedule_w_ << " active threads changed from " << active_threads_b << " to " << active_threads_a);
      }

      this->getCacheDelays(&inst_in_fetch_);

      if (inst_in_fetch_.stall_warp) {
        stalled_warps_[inst_in_fetch_.wid] = true;
      }
    }
    this->warpScheduler();
  } else {
    inst_in_fetch_.stalled = false;
    if (inst_in_fetch_.fetch_stall_cycles > 0)
      --inst_in_fetch_.fetch_stall_cycles;
  }

  printTrace(&inst_in_fetch_, "Fetch");
}

void Core::decode() {
  if ((inst_in_fetch_.fetch_stall_cycles == 0) 
   && !inst_in_scheduler_.stalled) {
    CPY_TRACE(inst_in_decode_, inst_in_fetch_);
    INIT_TRACE(inst_in_fetch_);
  }
}

void Core::scheduler() {
  if (!inst_in_scheduler_.stalled) {
    CPY_TRACE(inst_in_scheduler_, inst_in_decode_);
    INIT_TRACE(inst_in_decode_);
  }
}

void Core::load_store() {
  if ((inst_in_lsu_.mem_stall_cycles > 0) || inst_in_lsu_.stalled) {
    // LSU currently busy
    if ((inst_in_scheduler_.is_lw || inst_in_scheduler_.is_sw)) {
      inst_in_scheduler_.stalled = true;
    }
  } else {
    if (!inst_in_scheduler_.is_lw && !inst_in_scheduler_.is_sw)
      return;

    // Scheduler has LSU inst
    bool scheduler_srcs_busy = false;

    if (inst_in_scheduler_.irs1 > 0) {
      scheduler_srcs_busy = scheduler_srcs_busy || iRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.irs1];
    }

    if (inst_in_scheduler_.irs2 > 0) {
      scheduler_srcs_busy = scheduler_srcs_busy || iRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.irs2];
    }

    if (inst_in_scheduler_.frs1 >= 0) {
      scheduler_srcs_busy = scheduler_srcs_busy || fRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.frs1];
    }

    if (inst_in_scheduler_.frs2 >= 0) {
      scheduler_srcs_busy = scheduler_srcs_busy || fRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.frs2];
    }

    if (inst_in_scheduler_.frs3 >= 0) {
      scheduler_srcs_busy = scheduler_srcs_busy || fRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.frs3];
    }

    if (inst_in_scheduler_.vrs1 >= 0) {
      scheduler_srcs_busy = scheduler_srcs_busy || vRenameTable_[inst_in_scheduler_.vrs1];
    }
    if (inst_in_scheduler_.vrs2 >= 0) {
      scheduler_srcs_busy = scheduler_srcs_busy || vRenameTable_[inst_in_scheduler_.vrs2];
    }

    if (scheduler_srcs_busy) {
      inst_in_scheduler_.stalled = true;
    } else {        
      if (inst_in_scheduler_.ird > 0)
        iRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.ird] = true;

      if (inst_in_scheduler_.frd >= 0)
        fRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.frd] = true;
      
      if (inst_in_scheduler_.vrd >= 0)
        vRenameTable_[inst_in_scheduler_.vrd] = true;

      CPY_TRACE(inst_in_lsu_, inst_in_scheduler_);
      INIT_TRACE(inst_in_scheduler_);
    }
  }

  if (inst_in_lsu_.mem_stall_cycles > 0)
    inst_in_lsu_.mem_stall_cycles--;
}

void Core::execute_unit() {
  if (inst_in_scheduler_.is_lw || inst_in_scheduler_.is_sw)
    return;
  
  bool scheduler_srcs_busy = false;

  if (inst_in_scheduler_.irs1 > 0) {
    scheduler_srcs_busy = scheduler_srcs_busy || iRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.irs1];
  }

  if (inst_in_scheduler_.irs2 > 0) {
    scheduler_srcs_busy = scheduler_srcs_busy || iRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.irs2];
  }

  if (inst_in_scheduler_.frs1 >= 0) {
    scheduler_srcs_busy = scheduler_srcs_busy || fRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.frs1];
  }

  if (inst_in_scheduler_.frs2 >= 0) {
    scheduler_srcs_busy = scheduler_srcs_busy || fRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.frs2];
  }    

  if (inst_in_scheduler_.frs3 >= 0) {
    scheduler_srcs_busy = scheduler_srcs_busy || fRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.frs3];
  }

  if (inst_in_scheduler_.vrs1 >= 0) {
    scheduler_srcs_busy = scheduler_srcs_busy || vRenameTable_[inst_in_scheduler_.vrs1];
  }

  if (inst_in_scheduler_.vrs2 >= 0) {
    scheduler_srcs_busy = scheduler_srcs_busy || vRenameTable_[inst_in_scheduler_.vrs2];
  }

  if (scheduler_srcs_busy) {      
    D(3, "Execute: srcs not ready!");
    inst_in_scheduler_.stalled = true;
  } else {
    if (inst_in_scheduler_.ird > 0) {
      iRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.ird] = true;
    }

    if (inst_in_scheduler_.frd >= 0) {
      fRenameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.frd] = true;
    }

    if (inst_in_scheduler_.vrd >= 0) {
      vRenameTable_[inst_in_scheduler_.vrd] = true;
    }

    CPY_TRACE(inst_in_exe_, inst_in_scheduler_);
    INIT_TRACE(inst_in_scheduler_);
  }
}

void Core::writeback() {
  if (inst_in_wb_.ird > 0) {
    iRenameTable_[inst_in_wb_.wid][inst_in_wb_.ird] = false;
  }

  if (inst_in_wb_.frd >= 0) {
    fRenameTable_[inst_in_wb_.wid][inst_in_wb_.frd] = false;
  }

  if (inst_in_wb_.vrd >= 0) {
    vRenameTable_[inst_in_wb_.vrd] = false;
  }

  if (inst_in_wb_.stall_warp) {
    stalled_warps_[inst_in_wb_.wid] = false;
  }

  INIT_TRACE(inst_in_wb_);

  bool serviced_exe = false;
  if ((inst_in_exe_.ird > 0) 
   || (inst_in_exe_.frd >= 0) 
   || (inst_in_exe_.vrd >= 0) 
   || (inst_in_exe_.stall_warp)) {
    CPY_TRACE(inst_in_wb_, inst_in_exe_);
    INIT_TRACE(inst_in_exe_);
    serviced_exe = true;
  }

  if (inst_in_lsu_.is_sw) {
    INIT_TRACE(inst_in_lsu_);
  } else {
    if (((inst_in_lsu_.ird > 0) 
      || (inst_in_lsu_.frd >= 0) 
      || (inst_in_lsu_.vrd >= 0)) 
     && (inst_in_lsu_.mem_stall_cycles == 0)) {
      if (serviced_exe) {
        // Stalling LSU because EXE is busy
        inst_in_lsu_.stalled = true;
      } else {
        CPY_TRACE(inst_in_wb_, inst_in_lsu_);
        INIT_TRACE(inst_in_lsu_);
      }
    }
  }
}

Word Core::get_csr(Addr addr, int tid, int wid) {
  if (addr == CSR_WTID) {
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
  } else if (addr == CSR_NT) {
    // Number of threads per warp
    return arch_.num_threads();
  } else if (addr == CSR_NW) {
    // Number of warps per core
    return arch_.num_warps();
  } else if (addr == CSR_NC) {
    // Number of cores
    return arch_.num_cores();
  } else if (addr == CSR_INSTRET) {
    // NumInsts
    return num_insts_;
  } else if (addr == CSR_INSTRET_H) {
    // NumInsts
    return (Word)(num_insts_ >> 32);
  } else if (addr == CSR_CYCLE) {
    // NumCycles
    return (Word)steps_;
  } else if (addr == CSR_CYCLE_H) {
    // NumCycles
    return (Word)(steps_ >> 32);
  } else {
    return csrs_.at(addr);
  }
}

void Core::set_csr(Addr addr, Word value) {
  csrs_.at(addr) = value;
}

void Core::barrier(int bar_id, int count, int warp_id) {
  auto& barrier = barriers_.at(bar_id);
  barrier.set(warp_id);
  if (barrier.count() < (size_t)count)    
    return;
  for (int i = 0; i < arch_.num_warps(); ++i) {
    if (barrier.test(i)) {
      warps_.at(i).activate();
    }
  }
  barrier.reset();
}

Word Core::icache_fetch(Addr addr, bool sup) {
  return mem_.fetch(addr, sup);
}

Word Core::dcache_read(Addr addr, bool sup) {
#ifdef SM_ENABLE
  if ((addr >= (SHARED_MEM_BASE_ADDR - SMEM_SIZE))
   && ((addr + 4) <= SHARED_MEM_BASE_ADDR)) {
     return shared_mem_.read(addr & (SMEM_SIZE-1));
  }
#endif
  return mem_.read(addr, sup);
}

void Core::dcache_write(Addr addr, Word data, bool sup, Size size) {
#ifdef SM_ENABLE
  if ((addr >= (SHARED_MEM_BASE_ADDR - SMEM_SIZE))
   && ((addr + 4) <= SHARED_MEM_BASE_ADDR)) {
     shared_mem_.write(addr & (SMEM_SIZE-1), data);
     return;
  }
#endif
  mem_.write(addr, data, sup, size);
}

void Core::getCacheDelays(trace_inst_t *trace_inst) {
  trace_inst->fetch_stall_cycles += 1;
  if (trace_inst->is_sw || trace_inst->is_lw) {
    trace_inst->mem_stall_cycles += 3;
  }
}

bool Core::running() const {
  bool stages_have_valid = inst_in_fetch_.valid 
                        || inst_in_decode_.valid 
                        || inst_in_scheduler_.valid 
                        || inst_in_lsu_.valid 
                        || inst_in_exe_.valid 
                        || inst_in_wb_.valid;

  if (stages_have_valid)
    return true;

  for (unsigned i = 0; i < warps_.size(); ++i) {
    if (warps_[i].active()) {
      return true;
    }
  }
  return false;
}

void Core::printStats() const {
  std::cout << "Total steps: " << steps_ << std::endl;
  for (unsigned i = 0; i < warps_.size(); ++i) {
    std::cout << "=== Warp " << i << " ===" << std::endl;
    warps_[i].printStats();
  }
}