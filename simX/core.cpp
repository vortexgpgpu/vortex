#include <iostream>
#include <iomanip>
#include <string.h>

// #define USE_DEBUG 7
// #define PRINT_ACTIVE_THREADS

#include "types.h"
#include "util.h"
#include "archdef.h"
#include "mem.h"
#include "decode.h"
#include "core.h"
#include "debug.h"

#define INIT_TRACE(trace_inst)                                          \
  trace_inst.valid_inst = false;                                        \
  trace_inst.pc = 0;                                                    \
  trace_inst.wid = schedule_w_;                                         \
  trace_inst.rs1 = -1;                                                  \
  trace_inst.rs2 = -1;                                                  \
  trace_inst.rd = -1;                                                   \
  trace_inst.vs1 = -1;                                                  \
  trace_inst.vs2 = -1;                                                  \
  trace_inst.vd = -1;                                                   \
  trace_inst.is_lw = false;                                             \
  trace_inst.is_sw = false;                                             \
  if (trace_inst.mem_addresses != NULL)                                 \
    free(trace_inst.mem_addresses);                                     \
  trace_inst.mem_addresses = (unsigned *)malloc(32 * sizeof(unsigned)); \
  for (ThdNum tid = 0; tid < arch_.getNumThreads(); tid++)              \
    trace_inst.mem_addresses[tid] = 0xdeadbeef;                         \
  trace_inst.mem_stall_cycles = 0;                                      \
  trace_inst.fetch_stall_cycles = 0;                                    \
  trace_inst.stall_warp = false;                                        \
  trace_inst.wspawn = false;                                            \
  trace_inst.stalled = false;

#define CPY_TRACE(drain, source)                          \
  drain.valid_inst = source.valid_inst;                   \
  drain.pc = source.pc;                                   \
  drain.wid = source.wid;                                 \
  drain.rs1 = source.rs1;                                 \
  drain.rs2 = source.rs2;                                 \
  drain.rd = source.rd;                                   \
  drain.vs1 = source.vs1;                                 \
  drain.vs2 = source.vs2;                                 \
  drain.vd = source.vd;                                   \
  drain.is_lw = source.is_lw;                             \
  drain.is_sw = source.is_sw;                             \
  for (ThdNum tid = 0; tid < arch_.getNumThreads(); tid++)\
    drain.mem_addresses[tid] = source.mem_addresses[tid]; \
  drain.mem_stall_cycles = source.mem_stall_cycles;       \
  drain.fetch_stall_cycles = source.fetch_stall_cycles;   \
  drain.stall_warp = source.stall_warp;                   \
  drain.wspawn = source.wspawn;                           \
  drain.stalled = false;

using namespace vortex;

void printTrace(trace_inst_t *trace, const char *stage_name) {
  __unused(trace, stage_name);
  D(3, stage_name << ": valid=" << trace->valid_inst);
  D(3, stage_name << ": PC=" << std::hex << trace->pc << std::dec);
  D(3, stage_name << ": wid=" << trace->wid);
  D(3, stage_name << ": rd=" << trace->rd << ", rs1=" << trace->rs1 << ", trs2=" << trace->rs2);
  D(3, stage_name << ": is_lw=" << trace->is_lw);
  D(3, stage_name << ": is_sw=" << trace->is_sw);
  D(3, stage_name << ": fetch_stall_cycles=" << trace->fetch_stall_cycles);
  D(3, stage_name << ": mem_stall_cycles=" << trace->mem_stall_cycles);
  D(3, stage_name << ": stall_warp=" << trace->stall_warp);
  D(3, stage_name << ": wspawn=" << trace->wspawn);
  D(3, stage_name << ": stalled=" << trace->stalled);
}

Core::Core(const ArchDef &arch, Decoder &decoder, MemoryUnit &mem, Word id)
    : id_(id)
    , arch_(arch)
    , decoder_(decoder)
    , mem_(mem)
    , steps_(0)
    , num_instructions_(0) {
  release_warp_ = false;
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

  for (int i = 0; i < 32; i++) {
    stalled_warps_[i] = false;
    for (int j = 0; j < 32; j++) {
      renameTable_[i][j] = true;
    }
  }

  for (int i = 0; i < 32; i++) {
    vecRenameTable_[i] = true;
  }

  for (unsigned i = 0; i < arch_.getNumWarps(); ++i) {
    warps_.push_back(Warp(this, i));
  }

  warps_[0].setActiveThreads(1);
  warps_[0].setSpawned(true);
}

Core::~Core() {
  //--
}

void Core::step() {
  D(3, "###########################################################");

  steps_++;
  D(3, "cycle: " << steps_);

  DPH(3, "stalled warps:");
  for (ThdNum widd = 0; widd < arch_.getNumWarps(); widd++) {
    DPN(3, " " << stalled_warps_[widd]);
  }
  DPN(3, "\n");

  // cout << "About to call writeback" << std::endl;
  this->writeback();
  // cout << "About to call load_store" << std::endl;
  this->load_store();
  // cout << "About to call execute_unit" << std::endl;
  this->execute_unit();
  // cout << "About to call scheduler" << std::endl;
  this->scheduler();
  // cout << "About to call decode" << std::endl;
  this->decode();
  // D(3, "About to call fetch" << std::flush);
  this->fetch();
  // D(3, "Finished fetch" << std::flush);

  if (release_warp_) {
    release_warp_ = false;
    stalled_warps_[release_warp_num_] = false;
  }

  DPN(3, std::flush);
}

void Core::warpScheduler() {
  foundSchedule_ = false;
  int next_warp = schedule_w_;
  for (size_t wid = 0; wid < warps_.size(); ++wid) {
    // round robin scheduling
    next_warp = (next_warp + 1) % warps_.size();

    bool has_active_threads = (warps_[next_warp].getActiveThreads() > 0);
    bool stalled = stalled_warps_[next_warp];

    if (has_active_threads && !stalled) {
      foundSchedule_ = true;
      break;
    }
  }
  schedule_w_ = next_warp;
}

void Core::fetch() {

  // D(-1, "Found schedule: " << foundSchedule_);

  if ((!inst_in_scheduler_.stalled) 
   && (inst_in_fetch_.fetch_stall_cycles == 0)) {
    // CPY_TRACE(inst_in_decode_, inst_in_fetch_);
    // if (warps_[schedule_w_].activeThreads)
    {
      INIT_TRACE(inst_in_fetch_);

      if (foundSchedule_) {
        auto active_threads_b = warps_[schedule_w_].getActiveThreads();

        num_instructions_ = num_instructions_ + warps_[schedule_w_].getActiveThreads();
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
    }
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
  //printTrace(&inst_in_decode_, "Decode");
}

void Core::scheduler() {
  if (!inst_in_scheduler_.stalled) {
    CPY_TRACE(inst_in_scheduler_, inst_in_decode_);
    INIT_TRACE(inst_in_decode_);
  }
  //printTrace(&inst_in_scheduler_, "Scheduler");
}

void Core::load_store() {
  if ((inst_in_lsu_.mem_stall_cycles > 0) || (inst_in_lsu_.stalled)) {
    // LSU currently busy
    if ((inst_in_scheduler_.is_lw || inst_in_scheduler_.is_sw)) {
      inst_in_scheduler_.stalled = true;
    }
  } else {
    // LSU not busy
    if (inst_in_scheduler_.is_lw || inst_in_scheduler_.is_sw) {
      // Scheduler has LSU inst
      bool scheduler_srcs_ready = true;
      if (inst_in_scheduler_.rs1 > 0) {
        scheduler_srcs_ready = scheduler_srcs_ready && renameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.rs1];
      }

      if (inst_in_scheduler_.rs2 > 0) {
        scheduler_srcs_ready = scheduler_srcs_ready && renameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.rs2];
      }

      if (inst_in_scheduler_.vs1 > 0) {
        scheduler_srcs_ready = scheduler_srcs_ready && vecRenameTable_[inst_in_scheduler_.vs1];
      }
      if (inst_in_scheduler_.vs2 > 0) {
        scheduler_srcs_ready = scheduler_srcs_ready && vecRenameTable_[inst_in_scheduler_.vs2];
      }

      if (scheduler_srcs_ready) {
        if (inst_in_scheduler_.rd != -1)
          renameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.rd] = false;
        if (inst_in_scheduler_.rd != -1)
          vecRenameTable_[inst_in_scheduler_.vd] = false;
        CPY_TRACE(inst_in_lsu_, inst_in_scheduler_);
        INIT_TRACE(inst_in_scheduler_);
      } else {
        inst_in_scheduler_.stalled = true;
        // INIT_TRACE(inst_in_lsu_);
      }
    } else {
      // INIT_TRACE(inst_in_lsu_);
    }
  }

  if (inst_in_lsu_.mem_stall_cycles > 0)
    inst_in_lsu_.mem_stall_cycles--;

  //printTrace(&inst_in_lsu_, "LSU");
}

void Core::execute_unit() {
  // EXEC is always not busy
  if (inst_in_scheduler_.is_lw || inst_in_scheduler_.is_sw) {
    // Not an execute instruction
    // INIT_TRACE(inst_in_exe_);
  } else {
    bool scheduler_srcs_ready = true;
    if (inst_in_scheduler_.rs1 > 0) {
      scheduler_srcs_ready = scheduler_srcs_ready && renameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.rs1];
      // cout << "Rename RS1: " << inst_in_scheduler_.rs1 << " is " << renameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.rs1] << " wid: " << inst_in_scheduler_.wid << '\n';
    }

    if (inst_in_scheduler_.rs2 > 0) {
      scheduler_srcs_ready = scheduler_srcs_ready && renameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.rs2];
      // cout << "Rename RS2: " << inst_in_scheduler_.rs1 << " is " << renameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.rs2] << " wid: " << inst_in_scheduler_.wid << '\n';
    }

    // cout << "About to check vs*\n" << std::flush;
    if (inst_in_scheduler_.vs1 > 0) {
      scheduler_srcs_ready = scheduler_srcs_ready && vecRenameTable_[inst_in_scheduler_.vs1];
    }
    if (inst_in_scheduler_.vs2 > 0) {
      scheduler_srcs_ready = scheduler_srcs_ready && vecRenameTable_[inst_in_scheduler_.vs2];
    }
    // cout << "Finished sources\n" << std::flush;

    if (scheduler_srcs_ready) {
      if (inst_in_scheduler_.rd != -1) {
        // cout << "rename setting rd: " << inst_in_scheduler_.rd << " to not useabel wid: " << inst_in_scheduler_.wid << '\n';
        renameTable_[inst_in_scheduler_.wid][inst_in_scheduler_.rd] = false;
      }

      // cout << "About to check vector wb: " << inst_in_scheduler_.vd << "\n" << std::flush;
      if (inst_in_scheduler_.vd != -1) {
        vecRenameTable_[inst_in_scheduler_.vd] = false;
      }
      // cout << "Finished wb checking" << "\n" << std::flush;
      CPY_TRACE(inst_in_exe_, inst_in_scheduler_);
      INIT_TRACE(inst_in_scheduler_);
      // cout << "Finished trace copying and clearning" << "\n" << std::flush;
    } else {
      D(3, "Execute: srcs not ready!");
      inst_in_scheduler_.stalled = true;
      // INIT_TRACE(inst_in_exe_);
    }
  }

  //printTrace(&inst_in_exe_, "EXE");
  // INIT_TRACE(inst_in_exe_);
}

void Core::writeback() {
  if (inst_in_wb_.rd > 0)
    renameTable_[inst_in_wb_.wid][inst_in_wb_.rd] = true;
  if (inst_in_wb_.vd > 0)
    vecRenameTable_[inst_in_wb_.vd] = true;

  if (inst_in_wb_.stall_warp) {
    stalled_warps_[inst_in_wb_.wid] = false;
    // release_warp_ = true;
    // release_warp_num_ = inst_in_wb_.wid;
  }

  INIT_TRACE(inst_in_wb_);

  bool serviced_exe = false;
  if ((inst_in_exe_.rd > 0) || (inst_in_exe_.stall_warp)) {
    CPY_TRACE(inst_in_wb_, inst_in_exe_);
    INIT_TRACE(inst_in_exe_);
    serviced_exe = true;
    // cout << "WRITEBACK SERVICED EXE\n";
  }

  if (inst_in_lsu_.is_sw) {
    INIT_TRACE(inst_in_lsu_);
  } else {
    if (((inst_in_lsu_.rd > 0) || (inst_in_lsu_.vd > 0)) && (inst_in_lsu_.mem_stall_cycles == 0)) {
      if (serviced_exe) {
        D(3, "$$$$$$$$$$$$$$$$$$$$ Stalling LSU because EXE is being used");
        inst_in_lsu_.stalled = true;
      } else {
        CPY_TRACE(inst_in_wb_, inst_in_lsu_);
        INIT_TRACE(inst_in_lsu_);
      }
    }
  }
}

void Core::getCacheDelays(trace_inst_t *trace_inst) {
  trace_inst->fetch_stall_cycles += 3;
  if (trace_inst->is_sw || trace_inst->is_lw) {
    trace_inst->mem_stall_cycles += 5;
  }
}

bool Core::running() const {
  bool stages_have_valid = inst_in_fetch_.valid_inst 
                        || inst_in_decode_.valid_inst 
                        || inst_in_scheduler_.valid_inst 
                        || inst_in_lsu_.valid_inst 
                        || inst_in_exe_.valid_inst 
                        || inst_in_wb_.valid_inst;

  if (stages_have_valid)
    return true;

  for (unsigned i = 0; i < warps_.size(); ++i)
    if (warps_[i].running()) {
      return true;
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