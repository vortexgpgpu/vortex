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

Core::Core(const ArchDef &arch, Decoder &decoder, MemoryUnit &mem, Word id)
    : id_(id)
    , arch_(arch)
    , decoder_(decoder)
    , mem_(mem)
    , shared_mem_(1, SMEM_SIZE)
    , inst_in_schedule_("schedule")
    , inst_in_fetch_("fetch")
    , inst_in_decode_("decode")
    , inst_in_issue_("issue")
    , inst_in_execute_("execute")
    , inst_in_writeback_("writeback") {
  in_use_iregs_.resize(arch.num_warps(), 0);
  in_use_fregs_.resize(arch.num_warps(), 0);
  in_use_vregs_.reset();

  csrs_.resize(arch_.num_csrs(), 0);

  fcsrs_.resize(arch_.num_warps(), 0);

  barriers_.resize(arch_.num_barriers(), 0);

  warps_.resize(arch_.num_warps());
  for (int i = 0; i < arch_.num_warps(); ++i) {
    warps_[i] = std::make_shared<Warp>(this, i);
  }

  this->clear();
}

Core::~Core() {
  for (auto& buf : print_bufs_) {
    auto str = buf.second.str();
    if (!str.empty()) {
      std::cout << "#" << buf.first << ": " << str << std::endl;
    }
  }
}

void Core::clear() {
  for (int w = 0; w < arch_.num_warps(); ++w) {    
    in_use_iregs_[w].reset();
    in_use_fregs_[w].reset();    
  }
  stalled_warps_.reset();

  in_use_vregs_.reset();
  
  for (auto& csr : csrs_) {
    csr = 0;
  }

  for (auto& fcsr : fcsrs_) {
    fcsr = 0;
  }

  for (auto& barrier : barriers_) {
    barrier.reset();
  }
  
  for (auto warp : warps_) {
    warp->clear();
  }  

  inst_in_schedule_.clear();
  inst_in_fetch_.clear();
  inst_in_decode_.clear();
  inst_in_issue_.clear();
  inst_in_execute_.clear();
  inst_in_writeback_.clear();
  print_bufs_.clear();

  steps_  = 0;
  insts_  = 0;
  loads_  = 0;
  stores_ = 0;

  inst_in_schedule_.valid = true;
  warps_[0]->setTmask(0, true);

  ebreak_ = false;
}

void Core::step() {
  D(2, "###########################################################");

  steps_++;
  D(2, std::dec << "Core" << id_ << ": cycle: " << steps_);

  this->writeback();
  this->execute();
  this->issue();
  this->decode();
  this->fetch();
  this->schedule();

  DPN(2, std::flush);
}

void Core::schedule() {
  if (!inst_in_schedule_.enter(&inst_in_fetch_))
    return;

  bool foundSchedule = false;
  int scheduled_warp = inst_in_schedule_.wid;

  for (size_t wid = 0; wid < warps_.size(); ++wid) {
    // round robin scheduling
    scheduled_warp = (scheduled_warp + 1) % warps_.size();
    bool is_active = warps_[scheduled_warp]->active();
    bool stalled = stalled_warps_[scheduled_warp];
    if (is_active && !stalled) {
      foundSchedule = true;
      break;
    }
  }

  if (!foundSchedule)
    return;

  D(2, "Schedule: wid=" << scheduled_warp);
  inst_in_schedule_.wid = scheduled_warp;

  // advance pipeline
  inst_in_schedule_.next(&inst_in_fetch_);
}

void Core::fetch() {
  if (!inst_in_fetch_.enter(&inst_in_issue_))
    return;

  int wid = inst_in_fetch_.wid;
  
  auto active_threads_b = warps_[wid]->getActiveThreads();    
  warps_[wid]->step(&inst_in_fetch_);
  auto active_threads_a = warps_[wid]->getActiveThreads();   

  insts_ += active_threads_b;
  if (active_threads_b != active_threads_a) {
    D(3, "*** warp#" << wid << " active threads changed to " << active_threads_a);
  }

  if (inst_in_fetch_.stall_warp) {
    D(3, "*** warp#" << wid << " fetch stalled");
    stalled_warps_[wid] = true;
  }
  
  D(4, inst_in_fetch_);

  // advance pipeline
  inst_in_fetch_.next(&inst_in_issue_);
}

void Core::decode() {
  if (!inst_in_decode_.enter(&inst_in_issue_))
    return;
  
  // advance pipeline
  inst_in_decode_.next(&inst_in_issue_);
}

void Core::issue() {
  if (!inst_in_issue_.enter(&inst_in_execute_))
    return;

  bool in_use_regs = (inst_in_issue_.used_iregs & in_use_iregs_[inst_in_issue_.wid]) != 0 
                  || (inst_in_issue_.used_fregs & in_use_fregs_[inst_in_issue_.wid]) != 0 
                  || (inst_in_issue_.used_vregs & in_use_vregs_) != 0;
  
  if (in_use_regs) {      
    D(3, "*** Issue: registers not ready!");
    inst_in_issue_.stalled = true;
    return;
  } 

  switch (inst_in_issue_.rdest_type) {
  case 1:
    if (inst_in_issue_.rdest)
      in_use_iregs_[inst_in_issue_.wid][inst_in_issue_.rdest] = 1;
    break;
  case 2:
    in_use_fregs_[inst_in_issue_.wid][inst_in_issue_.rdest] = 1;
    break;
  case 3:
    in_use_vregs_[inst_in_issue_.rdest] = 1;
    break;
  default:  
    break;
  }

  // advance pipeline
  inst_in_issue_.next(&inst_in_execute_);
}

void Core::execute() {  
  if (!inst_in_execute_.enter(&inst_in_writeback_))
    return;

  // advance pipeline
  inst_in_execute_.next(&inst_in_writeback_);
}

void Core::writeback() {
  if (!inst_in_writeback_.enter(NULL))
    return;

  switch (inst_in_writeback_.rdest_type) {
  case 1:
    in_use_iregs_[inst_in_writeback_.wid][inst_in_writeback_.rdest] = 0;
    break;
  case 2:
    in_use_fregs_[inst_in_writeback_.wid][inst_in_writeback_.rdest] = 0;
    break;
  case 3:
    in_use_vregs_[inst_in_writeback_.rdest] = 0;
    break;
  default:  
    break;
  }

  if (inst_in_writeback_.stall_warp) {
    stalled_warps_[inst_in_writeback_.wid] = false;
    D(3, "*** warp#" << inst_in_writeback_.wid << " fetch released");
  }

  // advance pipeline
  inst_in_writeback_.next(NULL);
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
    return insts_;
  } else if (addr == CSR_MINSTRET_H) {
    // NumInsts
    return (Word)(insts_ >> 32);
  } else if (addr == CSR_MCYCLE) {
    // NumCycles
    return (Word)steps_;
  } else if (addr == CSR_MCYCLE_H) {
    // NumCycles
    return (Word)(steps_ >> 32);
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

Word Core::icache_fetch(Addr addr) {
  Word data;
  mem_.read(&data, addr, sizeof(Word), 0);
  return data;
}

Word Core::dcache_read(Addr addr, Size size) {
  ++loads_;
  Word data = 0;
#ifdef SM_ENABLE
  if ((addr >= (SMEM_BASE_ADDR - SMEM_SIZE))
   && ((addr + 3) < SMEM_BASE_ADDR)) {
     shared_mem_.read(&data, addr & (SMEM_SIZE-1), size);
     return data;
  }
#endif
  mem_.read(&data, addr, size, 0);
  return data;
}

void Core::dcache_write(Addr addr, Word data, Size size) {
  ++stores_;
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
  mem_.write(&data, addr, size, 0);
}

bool Core::running() const {
  return inst_in_fetch_.valid 
      || inst_in_decode_.valid 
      || inst_in_issue_.valid 
      || inst_in_execute_.valid 
      || inst_in_writeback_.valid;
}

void Core::printStats() const {
  std::cout << "Steps : " << steps_ << std::endl
            << "Insts : " << insts_ << std::endl
            << "Loads : " << loads_ << std::endl
            << "Stores: " << stores_ << std::endl;
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