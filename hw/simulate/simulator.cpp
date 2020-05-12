#include "simulator.h"
#include <iostream>
#include <iomanip>

uint64_t timestamp = 0;

double sc_time_stamp() { 
  return timestamp;
}

Simulator::Simulator() {    
  ram_ = nullptr;
  vortex_ = new VVortex_Socket();

#ifdef VCD_OUTPUT
  Verilated::traceEverOn(true);
  trace_ = new VerilatedVcdC;
  vortex_->trace(trace_, 99);
  trace_->open("trace.vcd");
#endif  
}

Simulator::~Simulator() {
#ifdef VCD_OUTPUT
  trace_->close();
#endif
  delete vortex_;
}

void Simulator::attach_ram(RAM* ram) {
  ram_ = ram;
  dram_rsp_vec_.clear();
}

void Simulator::print_stats(std::ostream& out) {
  out << std::left;
  out << std::setw(24) << "# of total cycles:" << std::dec << timestamp/2 << std::endl;
}

void Simulator::dbus_driver() {
  if (ram_ == nullptr) {
    vortex_->dram_req_ready = false;
    return;
  }

  // handle DRAM response cycle
  int dequeue_index = -1;
  for (int i = 0; i < dram_rsp_vec_.size(); i++) {
    if (dram_rsp_vec_[i].cycles_left > 0) {
      dram_rsp_vec_[i].cycles_left -= 1;
    }
    if ((dequeue_index == -1) 
     && (dram_rsp_vec_[i].cycles_left == 0)) {
      dequeue_index = i;
    }
  }

  // handle DRAM response message
  if ((dequeue_index != -1) 
   && vortex_->dram_rsp_ready) {
    vortex_->dram_rsp_valid = 1;
    for (int i = 0; i < (GLOBAL_BLOCK_SIZE / 4); i++) {
      vortex_->dram_rsp_data[i] = dram_rsp_vec_[dequeue_index].data[i];
    }
    vortex_->dram_rsp_tag = dram_rsp_vec_[dequeue_index].tag;
    free(dram_rsp_vec_[dequeue_index].data);
    dram_rsp_vec_.erase(dram_rsp_vec_.begin() + dequeue_index);
  } else {
    vortex_->dram_rsp_valid = 0;
  }

  // handle DRAM stalls
  bool dram_stalled = false;
#ifdef ENABLE_DRAM_STALLS
  if (0 == ((timestamp/2) % DRAM_STALLS_MODULO)) { 
    dram_stalled = true;
  } else
  if (dram_rsp_vec_.size() >= DRAM_RQ_SIZE) {
    dram_stalled = true;
  }
#endif

  // handle DRAM requests
  if (!dram_stalled) {
    if (vortex_->dram_req_read) {
      dram_req_t dram_req;
      dram_req.cycles_left = DRAM_LATENCY;     
      dram_req.data = (unsigned*)malloc(GLOBAL_BLOCK_SIZE);
      dram_req.tag = vortex_->dram_req_tag;

      unsigned base_addr = (vortex_->dram_req_addr * GLOBAL_BLOCK_SIZE);
      for (int i = 0; i < (GLOBAL_BLOCK_SIZE / 4); i++) {
        unsigned curr_addr = base_addr + (i * 4);
        unsigned data_rd;
        ram_->getWord(curr_addr, &data_rd);
        dram_req.data[i] = data_rd;
      }
      dram_rsp_vec_.push_back(dram_req);
    }

    if (vortex_->dram_req_write) {
      unsigned base_addr = (vortex_->dram_req_addr * GLOBAL_BLOCK_SIZE);
      for (int i = 0; i < (GLOBAL_BLOCK_SIZE / 4); i++) {
        unsigned curr_addr = base_addr + (i * 4);
        unsigned data_wr = vortex_->dram_req_data[i];
        ram_->writeWord(curr_addr, &data_wr);
      }
    }
  }

  vortex_->dram_req_ready = ~dram_stalled;
}

void Simulator::io_driver() {
  if (vortex_->io_req_write 
   && vortex_->io_req_addr == IO_BUS_ADDR_COUT) {
    uint32_t data_write = (uint32_t)vortex_->io_req_data;
    char c = (char)data_write;
    std::cout << c;      
  }
  vortex_->io_req_ready = true;
}

void Simulator::reset() {     
#ifndef NDEBUG
  std::cout << timestamp << ": [sim] reset()" << std::endl;
#endif 
  vortex_->reset = 1;
  this->step();  
  vortex_->reset = 0;

  dram_rsp_vec_.clear();
}

void Simulator::step() {
  vortex_->clk = 0;
  this->eval();

  vortex_->clk = 1;
  this->eval();

  dbus_driver();
  io_driver();
}

void Simulator::eval() {
  vortex_->eval();
#ifdef VCD_OUTPUT
  trace_->dump(timestamp);
#endif
  ++timestamp;
}

void Simulator::wait(uint32_t cycles) {
  for (int i = 0; i < cycles; ++i) {
    this->step();
  }
}

bool Simulator::is_busy() {
  return vortex_->busy;
}

void Simulator::flush_caches(uint32_t mem_addr, uint32_t size) {  
#ifndef NDEBUG
  std::cout << timestamp << ": [sim] flush_caches()" << std::endl;
#endif
  // align address to LLC block boundaries
  auto aligned_addr_start = mem_addr / GLOBAL_BLOCK_SIZE;
  auto aligned_addr_end = (mem_addr + size + GLOBAL_BLOCK_SIZE - 1) / GLOBAL_BLOCK_SIZE;
  int outstanding_snp_reqs = 0;

  // submit snoop requests for the needed blocks
  vortex_->snp_req_addr  = aligned_addr_start;
  vortex_->snp_req_valid = true;
  vortex_->snp_rsp_ready = true;
  for (;;) {
    this->step();
    if (vortex_->snp_rsp_valid) {
      assert(outstanding_snp_reqs > 0);
      --outstanding_snp_reqs;
    }
    if (vortex_->snp_req_valid && vortex_->snp_req_ready) {
      ++outstanding_snp_reqs;
      vortex_->snp_req_addr += 1;
      if (vortex_->snp_req_addr >= aligned_addr_end) {
        vortex_->snp_req_valid = false;        
      }
    }
    if (!vortex_->snp_req_valid 
     && 0 == outstanding_snp_reqs) {
      break;
    }
  }  
}

bool Simulator::run() {
#ifndef NDEBUG
  std::cout << timestamp << ": [sim] run()" << std::endl;
#endif 

  // reset the device
  this->reset();

  // execute program
  while (vortex_->busy 
      && !vortex_->ebreak) {
    this->step();
  }

  // wait 5 cycles to flush the pipeline
  this->wait(5);

  // check riscv-tests PASSED/FAILED status
#if (NUM_CLUSTERS == 1 && NUM_CORES == 1)
  int status = (int)vortex_->Vortex_Socket->genblk1__DOT__Vortex_Cluster->genblk1__BRA__0__KET____DOT__vortex_core->back_end->writeback->last_data_wb & 0xf;
#else
#if (NUM_CLUSTERS == 1)
  int status = (int)vortex_->Vortex_Socket->genblk1__DOT__Vortex_Cluster->genblk1__BRA__0__KET____DOT__vortex_core->back_end->writeback->last_data_wb & 0xf;
#else
  int status = (int)vortex_->Vortex_Socket->genblk2__DOT__genblk1__BRA__0__KET____DOT__Vortex_Cluster->genblk1__BRA__0__KET____DOT__vortex_core->back_end->writeback->last_data_wb & 0xf;
#endif
#endif

  return (status == 1);
}
