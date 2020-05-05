#include "simulator.h"
#include <iostream>
#include <iomanip>

uint64_t time_stamp = 0;

double sc_time_stamp() { 
  return time_stamp;
}

Simulator::Simulator(RAM *ram)
    : dram_stalled_(false) {
  ram_ = ram;
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

void Simulator::print_stats(std::ostream& out) {
  out << std::left;
  out << std::setw(24) << "# of total cycles:" << std::dec << time_stamp/2 << std::endl;
}

void Simulator::dbus_driver() {
  int dequeue_index = -1;
  for (int i = 0; i < dram_req_vec_.size(); i++) {
    if (dram_req_vec_[i].cycles_left > 0) {
      dram_req_vec_[i].cycles_left -= 1;
    }
    if ((dequeue_index == -1) 
     && (dram_req_vec_[i].cycles_left == 0)) {
      dequeue_index = i;
    }
  }

  if ((dequeue_index != -1) 
   && vortex_->dram_rsp_ready) {
    vortex_->dram_rsp_valid = 1;
    for (int i = 0; i < (GLOBAL_BLOCK_SIZE / 4); i++) {
      vortex_->dram_rsp_data[i] = dram_req_vec_[dequeue_index].data[i];
    }
    vortex_->dram_rsp_tag = dram_req_vec_[dequeue_index].tag;
    free(dram_req_vec_[dequeue_index].data);
    dram_req_vec_.erase(dram_req_vec_.begin() + dequeue_index);
  } else {
    vortex_->dram_rsp_valid = 0;
  }

  if (!dram_stalled_) {
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
      dram_req_vec_.push_back(dram_req);
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

#ifdef ENABLE_DRAM_STALLS
  dram_stalled_ = false;
  if (0 == ((time_stamp/2) % DRAM_STALLS_MODULO)) { 
    dram_stalled_ = true;
  } else
  if (dram_req_vec_.size() >= DRAM_RQ_SIZE) {
    dram_stalled_ = true;
  }  
#endif

  vortex_->dram_req_ready = ~dram_stalled_;
}

void Simulator::io_driver() {
  if (vortex_->io_req_write 
   && vortex_->io_req_addr == IO_BUS_ADDR_COUT) {
    uint32_t data_write = (uint32_t)vortex_->io_req_data;
    char c = (char)data_write;
    std::cerr << c;      
  }
  vortex_->io_req_ready = true;
}

void Simulator::reset() {  
  time_stamp = 0;
  vortex_->reset = 1;
  this->step();
  vortex_->reset = 0;
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
  trace_->dump(time_stamp);
#endif
  ++time_stamp;
}

void Simulator::wait(uint32_t cycles) {
  for (int i = 0; i < cycles; ++i) {
    this->step();
  }
}

bool Simulator::is_busy() {
  return (0 == vortex_->ebreak);
}

void Simulator::flush_caches(uint32_t mem_addr, uint32_t size) {  
  // send snoop requests to the caches
  printf("[sim] total cycles: %ld\n", time_stamp/2);
  // align address to LLC block boundaries
  auto aligned_addr_start = mem_addr / GLOBAL_BLOCK_SIZE;
  auto aligned_addr_end = (mem_addr + size + GLOBAL_BLOCK_SIZE - 1) / GLOBAL_BLOCK_SIZE;

  // submit snoop requests for the needed blocks
  vortex_->llc_snp_req_addr = aligned_addr_start;
  vortex_->llc_snp_req_valid = false;
  for (;;) {
    this->step();
    if (vortex_->llc_snp_req_valid) {
      vortex_->llc_snp_req_valid = false;
      if (vortex_->llc_snp_req_addr >= aligned_addr_end)
        break;
      vortex_->llc_snp_req_addr += 1;
    }    
    if (vortex_->llc_snp_req_ready) {
      vortex_->llc_snp_req_valid = true;      
    }
  }
  this->wait(PIPELINE_FLUSH_LATENCY);
}

bool Simulator::run() {
  // reset the device
  this->reset();

  // execute program
  while (!vortex_->ebreak) {
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
