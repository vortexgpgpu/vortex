#include "Vortex_SOC.h"

unsigned long time_stamp = 0;

double sc_time_stamp() {
  return time_stamp / 1000.0;
}

Vortex_SOC::Vortex_SOC(RAM *ram)
    : start_pc(0), curr_cycle(0), stop(true), unit_test(true), stats_static_inst(0), stats_dynamic_inst(-1),
      stats_total_cycles(0), stats_fwd_stalls(0), stats_branch_stalls(0),
      debug_state(0), ibus_state(0), dbus_state(0), debug_return(0),
      debug_wait_num(0), debug_inst_num(0), debug_end_wait(0), debug_debugAddr(0) {
  this->ram = ram;
  this->vortex = new VVortex_SOC;
#ifdef VCD_OUTPUT
  Verilated::traceEverOn(true);
  this->m_trace = new VerilatedVcdC;
  this->vortex->trace(m_trace, 99);
  this->m_trace->open("trace.vcd");
#endif
  this->results.open("../results.txt");
}

Vortex_SOC::~Vortex_SOC() {
#ifdef VCD_OUTPUT
  m_trace->close();
#endif
  this->results.close();
  delete this->vortex;
}

void Vortex_SOC::print_stats(bool cycle_test) {

  if (cycle_test) {
    this->results << std::left;
    // this->results << "# Static Instructions:\t" << std::dec << this->stats_static_inst << std::endl;
    this->results << std::setw(24) << "# Dynamic Instructions:" << std::dec << this->stats_dynamic_inst << std::endl;
    this->results << std::setw(24) << "# of total cycles:" << std::dec << this->stats_total_cycles << std::endl;
    this->results << std::setw(24) << "# of forwarding stalls:" << std::dec << this->stats_fwd_stalls << std::endl;
    this->results << std::setw(24) << "# of branch stalls:" << std::dec << this->stats_branch_stalls << std::endl;
    this->results << std::setw(24) << "# CPI:" << std::dec << (double)this->stats_total_cycles / (double)this->stats_dynamic_inst << std::endl;
    this->results << std::setw(24) << "# time to simulate: " << std::dec << this->stats_sim_time << " milliseconds" << std::endl;
  } else {
    this->results << std::left;
    this->results << std::setw(24) << "# of total cycles:" << std::dec << this->stats_total_cycles << std::endl;
    this->results << std::setw(24) << "# time to simulate: " << std::dec << this->stats_sim_time << " milliseconds" << std::endl;
  }

  uint32_t status;
  ram->getWord(0, &status);

  if (this->unit_test) {
    if (status == 1) {
      this->results << std::setw(24) << "# GRADE:"
                    << "PASSING\n";
    } else {
      this->results << std::setw(24) << "# GRADE:"
                    << "Failed on test: " << status << "\n";
    }
  } else {
    this->results << std::setw(24) << "# GRADE:"
                  << "N/A [NOT A UNIT TEST]\n";
  }

  this->stats_static_inst = 0;
  this->stats_dynamic_inst = -1;
  this->stats_total_cycles = 0;
  this->stats_fwd_stalls = 0;
  this->stats_branch_stalls = 0;
}

bool Vortex_SOC::ibus_driver() {
  return false;
}

void Vortex_SOC::io_handler() {
  // std::cout << "Checking\n";
  for (int c = 0; c < vortex->number_cores; c++) {
    if (vortex->io_valid[c]) {
      uint32_t data_write = (uint32_t)vortex->io_data[c];
      // std::cout << "IO VALID!\n";
      char c = (char)data_write;
      std::cerr << c;
      // std::cout << c;

      std::cout << std::flush;
    }
  }
}

bool Vortex_SOC::dbus_driver() {
  // Iterate through each element, and get pop index
  int dequeue_index = -1;
  bool dequeue_valid = false;
  for (int i = 0; i < this->dram_req_vec.size(); i++) {
    if (this->dram_req_vec[i].cycles_left > 0) {
      this->dram_req_vec[i].cycles_left -= 1;
    }

    if ((this->dram_req_vec[i].cycles_left == 0) && (!dequeue_valid)) {
      dequeue_index = i;
      dequeue_valid = true;
    }
  }

  if (vortex->out_dram_req) {
    if (vortex->out_dram_req_read) {
      // Need to add an element
      dram_req_t dram_req;
      dram_req.cycles_left = vortex->out_dram_expected_lat;
      dram_req.data_length = vortex->out_dram_req_size / 4;
      dram_req.base_addr = vortex->out_dram_req_addr;
      dram_req.data = (unsigned *)malloc(dram_req.data_length * sizeof(unsigned));

      for (int i = 0; i < dram_req.data_length; i++) {
        unsigned curr_addr = dram_req.base_addr + (i * 4);
        unsigned data_rd;
        ram->getWord(curr_addr, &data_rd);
        dram_req.data[i] = data_rd;
      }
      // std::cout << "Fill Req -> Addr: " << std::hex << dram_req.base_addr << std::dec << "\n";
      this->dram_req_vec.push_back(dram_req);
    }

    if (vortex->out_dram_req_write) {
      unsigned base_addr = vortex->out_dram_req_addr;
      unsigned data_length = vortex->out_dram_req_size / 4;

      for (int i = 0; i < data_length; i++) {
        unsigned curr_addr = base_addr + (i * 4);
        unsigned data_wr = vortex->out_dram_req_data[i];
        ram->writeWord(curr_addr, &data_wr);
      }
    }
  }

  if (vortex->out_dram_fill_accept && dequeue_valid) {
    vortex->out_dram_fill_rsp = 1;
    vortex->out_dram_fill_rsp_addr = this->dram_req_vec[dequeue_index].base_addr;
    // std::cout << "Fill Rsp -> Addr: " << std::hex << (this->dram_req_vec[dequeue_index].base_addr) << std::dec << "\n";

    for (int i = 0; i < this->dram_req_vec[dequeue_index].data_length; i++) {
      vortex->out_dram_fill_rsp_data[i] = this->dram_req_vec[dequeue_index].data[i];
    }
    free(this->dram_req_vec[dequeue_index].data);

    this->dram_req_vec.erase(this->dram_req_vec.begin() + dequeue_index);
  } else {
    vortex->out_dram_fill_rsp = 0;
    vortex->out_dram_fill_rsp_addr = 0;
  }

  return false;
}

void Vortex_SOC::reset() {
  vortex->reset = 1;
  this->step();
  vortex->reset = 0;
}

void Vortex_SOC::step() {  
  vortex->clk = 0;
  vortex->eval();

#ifdef VCD_OUTPUT
  m_trace->dump(2 * this->stats_total_cycles + 0);
#endif

  vortex->clk = 1;
  vortex->eval();

  ibus_driver();
  dbus_driver();
  io_handler();

#ifdef VCD_OUTPUT
  m_trace->dump(2 * this->stats_total_cycles + 1);
#endif

  ++time_stamp;
  ++stats_total_cycles;
}

void Vortex_SOC::wait(uint32_t cycles) {
  for (int i = 0; i < cycles; ++i) {
    this->step();
  }
}

bool Vortex_SOC::is_busy() {
  return (0 == vortex->out_ebreak);
}

void Vortex_SOC::send_snoops(uint32_t mem_addr, uint32_t size) {
  // align address to LLC block boundaries
  auto aligned_addr_start = GLOBAL_BLOCK_SIZE_BYTES * (mem_addr / GLOBAL_BLOCK_SIZE_BYTES);
  auto aligned_addr_end = GLOBAL_BLOCK_SIZE_BYTES * ((mem_addr + size + GLOBAL_BLOCK_SIZE_BYTES - 1) / GLOBAL_BLOCK_SIZE_BYTES);

  // submit snoop requests for the needed blocks
  vortex->llc_snp_req_addr = aligned_addr_start;
  vortex->llc_snp_req = false;
  for (;;) {
    this->step();
    if (vortex->llc_snp_req) {
      vortex->llc_snp_req = false;
      if (vortex->llc_snp_req_addr >= aligned_addr_end)
        break;
      vortex->llc_snp_req_addr += GLOBAL_BLOCK_SIZE_BYTES;
    }    
    if (!vortex->llc_snp_req_delay) {
      vortex->llc_snp_req = true;      
    }
  }
}

void Vortex_SOC::flush_caches(uint32_t mem_addr, uint32_t size) {
  // send snoops for L1 flush
  this->send_snoops(mem_addr, size);

#if NUMBER_CORES != 1
  // send snoops for L2 flush
  this->send_snoops(mem_addr, size);
#endif

  // wait 50 cycles to ensure that the request has committed
  this->wait(50);
}

bool Vortex_SOC::simulate() {
  // reset the device
  this->reset();

  // execute program
  while (!vortex->out_ebreak) {
    this->step();
  }

  // wait 5 cycles to flush the pipeline
  this->wait(5);

  std::cerr << "New Total Cycles: " << (this->stats_total_cycles) << "\n";

  this->print_stats();

  return false;
}