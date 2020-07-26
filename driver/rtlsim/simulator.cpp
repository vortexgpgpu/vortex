#include "simulator.h"
#include <iostream>
#include <fstream>
#include <iomanip>

uint64_t timestamp = 0;

double sc_time_stamp() { 
  return timestamp;
}

Simulator::Simulator() {    
  // force random values for unitialized signals
  const char* args[] = {"", "+verilator+rand+reset+2", "+verilator+seed+50"};
  Verilated::commandArgs(3, args);

  vortex_ = new Vvortex_afu_sim();

#ifdef VCD_OUTPUT
  Verilated::traceEverOn(true);
  trace_ = new VerilatedVcdC;
  trace_->set_time_unit("1ns");
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

  avs_driver();
  ccip_driver();
}

void Simulator::eval() {
  vortex_->eval();
#ifdef VCD_OUTPUT
  trace_->dump(timestamp);
#endif
  ++timestamp;
}

void Simulator::avs_driver() {
  //--
}

 void Simulator::ccip_driver() {
   //--
 }