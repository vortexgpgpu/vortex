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
  Verilated::randReset(1);

  vortex_ = new VVX_priority_encoder();

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



void Simulator::step() {
  this->eval();

  this->eval();
}


void Simulator::reset() {     
#ifndef NDEBUG
  std::cout << timestamp << ": [sim] reset()" << std::endl;
#endif

  this->step();  
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



bool Simulator::run() {
  // reset the device
  this->reset();

  // execute program
  this->step();
  

  // wait 5 cycles to flush the pipeline
  this->wait(5);



  return 0;
}


