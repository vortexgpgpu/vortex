#include "vl_simulator.h"
#include "VVX_fifo_queue.h"
#include <iostream>

#define MAX_TICKS 20

#define CHECK(x)                                  \
   do {                                           \
     if (x)                                       \
       break;                                     \
     std::cout << "FAILED: " << #x << std::endl;  \
	   std::abort();			                          \
   } while (false)

uint64_t ticks = 0;

double sc_time_stamp() { 
  return ticks;
}

using Device = VVX_fifo_queue;

int main(int argc, char **argv) {
  // Initialize Verilators variables
  Verilated::commandArgs(argc, argv);

  vl_simulator<Device> sim;

  // run test
  ticks = sim.reset(0);
  while (ticks < MAX_TICKS) {
    switch (ticks) {
    case 0:
      // initial values
      sim->pop  = 0;
      sim->push = 0;
      ticks = sim.step(ticks, 2);
      break;
    case 2:
      // Verify outputs    
      CHECK(sim->full == 0x0);
      CHECK(sim->empty == 0x1);
      // push 0xa
      sim->pop     = 0;
      sim->push    = 1;
      sim->data_in = 0xa;
      break;
    case 4:
      // verify outputs    
      CHECK(sim->data_out == 0xa);
      CHECK(sim->full == 0x0);
      CHECK(sim->empty == 0x0);
      // push 0xb
      sim->pop     = 0;
      sim->push    = 1;
      sim->data_in = 0xb;
      break;
    case 6:
      // verify outputs    
      CHECK(sim->data_out == 0xa);
      CHECK(sim->full == 0x1);
      CHECK(sim->empty == 0x0);
      // pop
      sim->pop  = 1;
      sim->push = 0;
      break;
    case 8:
      // verify outputs    
      CHECK(sim->data_out == 0xb);
      CHECK(sim->full == 0x0);
      CHECK(sim->empty == 0x0);
      // pop
      sim->pop  = 1;
      sim->push = 0;
      break;
    case 10:
      // verify outputs    
      CHECK(sim->full == 0x0);
      CHECK(sim->empty == 0x1);
      sim->pop  = 0;
      sim->push = 0;
      break;
    }

    // advance clock
    ticks = sim.step(ticks, 2);
  }

  std::cout << "PASSED!" << std::endl;
  std::cout << "Simulation time: " << std::dec << ticks/2 << " cycles" << std::endl;

  return 0;
}