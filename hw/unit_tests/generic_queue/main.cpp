#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include "vl_simulator.h"
#include "VVX_fifo_queue.h"
#include "VVX_fifo_queue__Syms.h"

static const uint32_t MAX_TICKS  = 10000000;
static const uint32_t NUM_ITERATIONS = 20000;

using Device = VVX_fifo_queue;

int main(int argc, char **argv) {
  // Initialize Verilators variables
  Verilated::commandArgs(argc, argv);

  vl_simulator<Device> sim;

  auto start_time = std::chrono::system_clock::now();

  // run simulation
  unit64_t ticks = sim.reset(0);
  
  for (;;) {
    //sim->io_in_valid = (in_sample < num_iterations * FFT_SIZE);
    //sim->io_out_ready = (out_sample < num_iterations * FFT_SIZE);

    // enqueue data
    //if (sim->io_in_valid && sim->io_in_ready) {
      //std::cout << "t" << std::dec << ticks << std::hex << " input: re=" << re << ", im=" << im << std::endl;
      //sim->io_in_data = sample;
    //}

    // dequeue data
    //if (sim->io_out_valid && sim->io_out_ready) {
      //std::cout << "t" << std::dec << ticks << std::hex << " output: re=" << re << ", im=" << im << std::endl;
      //test_outputs[out_sample++] = sample;
    //}

    // check for completion
    

    // advance clock
    ticks = sim.step(ticks, 2);
  }

  auto end_time = std::chrono::system_clock::now();
  auto latency = end_time - start_time;
  std::cout << "Average elapsed time = "
            << std::chrono::duration<double, std::milli>(latency).count()
            << " ms" << std::endl;

  std::cout << "Simulation run time: " << std::dec << ticks/2 << " cycles" << std::endl;

  return 0;
}