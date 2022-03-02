#include <iostream>
#include <map>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "VVX_mem_streamer_test.h"
#include "VVX_mem_streamer_test__Syms.h"

// Number of clock edges that we'll simulate
#define MAX_SIM_TIME 300
vluint64_t sim_time = 0;
vluint64_t posedges = 0;

static uint64_t timestamp = 0;
static bool trace_enabled = false;
static uint64_t trace_start_time = 0;
static uint64_t trace_stop_time = -1ull;

double sc_time_stamp() { 
  return timestamp;
}

bool sim_trace_enabled() {
  if (timestamp >= trace_start_time 
   && timestamp < trace_stop_time)
    return true;
  return trace_enabled;
}

void sim_trace_enable (bool enable) {
  trace_enabled = enable;
}

int main (int argc, char** argv, char** env) {
    Verilated::commandArgs(argc, argv);
    VVX_mem_streamer_test *dut = new VVX_mem_streamer_test();

    // Generate trace
    Verilated::traceEverOn(true);
    VerilatedVcdC *tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("trace.vcd");

    // Simulate
    while (sim_time < MAX_SIM_TIME) {
        dut->reset = 0;
        if (sim_time >=2 && sim_time <=4) {
            dut->reset = 1;
            dut->req_valid = 0;
            dut->req_tag = 0x0;
        }
        dut->clk ^= 1;
        
        // Positive clock edge
        if (1 == dut->clk) {
            posedges++;
            if (5 == posedges) {
                dut->req_valid = 1;
                dut->req_tag = 0x12345678;
            } else {
                dut->req_valid = 0;
                dut->req_tag = 0;
            }
        }

        // Evaluate module
        dut->eval();
        tfp->dump(sim_time);
        sim_time++;
    }

    tfp->close();
    delete dut;
    exit(EXIT_SUCCESS);
}
