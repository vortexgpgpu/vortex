// TEX unit smoke test — elaboration + reset.

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include "VVX_tex_core_top.h"
#include "verilated.h"
#ifdef VCD_OUTPUT
#include "verilated_vcd_c.h"
#endif

#define MAX_SIM_CYCLES 100

static VVX_tex_core_top* dut;
static uint64_t sim_time = 0;

#ifdef VCD_OUTPUT
static VerilatedVcdC* vcd;
#endif

static void tick() {
    dut->clk ^= 1;
    dut->eval();
#ifdef VCD_OUTPUT
    vcd->dump(sim_time);
#endif
    ++sim_time;
}

static void clock_cycle() {
    tick();
    tick();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new VVX_tex_core_top;
    dut->clk = 0;
    dut->reset = 1;

#ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    vcd = new VerilatedVcdC;
    dut->trace(vcd, 99);
    vcd->open("trace.vcd");
#endif

    // tie off
    dut->dcr_write_valid = 0;
    dut->tex_req_valid = 0;
    dut->tex_rsp_ready = 1;
    dut->cache_req_ready = ~0ULL;
    dut->cache_rsp_valid = 0;

    for (int i = 0; i < 4; ++i) clock_cycle();
    dut->reset = 0;

    for (int i = 0; i < MAX_SIM_CYCLES; ++i)
        clock_cycle();

    std::cout << "PASSED" << std::endl;
#ifdef VCD_OUTPUT
    vcd->close();
    delete vcd;
#endif
    dut->final();
    delete dut;
    return 0;
}
