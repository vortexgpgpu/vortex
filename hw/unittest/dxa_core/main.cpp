// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// DXA core unit test — smoke / connectivity check.
// Drives reset and a handful of clock cycles to verify that VX_dxa_core
// elaborates and resets cleanly.  Functional DXA transaction tests are
// added here incrementally.

#include <iostream>
#include <cassert>
#include "VVX_dxa_core_top.h"
#include "verilated.h"
#ifdef VCD_OUTPUT
#include "verilated_vcd_c.h"
#endif
#ifdef SAIF_OUTPUT
#include "verilated_saif_c.h"
#endif

#define MAX_SIM_CYCLES 1000

static VVX_dxa_core_top* dut;
static uint64_t sim_time = 0;

#ifdef VCD_OUTPUT
static VerilatedVcdC* vcd;
#endif
#ifdef SAIF_OUTPUT
static VerilatedSaifC* saif;
#endif

static void tick() {
    dut->clk ^= 1;
    dut->eval();
#ifdef VCD_OUTPUT
    vcd->dump(sim_time);
#endif
#ifdef SAIF_OUTPUT
    saif->dump(sim_time);
#endif
    ++sim_time;
}

static void clock_cycle() {
    tick(); // posedge
    tick(); // negedge
}

static void reset_dut(int cycles = 4) {
    dut->reset = 1;
    for (int i = 0; i < cycles; i++)
        clock_cycle();
    dut->reset = 0;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    dut = new VVX_dxa_core_top;
    dut->clk   = 0;
    dut->reset = 0;

#ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    vcd = new VerilatedVcdC;
    dut->trace(vcd, 99);
    vcd->open("trace.vcd");
#endif
#ifdef SAIF_OUTPUT
    Verilated::traceEverOn(true);
    saif = new VerilatedSaifC;
    dut->trace(saif, 99);
    saif->open("trace.saif");
#endif

    // ---- tie off unused inputs -------------------------------------------
    dut->dcr_req_valid  = 0;
    dut->dcr_req_rw     = 0;
    dut->dxa_req_valid  = 0;
    dut->gmem_req_ready = ~0ULL; // all ready
    dut->gmem_rsp_valid = 0;
    dut->smem_wr_ready  = ~0ULL; // all ready

    reset_dut();

    // ---- run a few idle cycles -------------------------------------------
    for (int i = 0; i < MAX_SIM_CYCLES; i++)
        clock_cycle();

    std::cout << "PASSED" << std::endl;

#ifdef VCD_OUTPUT
    vcd->close();
    delete vcd;
#endif
#ifdef SAIF_OUTPUT
    saif->close();
    delete saif;
#endif
    dut->final();
    delete dut;
    return 0;
}
