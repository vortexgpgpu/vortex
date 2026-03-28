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

// TCU tile buffer (VX_tcu_tbuf) unit test — smoke / connectivity check.
// Drives reset, asserts idle inputs, and runs a handful of clock cycles
// to verify that the module elaborates and resets cleanly.
// Functional fetch / gather sequence tests are added here incrementally.

#include <iostream>
#include <cassert>
#include "VVX_tcu_tbuf_top.h"
#include "verilated.h"
#ifdef VCD_OUTPUT
#include "verilated_vcd_c.h"
#endif

#define MAX_SIM_CYCLES 1000

static VVX_tcu_tbuf_top* dut;
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

    dut = new VVX_tcu_tbuf_top;
    dut->clk   = 0;
    dut->reset = 0;

#ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    vcd = new VerilatedVcdC;
    dut->trace(vcd, 99);
    vcd->open("trace.vcd");
#endif

    // ---- idle inputs -------------------------------------------------------
    dut->req_valid      = 0;
    dut->req_fire       = 0;
    dut->req_wid        = 0;
    dut->req_is_sparse  = 0;
    dut->req_step_m     = 0;
    dut->req_step_n     = 0;
    dut->req_step_k     = 0;
    dut->req_fmt_s      = 0;
    dut->req_desc_a     = 0;
    dut->req_desc_b     = 0;
    dut->lmem_req_ready = 1;
    dut->lmem_rsp_valid = 0;

    reset_dut();

    // ---- idle run ----------------------------------------------------------
    for (int i = 0; i < MAX_SIM_CYCLES; i++)
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
