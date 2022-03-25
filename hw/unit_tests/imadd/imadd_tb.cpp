#include <stdlib.h>
#include <iostream>
#include "VVX_imadd.h"
#include "VVX_imadd__Syms.h"
#include <verilated.h>
#include <verilated_vcd_c.h>

vluint64_t sim_time = 0;
vluint64_t posedges = 0;

int main(int argc, char** argv) {
    VVX_imadd *dut = new VVX_imadd;

    Verilated::traceEverOn(true);
    VerilatedVcdC *m_trace = new VerilatedVcdC;
    dut->trace(m_trace, 5);
    m_trace->open("wave.vcd");

    while (sim_time < 50) {
        dut->alu_in1[0] = 0x2;
        dut->alu_in2[0] = 0x3;
        dut->alu_in3[0] = 0x4;

        dut->clk ^= 1;
        dut->eval();
        m_trace->dump(sim_time);
        sim_time++;
        if(dut->clk) posedges++;
    }

    m_trace->close();
    delete dut;
    return 0;
}

