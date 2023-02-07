#include <iostream>
#include <verilated.h>
#include <verilated_fst_c.h>
#include "Vtb_ahb_bus_system.h"


int main() {

    uint64_t sim_time = 0;
    Vtb_ahb_bus_system TB;
    VerilatedFstC trace;
    Verilated::traceEverOn(true);

    TB.trace(&trace, 5);
    trace.open("waveform.fst");

    int clk_value = 0;

    while(!Verilated::gotFinish() && sim_time < 2000) {
        TB.CLK = clk_value;
        clk_value = !clk_value;
        TB.eval();
        trace.dump(sim_time);
        sim_time++;
    }

    if(sim_time == 2000) {
        std::cout << "Warning: Sim stopped due to timeout!\n";
    }


    TB.final();
    trace.close();

    return 0;
}
