#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "memsim.h"

uint64_t timestamp = 0;

double sc_time_stamp(){
    return timestamp;
}

MemSim::MemSim(){

    msu_ = new VVX_mem_streamer();

    Verilated::traceEverOn(true);
    trace_ = new VerilatedVcdC;
    msu_->trace(trace_, 99);
    trace_->open("trace.vcd");

}


int main(int argc, char **argv){

    return 0;
}

