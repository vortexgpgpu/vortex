#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "VVX_mem_streamer_test.h"
#include "VVX_mem_streamer_test__Syms.h"
#include "ram.h"

#define SIM_TIME 1000

int generate_rand (int min, int max);

class MemSim {
    private:
        VVX_mem_streamer_test *msu_;
        VerilatedVcdC *trace_;

        void eval();
        void step();
        void reset();

        void attach_core();
        void attach_ram(RAM *ram);

    public:
        MemSim();
        virtual ~MemSim();

        void run(RAM *ram);  
};
