#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "VVX_mem_streamer.h"
#include "VVX_mem_streamer__Syms.h"

#define SIM_TIME 50
#define CYCLE_DELAY 4


typedef struct {
    bool        valid;
    bool        rw;
    uint8_t     byteen;
    uint32_t    addr;
    uint32_t    data;
    char        tag;
    double      tick;

    uint8_t     ready;
} mem_req_t;

typedef struct {
    bool        valid;
    uint8_t     mask;
    uint32_t    data;
    char        tag;

    bool        ready;
} mem_rsp_t;

int generate_rand (int min, int max);

class MemSim {
    private:
        VVX_mem_streamer *msu_;
        VerilatedVcdC *trace_;

        mem_req_t *mem_req_;
        mem_rsp_t *mem_rsp_;
        
        std::vector<mem_req_t*> ram_;

        void eval();

    public:
        MemSim();
        virtual ~MemSim();

        void step();
        void reset();

        void attach_core();
        void attach_ram();

        void run();
        
};
