#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "VVX_mem_streamer.h"
#include "VVX_mem_streamer__Syms.h"
#include "ram.h"

#define SIM_TIME 300

typedef struct {
    uint8_t     valid;
    uint8_t     rsp_sent_mask;
    bool        rw;
    uint8_t     byteen;
    uint32_t    addr;
    uint32_t    data;
    char        tag;
    double      cycles_left;
    bool     ready;
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
        std::vector<mem_req_t> ram_;
        bool mem_rsp_active_;
        bool mem_rsp_stall_;

        void eval();
        void step();
        void reset();
        void attach_core();
        void attach_ram();
        void attach_ram2(RAM *ram);

    public:
        MemSim();
        virtual ~MemSim();

        void run(RAM *ram);  
};
