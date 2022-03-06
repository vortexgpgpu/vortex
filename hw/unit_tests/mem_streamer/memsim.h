#pragma once

#include <iostream>
#include <map>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "VVX_mem_streamer_test.h"
#include "VVX_mem_streamer_test__Syms.h"

// Input request
typedef struct {
    uint8_t valid;
    uint8_t rw;
    uint8_t mask;
    uint8_t byteen;
    uint32_t addr;
    uint32_t data;
    uint32_t tag;
    uint8_t ready;
} req_t;

// Output response
typedef struct {
    uint8_t valid;
    uint8_t mask;
    uint32_t* data;
    uint32_t tag;
    uint8_t ready;
} rsp_t;

class MemSim {
    private:
        VVX_mem_streamer_test *msu_;
        VerilatedVcdC *trace_;

        void eval();

    public:
        MemSim();
        virtual ~MemSim();

        void step();
        void reset();
        void set_core_req(req_t *req);
};
