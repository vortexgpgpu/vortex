#pragma once

#include <iostream>
#include <map>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "VVX_mem_streamer.h"
#include "VVX_mem_streamer__Syms.h"


#define NUM_REQS 4
#define ADDRW 32
#define DATAW 32
#define TAGW 32
#define WORD_SIZE 4
#define QUEUE_SIZE 16
#define QUEUE_ADDRW 4
#define PARTIAL_RESPONSE

// Input request
struct Input {
    uint8_t valid;
    uint8_t rw;
    uint8_t mask;
    uint8_t byteen;
    uint32_t* addr;
    uint32_t* data;
    uint32_t tag;
    uint8_t ready;
};

// Output response
struct Output{
    uint8_t valid;
    uint8_t mask;
    uint32_t* data;
    uint32_t tag;
    uint8_t ready;
};

class MemSim {
    public:
        MemSim();
        virtual ~MemSim();

        void send_req(req_in_t *req);
        void get_rsp(rsp_out_t *rsp);

    private:
        VVX_mem_streamer *msu_;
        VerilatedVcdC *trace_;
};
