#pragma once

#include <iostream>
#include "VVX_mem_streamer.h"
#include "VVX_mem_streamer__Syms.h"
#include "verilated.h"
#include <verilated_vcd_c.h>


#define NUM_REQS 4
#define ADDRW 32
#define DATAW 32
#define TAGW 32
#define WORD_SIZE 4
#define QUEUE_SIZE 16
#define QUEUE_ADDRW 4
#define PARTIAL_RESPONSE

// Input request
typedef struct {
    uint8_t valid;
    uint8_t rw;
    uint8_t mask;
    uint8_t byteen;
    uint32_t* addr;
    uint32_t* data;
    uint32_t tag;
    uint8_t ready;
} req_in;

// Output response
typedef struct {
    uint8_t valid;
    uint8_t mask;
    uint32_t* data;
    uint32_t tag;
    uint8_t ready;
} rsp_out;

class MemSim {
    public:
        MemSim();
        virtual ~MemSim();

        void send_req(req_in *req);
        void get_rsp(rsp_out *rsp);

    private:
        VVX_mem_streamer *msu_;
        VerilatedVcdC *trace_;
};
