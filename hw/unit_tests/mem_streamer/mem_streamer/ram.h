#pragma once

#include <iostream>
#include <vector>

#define MEM_LATENCY 4

typedef struct {
    uint8_t     valid;
    bool        rw;
    uint8_t     byteen;
    uint32_t    addr;
    uint32_t    data;
    uint8_t     tag;
    uint8_t     ready;

    // Metadata
    uint8_t     rsp_sent_mask;
    double      cycles_left;
} req_t;

typedef struct {
    bool        valid;
    uint8_t     mask;
    uint32_t    data;
    uint8_t     tag;
    bool        ready;
} rsp_t;

class RAM {

    private:
        std::vector<req_t> ram_;

        bool is_rsp_active_;
        bool is_rsp_stall_;

        bool    check_duplicate_req(req_t req);
        int     simulate_cycle_delay();
    
    public:
        RAM();
        
        uint8_t is_ready();
        void    insert_req(req_t req);
        rsp_t   schedule_rsp();
        void    halt_rsp(rsp_t rsp);

};

//////////////////////////////////////////////////////
