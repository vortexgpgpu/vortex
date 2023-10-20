// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
