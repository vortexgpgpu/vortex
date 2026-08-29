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

#include "ram.h"
#include "memsim.h"

RAM::RAM() {

    ram_.clear();
    is_rsp_active_ = false;
    is_rsp_stall_ = false;
}

bool RAM::check_duplicate_req(req_t req) {
    for(int i = 0; i < ram_.size(); i++) {
        if (ram_[i].addr == req.addr) {
            std::cout<<"RAM: Duplicate entry. Do not insert..."<<std::endl;
            return true;
        }
    }
    return false;
}

int RAM::simulate_cycle_delay() {

    std::cout<<"RAM: # entries: "<<ram_.size()<<std::endl;

    int dequeue_index = -1;

    for (int i = 0; i < ram_.size(); i++) {
        if (!is_rsp_stall_) {
            if (ram_[i].cycles_left > 0) {
                ram_[i].cycles_left -= 1;
            }
        }

        std::cout<<"RAM: # cycles left: "<<ram_[i].cycles_left<<std::endl;

        if (ram_[i].cycles_left == 0) {
            dequeue_index = i;
        }
    }
    return dequeue_index;
}

void RAM::insert_req(req_t req) {
    if ( !(this->check_duplicate_req(req)) && req.valid && !req.rw) {
        req_t r;
        r.valid     = req.valid;
        r.rw        = req.rw;
        r.byteen    = req.byteen;
        r.addr      = req.addr;
        r.data      = req.data;
        r.tag       = req.tag & 0b11;

        // Store metadata
        r.cycles_left = MEM_LATENCY;

        std::cout<<"RAM: Insert entry... "<<std::endl;
        std::cout<<"Write? : "<<req.rw<<std::endl;
        ram_.push_back(r);
    }
}

uint8_t RAM::is_ready() {    
    // return generate_rand(0b1000, 0b1111);
    return 0b1111;
}

rsp_t RAM::schedule_rsp() {
    rsp_t rsp;
    int dequeue_index = this->simulate_cycle_delay();

    if (!is_rsp_active_) {
        if (dequeue_index != -1) {

            std::cout<<"RAM: Scheduling response... "<<std::endl;

            is_rsp_active_ = true;
            rsp.valid   = 1;
            rsp.mask    = generate_rand_mask(ram_[dequeue_index].valid);
            rsp.data    = generate_rand(0x20000000, 0x30000000);
            rsp.tag     = ram_[dequeue_index].tag;

            std::cout<<std::hex;
            std::cout<<"RAM: Response mask: "<<+rsp.mask<<" | Required mask: "<<+ram_[dequeue_index].valid<<std::endl;

            ram_[dequeue_index].rsp_sent_mask = rsp.mask;
            ram_[dequeue_index].valid = ram_[dequeue_index].valid & ~ram_[dequeue_index].rsp_sent_mask;

            if (0 == ram_[dequeue_index].valid) {
                ram_.erase(ram_.begin() + dequeue_index);
                is_rsp_stall_ = false;
                std::cout<<"RAM: Clear entry... "<<std::endl;
            } else {
                is_rsp_stall_ = true;
                std::cout<<"RAM: Stall... "<<std::endl;
            }
        } else {
            rsp.valid = false;
        }
    } 
    
    return rsp;
}

// Schedule response for only one cycle
void RAM::halt_rsp(rsp_t rsp) {
    if (is_rsp_active_ && rsp.valid && rsp.ready) {
        std::cout<<"RAM: Halt response..."<<std::endl;
        is_rsp_active_ = false;
    }
}
