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

#include <simobject.h>
#include "mem_sim.h"
#include "debug.h"
#include <queue>

namespace vortex {

// Simplified Page Table Walker (PTW) for VM performance modeling
// Purpose: Model the latency of multi-level page table walks
class PTW : public SimObject<PTW> {
public:
    struct Config {
        uint8_t  pt_levels;      // Page table levels (2 for SV32, 3 for SV39)
        uint8_t  pte_size;       // PTE size in bytes (4 for SV32, 8 for SV39)
    };

    struct PerfStats {
        uint64_t walks;          // Total page table walks
        uint64_t mem_accesses;   // Total memory accesses (PT_LEVEL per walk)
        
        PerfStats() 
            : walks(0)
            , mem_accesses(0)
        {}
        
        PerfStats& operator+=(const PerfStats& rhs) {
            this->walks += rhs.walks;
            this->mem_accesses += rhs.mem_accesses;
            return *this;
        }
    };

    // Ports
    SimPort<MemReq> CoreReqPort;   // Request from TLB (on TLB miss)
    SimPort<MemRsp> CoreRspPort;   // Response to TLB (walk complete)
    SimPort<MemReq> MemReqPort;    // Memory requests for PTE reads
    SimPort<MemRsp> MemRspPort;    // Memory responses with PTE data

    PTW(const SimContext& ctx, const char* name, const Config& config)
        : SimObject(ctx, name)
        , CoreReqPort(this)
        , CoreRspPort(this)
        , MemReqPort(this)
        , MemRspPort(this)
        , config_(config)
        , state_(State::IDLE)
        , current_level_(0)
    {
        DT(1, "PTW created: " << name << " levels=" << (int)config.pt_levels);
    }

    ~PTW() {}

    void reset() {
        state_ = State::IDLE;
        current_level_ = 0;
        while (!pending_req_queue_.empty()) {
            pending_req_queue_.pop();
        }
        perf_stats_ = PerfStats();
    }

    void tick() {
        switch (state_) {
            case State::IDLE:
                // Accept new PTW request if available
                if (!CoreReqPort.empty()) {
                    current_req_ = CoreReqPort.front();
                    CoreReqPort.pop();
                    
                    current_level_ = config_.pt_levels - 1;  // Start at top level
                    state_ = State::READING_PTE;
                    
                    DT(2, this->name() << "-start-walk: vaddr=0x" << std::hex 
                       << current_req_.addr << std::dec);
                    
                    ++perf_stats_.walks;
                    
                    // Issue first PTE read
                    issue_pte_read();
                }
                break;
                
            case State::READING_PTE:
                // Wait for PTE response
                if (!MemRspPort.empty()) {
                    MemRspPort.pop();  // Consume response (we don't parse it)
                    
                    DT(3, this->name() << "-pte-read-done: level=" << (int)current_level_);
                    
                    // Check if we need more levels
                    if (current_level_ > 0) {
                        // Move to next level
                        current_level_--;
                        issue_pte_read();
                    } else {
                        // All levels done - send response back to TLB
                        MemRsp tlb_rsp;
                        tlb_rsp.tag = current_req_.tag;
                        tlb_rsp.cid = current_req_.cid;
                        tlb_rsp.uuid = current_req_.uuid;
                        
                        CoreRspPort.push(tlb_rsp, 1);
                        
                        DT(2, this->name() << "-walk-complete: vaddr=0x" << std::hex 
                           << current_req_.addr << std::dec);
                        
                        state_ = State::IDLE;
                    }
                }
                break;
        }
    }

    PerfStats perf_stats() const {
        return perf_stats_;
    }

private:
    enum class State {
        IDLE,           // No walk in progress
        READING_PTE     // Reading PTE, waiting for memory response
    };

    // Extract VPN (Virtual Page Number) for a given level
    uint64_t extract_vpn(uint64_t vaddr, uint8_t level) const {
        #ifdef XLEN_32
        // SV32: 10 bits per level
        // Level 1: bits [31:22], Level 0: bits [21:12]
        uint8_t shift = 12 + (level * 10);
        return (vaddr >> shift) & 0x3FF;  // 10 bits
        #else
        // SV39: 9 bits per level
        // Level 2: bits [38:30], Level 1: bits [29:21], Level 0: bits [20:12]
        uint8_t shift = 12 + (level * 9);
        return (vaddr >> shift) & 0x1FF;  // 9 bits
        #endif
    }

    // Calculate PTE address (simplified - assumes base_ppn from SATP passed in p_addr)
    uint64_t get_pte_address(uint64_t base_ppn, uint64_t vpn) const {
        // PTE_addr = base_ppn * PAGE_SIZE + vpn * PTE_SIZE
        return (base_ppn << 12) + (vpn * config_.pte_size);
    }

    // Issue PTE read for current level
    void issue_pte_read() {
        uint64_t vpn = extract_vpn(current_req_.addr, current_level_);
        uint64_t base_ppn = current_req_.p_addr >> 12;  // Base PPN from SATP
        uint64_t pte_addr = get_pte_address(base_ppn, vpn);
        
        DT(3, this->name() << "-issue-pte-read: level=" << (int)current_level_ 
           << " vpn=0x" << std::hex << vpn 
           << " pte_addr=0x" << pte_addr << std::dec);
        
        // Create memory request for PTE read
        MemReq mem_req;
        mem_req.addr = pte_addr;
        mem_req.p_addr = pte_addr;  // Physical address
        mem_req.write = false;
        mem_req.tag = 0;  // Don't need to track tag internally
        mem_req.cid = current_req_.cid;
        mem_req.uuid = current_req_.uuid;
        
        MemReqPort.push(mem_req, 1);
        
        ++perf_stats_.mem_accesses;
    }

    Config config_;
    State state_;
    uint8_t current_level_;
    MemReq current_req_;
    std::queue<MemReq> pending_req_queue_;  // For backpressure if needed
    PerfStats perf_stats_;
};

} // namespace vortex

