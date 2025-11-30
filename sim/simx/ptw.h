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
#include "types.h"
#include <queue>
#include <map>
#include <iostream>
#include <iomanip>
#include <climits>

namespace vortex {

// Simplified Page Table Walker (PTW) for VM performance modeling
// Purpose: Model the latency of multi-level page table walks
// Supports parallel walks: multiple walks can be in progress simultaneously
// Uses buffer_size to limit maximum concurrent walks
class PTW : public SimObject<PTW> {
public:
    struct Config {
        uint8_t  pt_levels;      // Page table levels (2 for SV32, 3 for SV39)
        uint8_t  pte_size;       // PTE size in bytes (4 for SV32, 8 for SV39)
        uint64_t base_ppn;       // Base PPN from SATP register
        uint32_t buffer_size;    // Size of request buffer
    };

    struct PerfStats {
        uint64_t walks;          // Total page table walks
        uint64_t mem_accesses;   // Total memory accesses (PT_LEVEL per walk)
        uint64_t max_concurrent_walks;  // Maximum concurrent walks observed

        PerfStats()
            : walks(0)
            , mem_accesses(0)
            , max_concurrent_walks(0)
        {}

        PerfStats& operator+=(const PerfStats& rhs) {
            this->walks += rhs.walks;
            this->mem_accesses += rhs.mem_accesses;
            this->max_concurrent_walks = std::max(this->max_concurrent_walks, rhs.max_concurrent_walks);
            return *this;
        }
    };

    // Ports
    SimPort<MemReq> CoreReqPort;   // Request from TLB (on TLB miss) - via arbiter
    SimPort<MemRsp> CoreRspPort;   // Response to TLB (walk complete) - via arbiter
    SimPort<MemReq> MemReqPort;    // Memory requests for PTE reads
    SimPort<MemRsp> MemRspPort;    // Memory responses with PTE data

    PTW(const SimContext& ctx, const char* name, const Config& config)
        : SimObject(ctx, name)
        , CoreReqPort(this)
        , CoreRspPort(this)
        , MemReqPort(this)
        , MemRspPort(this)
        , next_mem_tag_(1)  // Start at 1 so 0 can mean "no pending request"
        , config_(config)
        , next_walk_id_(0)
    {
        std::cout << "[PTW] CONSTRUCTOR: " << name << " levels=" << (int)config.pt_levels 
           << " buffer_size=" << config.buffer_size << " base_ppn=0x" << std::hex << config.base_ppn << std::dec << std::endl;
        std::cout.flush();
        DT(1, "PTW created: " << name << " levels=" << (int)config.pt_levels 
           << " buffer_size=" << config.buffer_size);
    }

    ~PTW() {}

    void reset() {
        active_walks_.clear();
        while (!request_queue_.empty()) {
            request_queue_.pop();
        }
        perf_stats_ = PerfStats();
        next_walk_id_ = 0;
        next_mem_tag_ = 1;  // Start at 1 so 0 can mean "no pending request"
    }

    void tick() {
        static uint64_t tick_count = 0;
        static int req_check_trace = 0;
        
        // Trace PTW status periodically
        if (req_check_trace < 10 && (tick_count % 100) == 0) {
            std::cout << "\n[PTW DEBUG] " << this->name() << " tick=" << tick_count 
               << " active_walks=" << active_walks_.size()
               << " queue_size=" << request_queue_.size() 
               << "/" << (int)config_.buffer_size << std::endl;
            req_check_trace++;
        }
        tick_count++;
        
        // Accept new PTW requests into buffer (from arbiter - handles multiple TLB sources)
        while (!CoreReqPort.empty() && (request_queue_.size() + active_walks_.size()) < config_.buffer_size) {
            MemReq req = CoreReqPort.front();
            request_queue_.push(req);
            CoreReqPort.pop();
            std::cout << "[PTW] " << this->name() << " RECEIVED REQUEST: vaddr=0x" << std::hex << req.addr 
               << std::dec << " tag=" << req.tag << " cid=" << req.cid << " queue_size=" << request_queue_.size() << std::endl;
        }

        // Update max concurrency
        uint64_t concurrent = request_queue_.size() + active_walks_.size();
        perf_stats_.max_concurrent_walks = std::max(perf_stats_.max_concurrent_walks, concurrent);

        // Step 1: Process memory responses (match to correct walk)
        while (!MemRspPort.empty()) {
            const MemRsp& pte_rsp = MemRspPort.front();
            MemRspPort.pop();
            
            // Find the walk that matches this memory response tag
            uint32_t matched_walk_id = UINT32_MAX;
            WalkState* walk = nullptr;
            for (auto& [walk_id, walk_state] : active_walks_) {
                if (walk_state.mem_req_tag == pte_rsp.tag) {
                    matched_walk_id = walk_id;
                    walk = &walk_state;
                    break;
                }
            }
            
            if (!walk) {
                std::cerr << "[PTW] ERROR: Memory response tag=" << pte_rsp.tag 
                   << " not found in active walks!" << std::endl;
                continue;
            }
            
            // Clear mem_req_tag to indicate we can issue next level request
            walk->mem_req_tag = 0;
            
            std::cout << "[PTW] " << this->name() << " RECEIVED MEMORY RESPONSE: tag=" << pte_rsp.tag 
               << " level=" << (int)walk->level 
               << " vaddr=0x" << std::hex << walk->req.addr << std::dec << std::endl;

            // Parse PTE to extract PPN for next level
            uint64_t pte_ppn = parse_pte_ppn(*walk);
            std::cout << "[PTW] " << this->name() << " PARSED PTE: level=" << (int)walk->level 
               << " pte_ppn=0x" << std::hex << pte_ppn << std::dec << std::endl;

            // Check if we need more levels
            if (walk->level > 0) {
                // Move to next level with PPN from current level's PTE
                walk->level = walk->level - 1;
                walk->level_ppn = pte_ppn;
                
                std::cout << "[PTW] " << this->name() << " TRANSITIONING TO NEXT LEVEL: to level=" 
                   << (int)walk->level 
                   << " using PPN=0x" << std::hex << pte_ppn << std::dec << std::endl;
                
                // Issue next level PTE read (will happen in Step 3)
            } else {
                // All levels done - send response back to TLB
                std::cout << "[PTW] " << this->name() << " ALL LEVELS COMPLETE: final_level=0" 
                   << " final_ppn=0x" << std::hex << pte_ppn << std::dec 
                   << " vaddr=0x" << std::hex << walk->req.addr << std::dec << std::endl;
                
                MemRsp tlb_rsp;
                tlb_rsp.tag = walk->req.tag;
                tlb_rsp.cid = walk->req.cid;
                tlb_rsp.uuid = walk->req.uuid;

                std::cout << "[WALK_COMPLETE] walk_tag=" << walk->req.tag
                         << " vaddr=0x" << std::hex << walk->req.addr << std::dec
                         << " final_ppn=0x" << std::hex << pte_ppn << std::dec
                         << " mem_issued=" << perf_stats_.mem_accesses
                         << " walks_completed=" << perf_stats_.walks << std::endl;
                CoreRspPort.push(tlb_rsp, 1);

                DT(1, "=== PTW COMPLETE: vaddr=0x" << std::hex << walk->req.addr << std::dec 
                   << " final_ppn=0x" << pte_ppn << " ===");

                // Remove completed walk
                active_walks_.erase(matched_walk_id);
            }
        }

        // Step 2: Start new walks from queue (up to buffer_size concurrent)
        while (!request_queue_.empty() && active_walks_.size() < config_.buffer_size) {
            MemReq req = request_queue_.front();
            request_queue_.pop();

            ++perf_stats_.walks;

            // Create walk state and add to active walks
            uint32_t walk_id = req.tag;  // Use request tag as walk ID
            
            std::cout << "[WALK_START] walk_id=" << walk_id 
                     << " vaddr=0x" << std::hex << req.addr << std::dec
                     << " walk_count=" << perf_stats_.walks
                     << " mem_count=" << perf_stats_.mem_accesses << std::endl;

            // If base_ppn is not set (0), use identity mapping (skip actual page table walk)
            if (config_.base_ppn == 0) {
                // Identity mapping: return immediately without memory accesses
                MemRsp tlb_rsp;
                tlb_rsp.tag = req.tag;
                tlb_rsp.cid = req.cid;
                tlb_rsp.uuid = req.uuid;

                static std::map<std::string, int> sent_count_identity;
                sent_count_identity[this->name()]++;
                std::cout << "[PTW] " << this->name() << " SENDING IDENTITY RESPONSE #" << sent_count_identity[this->name()] 
                   << ": tag=" << tlb_rsp.tag 
                   << " cid=" << tlb_rsp.cid << " vaddr=0x" << std::hex << req.addr << std::dec << std::endl;
                CoreRspPort.push(tlb_rsp, 1);

                DT(1, "=== PTW IDENTITY: vaddr=0x" << std::hex << req.addr 
                   << std::dec << " (base_ppn=0, skipping walk)");
                continue;  // Skip adding to active_walks_
            }
            uint8_t start_level = config_.pt_levels - 1;
            active_walks_[walk_id] = WalkState(req, start_level, config_.base_ppn);

            std::cout << "[PTW] " << this->name() << " STARTING WALK: vaddr=0x" << std::hex << req.addr 
               << std::dec << " tag=" << req.tag << " base_ppn=0x" << std::hex << config_.base_ppn << std::dec 
               << " total_levels=" << (int)config_.pt_levels 
               << " starting_level=" << (int)start_level << std::endl;

            DT(1, "=== PTW START: vaddr=0x" << std::hex << req.addr << std::dec
               << " tag=" << req.tag << " base_ppn=0x" << config_.base_ppn);
            
            // Issue first level PTE read immediately
            issue_pte_read(active_walks_[walk_id]);
        }

        // Step 3: Issue memory requests for walks waiting for next level PTE read
        // (Walks that just completed a level - mem_req_tag cleared in Step 1)
        // Note: New walks already issue first request in Step 2, so skip them here
        for (auto& [walk_id, walk] : active_walks_) {
            // Issue PTE read if walk is ready (mem_req_tag == 0 means level completed, ready for next)
            if (walk.mem_req_tag == 0) {
                issue_pte_read(walk);
            }
        }
    }

    PerfStats perf_stats() const {
        return perf_stats_;
    }

    void set_base_ppn(uint64_t base_ppn) {
        config_.base_ppn = base_ppn;
        std::cout << "[PTW] " << this->name() << " set_base_ppn CALLED: base_ppn=0x" << std::hex << base_ppn << std::dec << std::endl;
        DT(1, "PTW::set_base_ppn: base_ppn=0x" << std::hex << base_ppn << std::dec);
    }

private:
    // Per-walk state tracking for parallel walks
    struct WalkState {
        MemReq req;              // Original request
        uint8_t level;           // Current page table level
        uint64_t level_ppn;      // Current level's PPN
        uint32_t mem_req_tag;    // Tag used for memory request (to match response)
        bool active;             // Is this walk currently active?
        
        WalkState() : level(0), level_ppn(0), mem_req_tag(0), active(false) {}
        
        WalkState(const MemReq& r, uint8_t l, uint64_t ppn)
            : req(r), level(l), level_ppn(ppn), mem_req_tag(0), active(true) {}
    };
    
    // Generate unique tag for memory requests (to match responses)
    uint32_t next_mem_tag_;

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

    // Calculate PTE address
    uint64_t get_pte_address(uint64_t base_ppn, uint64_t vpn) const {
        // PTE_addr = base_ppn * PT_SIZE + vpn * PTE_SIZE
        // PT_SIZE = MEM_PAGE_SIZE (4096) = 4KB page size
        return (base_ppn << 12) + (vpn * config_.pte_size);
    }

    // Parse PTE to extract PPN (simplified for performance modeling)
    uint64_t parse_pte_ppn(const WalkState& walk) const {
        // For performance modeling, we simulate PPN extraction
        // In a real implementation, this would parse the actual PTE format
        // For now, we simulate realistic PPN values based on the address pattern

        uint64_t simulated_ppn = 0;

        #ifdef XLEN_32
        // SV32: PPN is bits [31:10] of PTE
        // For simulation, we use a pattern based on the virtual address
        uint64_t vpn = extract_vpn(walk.req.addr, walk.level);
        simulated_ppn = (vpn & 0x3FF) | ((walk.level_ppn & 0xFFFFF000) << 10);
        #else
        // SV39: PPN is bits [53:10] of PTE
        // For simulation, we use a pattern based on the virtual address
        uint64_t vpn = extract_vpn(walk.req.addr, walk.level);
        simulated_ppn = (vpn & 0x1FF) | ((walk.level_ppn & 0xFFFFF000) << 9);
        #endif

        return simulated_ppn;
    }

    // Issue PTE read for current level
    void issue_pte_read(WalkState& walk) {
        uint64_t vpn = extract_vpn(walk.req.addr, walk.level);
        uint64_t pte_addr = get_pte_address(walk.level_ppn, vpn);

        std::cout << "[PTW] " << this->name() << " ISSUING MEMORY REQUEST: level=" << (int)walk.level 
           << "/" << (int)(config_.pt_levels - 1) << " vpn=0x" << std::hex << vpn 
           << " pte_addr=0x" << pte_addr << std::dec 
           << " base_ppn=0x" << std::hex << walk.level_ppn << std::dec
           << " cid=" << walk.req.cid 
           << " vaddr=0x" << std::hex << walk.req.addr << std::dec << std::endl;

        DT(1, "  PTW Level " << (int)walk.level << " PTE read start: vpn=0x" << std::hex << vpn
           << " pte_addr=0x" << pte_addr << std::dec);

        // Generate unique tag for this memory request to match response
        walk.mem_req_tag = next_mem_tag_++;
        perf_stats_.mem_accesses++;
        
        std::cout << "[MEM_ACCESS] walk_tag=" << walk.req.tag
                 << " vaddr=0x" << std::hex << walk.req.addr << std::dec
                 << " level=" << (int)walk.level
                 << " mem_tag=" << walk.mem_req_tag
                 << " pte_addr=0x" << std::hex << pte_addr << std::dec
                 << " mem_count=" << perf_stats_.mem_accesses
                 << " walk_count=" << perf_stats_.walks << std::endl;

        // Create memory request for PTE read
        MemReq mem_req;
        mem_req.addr = pte_addr;
        mem_req.p_addr = pte_addr;  // Physical address
        mem_req.write = false;
        mem_req.tag = walk.mem_req_tag;  // Use unique tag to match response to walk
        mem_req.cid = walk.req.cid;
        mem_req.uuid = walk.req.uuid;

        // Push request to memory port (SimPort handles backpressure internally)
        MemReqPort.push(mem_req, 1);
    }

    Config config_;
    std::map<uint32_t, WalkState> active_walks_;  // Active walks keyed by unique tag
    std::queue<MemReq> request_queue_;  // Buffer for queued requests (from arbiter)
    PerfStats perf_stats_;
    uint32_t next_walk_id_;  // Unique ID generator for walks
};

} // namespace vortex
