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
        uint64_t base_ppn;       // Base PPN from SATP register
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
        , current_level_ppn_(0)
    {
        DT(1, "PTW created: " << name << " levels=" << (int)config.pt_levels);
    }

    ~PTW() {}

    void reset() {
        state_ = State::IDLE;
        current_level_ = 0;
        current_level_ppn_ = 0;
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
                    current_level_ppn_ = config_.base_ppn;  // Start with SATP base PPN
                    state_ = State::READING_PTE;

                    DT(1, "=== PTW START: vaddr=0x" << std::hex << current_req_.addr << std::dec
                       << " tag=" << current_req_.tag << " base_ppn=0x" << config_.base_ppn);

                    ++perf_stats_.walks;

                    // Issue first PTE read
                    issue_pte_read();
                }
                break;

            case State::READING_PTE:
                // Wait for PTE response
                if (!MemRspPort.empty()) {
                    const MemRsp& pte_rsp = MemRspPort.front();
                    MemRspPort.pop();  // Consume response

                    // PTE read completed - timing is determined by main simulation
                    DT(1, "  PTW Level " << (int)current_level_ << " PTE read done");

                    // Parse PTE to extract PPN for next level (simplified for performance modeling)
                    // In a real implementation, we would parse the actual PTE format
                    // For performance modeling, we simulate the PPN extraction
                    uint64_t pte_ppn = parse_pte_ppn(pte_rsp);

                    // Check if we need more levels
                    if (current_level_ > 0) {
                        // Move to next level with PPN from current level's PTE
                        current_level_--;
                        current_level_ppn_ = pte_ppn;
                        DT(1, "  PTW Level " << (int)current_level_ << " using PPN=0x" << std::hex << pte_ppn << std::dec);
                        issue_pte_read();
                    } else {
                        // All levels done - send response back to TLB
                        MemRsp tlb_rsp;
                        tlb_rsp.tag = current_req_.tag;
                        tlb_rsp.cid = current_req_.cid;
                        tlb_rsp.uuid = current_req_.uuid;

                        // PTW latency is determined by actual memory system response
                        // The PTW completes when memory responses arrive, not based on internal cycles

                        // Push response immediately - latency already accounted for by memory system
                        CoreRspPort.push(tlb_rsp, 1);

                        DT(1, "=== PTW COMPLETE: vaddr=0x" << std::hex << current_req_.addr << std::dec << " final_ppn=0x" << pte_ppn << " ===");

                        state_ = State::IDLE;
                    }
                }
                break;
        }

        // PTW timing is determined by main simulation, not internal counter
    }

    PerfStats perf_stats() const {
        return perf_stats_;
    }

    void set_base_ppn(uint64_t base_ppn) {
        config_.base_ppn = base_ppn;
        DT(1, "PTW::set_base_ppn: base_ppn=0x" << std::hex << base_ppn << std::dec);
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

    // Calculate PTE address
    uint64_t get_pte_address(uint64_t base_ppn, uint64_t vpn) const {
        // PTE_addr = base_ppn * PT_SIZE + vpn * PTE_SIZE
        return (base_ppn * PT_SIZE) + (vpn * config_.pte_size);
    }

    // Parse PTE to extract PPN (simplified for performance modeling)
    uint64_t parse_pte_ppn(const MemRsp& /*pte_rsp*/) const {
        // For performance modeling, we simulate PPN extraction
        // In a real implementation, this would parse the actual PTE format
        // For now, we simulate realistic PPN values based on the address pattern

        uint64_t simulated_ppn = 0;

        #ifdef XLEN_32
        // SV32: PPN is bits [31:10] of PTE
        // For simulation, we use a pattern based on the virtual address
        uint64_t vpn = extract_vpn(current_req_.addr, current_level_);
        simulated_ppn = (vpn & 0x3FF) | ((current_level_ppn_ & 0xFFFFF000) << 10);
        #else
        // SV39: PPN is bits [53:10] of PTE
        // For simulation, we use a pattern based on the virtual address
        uint64_t vpn = extract_vpn(current_req_.addr, current_level_);
        simulated_ppn = (vpn & 0x1FF) | ((current_level_ppn_ & 0xFFFFF000) << 9);
        #endif

        return simulated_ppn;
    }

    // Issue PTE read for current level
    void issue_pte_read() {
        uint64_t vpn = extract_vpn(current_req_.addr, current_level_);
        uint64_t pte_addr = get_pte_address(current_level_ppn_, vpn);

        DT(1, "  PTW Level " << (int)current_level_ << " PTE read start: vpn=0x" << std::hex << vpn
           << " pte_addr=0x" << pte_addr << std::dec);

        // Create memory request for PTE read
        MemReq mem_req;
        mem_req.addr = pte_addr;
        mem_req.p_addr = pte_addr;  // Physical address
        mem_req.write = false;
        mem_req.tag = 0;  // Don't need to track tag internally
        mem_req.cid = current_req_.cid;
        mem_req.uuid = current_req_.uuid;

        // Let the cache hierarchy determine the actual latency
        MemReqPort.push(mem_req, 1);

        ++perf_stats_.mem_accesses;
    }

    Config config_;
    State state_;
    uint8_t current_level_;
    uint64_t current_level_ppn_;  // Current level's PPN for PTE calculation
    MemReq current_req_;
    PerfStats perf_stats_;
};

} // namespace vortex

