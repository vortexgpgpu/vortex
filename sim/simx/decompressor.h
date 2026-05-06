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

#include <cstdint>
#include <queue>
#include <vector>
#include <simobject.h>
#include "types.h"

namespace vortex {

struct instr_trace_t;

struct DecompResult {
    uint32_t instr32;  // 32-bit base-ISA encoding
    uint8_t  size;     // bytes consumed: 2 (RVC) or 4 (full)
    bool     illegal;  // reserved/unsupported 16-bit pattern
};

// Stateless RVC decompression. `word` is interpreted as either:
//  - a full 32-bit instruction (low2 == 0b11), returned unchanged with size=4;
//  - a 16-bit compressed instruction in the low half, expanded to 32 bits.
DecompResult rvc_decompress(uint32_t word);

// Per-core RVC fetch SimObject. Owns:
//   - per-warp half-word buffer state (cross-word 32-bit low-half stash)
//   - the refetch queue (drains before fresh schedules so a cross-word
//     completion isn't starved by other warps' fetches)
//
// Mirrors VX_decompressor's behavior in the RTL:
//   - on_icache_rsp() classifies an icache rsp; either emits the raw bits
//     (RVC hword zero-extended, or 4-byte-aligned 32-bit, or combined
//     cross-word 32-bit) or queues a refetch.
//   - pick_request() / commit_request() implement follow_req-prioritized
//     icache request issue.
//
// Resets via SimObject::on_reset (called by SimPlatform on global reset).
class Decompressor : public SimObject<Decompressor> {
public:
    Decompressor(const SimContext& ctx, const char* name, uint32_t num_warps);
    ~Decompressor();

    // Pick the next trace to send to icache. Drains the refetch queue
    // first; otherwise picks `fetch_latch_head` (may be nullptr).
    // Caller pops fetch_latch on a successful send; the refetch queue is
    // popped via commit_request(true).
    struct Pick {
        instr_trace_t* trace;
        Word           addr;
        bool           from_refetch;
    };
    Pick pick_request(instr_trace_t* fetch_latch_head);

    // Notify a successful icache req send. Pops the refetch queue when
    // from_refetch is true.
    void commit_request(bool from_refetch);

    // Process an icache rsp for `trace`. Extracts the right word from
    // `line` (knowing whether this is a refetch from FSM state), runs the
    // RVC FSM, and either:
    //   - sets `trace->code` to the raw bits and returns true (caller
    //     pushes the trace to the decode latch); raw bits are 16-bit RVC
    //     zero-extended, 4-byte-aligned 32-bit, or combined cross-word
    //     32-bit. The decoder later detects RVC from code[1:0] and
    //     decompresses internally;
    //   - queues `trace` for refetch internally and returns false.
    bool on_icache_rsp(instr_trace_t* trace, const mem_block_t& line);

protected:
    void on_reset();
    friend class SimObject<Decompressor>;

private:
    struct RvcSlot {
        bool     needs_second = false;
        uint16_t low_half     = 0;
        Word     inst_pc      = 0;
    };

    // Aligned icache address that was fetched for `trace`, as a function
    // of FSM state. Internal helper used by on_icache_rsp() and pick_request().
    Word fetch_addr(const instr_trace_t* trace) const;

    std::vector<RvcSlot>       state_;
    std::queue<instr_trace_t*> refetch_queue_;
};

} // namespace vortex
