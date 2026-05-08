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

`include "VX_cache_define.vh"

// Per-LLC-bank AMO helper: the RVA RMW kernel + a small reservation
// table. Synthesizable mirror of SimX's AmoUnit (sim/simx/amo/amo_unit.h).
//
// Single port for reservation activity: at most one of
// {reserve, clear, invalidate} fires per cycle, matching the bank's
// one-commit-per-cycle invariant. The reservation `check` lookup is
// combinational so the SC outcome can be computed in the same S1
// cycle as the commit decision.
//
// LR semantics: install (or refresh) a reservation for hart_id at
// line_addr. LRU eviction when the table is full.
// SC check: a reservation exists for (hart_id, line_addr). RVA
// permits spurious failure but not spurious success — the LRU eviction
// path can drop a hart's reservation, producing failures that are
// still spec-compliant.
// invalidate: drop reservations on `line_addr` belonging to harts
// other than `except_hart_id`. Triggered by every committed write
// reaching the LLC bank's tag array.
module VX_amo_unit import VX_gpu_pkg::*; #(
    parameter NUM_RES_ENTRIES = 4,
    parameter LINE_ADDR_BITS  = 32
) (
    input  wire                          clk,
    input  wire                          reset,

    // Compute kernel (combinational mirror of SimX amo_compute).
    input  amo_op_e                      compute_op,
    input  wire                          compute_amo_unsigned,
    input  wire [1:0]                    compute_width,
    input  wire [63:0]                   compute_old,
    input  wire [63:0]                   compute_rhs,
    output wire [63:0]                   compute_new_word,
    output wire [63:0]                   compute_ret_word,

    // Reservation table activity (single-fire per cycle).
    input  wire                          res_reserve,    // LR commit
    input  wire                          res_clear,      // SC commit (success or fail)
    input  wire                          res_invalidate, // any other-hart write
    input  wire [HART_ID_WIDTH-1:0]      res_hart_id,
    input  wire [LINE_ADDR_BITS-1:0]     res_line_addr,
    output wire                          res_check       // SC outcome (1 = match exists)
);

    // Pure ALU (no state, no clock).
    VX_amo_alu alu (
        .op       (compute_op),
        .amo_unsigned (compute_amo_unsigned),
        .width    (compute_width),
        .old_word (compute_old),
        .rhs      (compute_rhs),
        .new_word (compute_new_word),
        .ret_word (compute_ret_word)
    );

    // Reservation table: register file of NUM_RES_ENTRIES rows.
    localparam LRU_BITS = $clog2(NUM_RES_ENTRIES + 1);

    typedef struct packed {
        logic                          valid;
        logic [HART_ID_WIDTH-1:0]      hart_id;
        logic [LINE_ADDR_BITS-1:0]     line_addr;
        logic [LRU_BITS-1:0]           lru;
    } res_entry_t;

    res_entry_t entries [NUM_RES_ENTRIES];
    reg [LRU_BITS-1:0]  lru_clock;

    // ============================================================
    // SC check (combinational)
    // ============================================================
    // Match exists when some entry holds (hart_id, line_addr).
    wire [NUM_RES_ENTRIES-1:0] check_hits;
    for (genvar i = 0; i < NUM_RES_ENTRIES; ++i) begin : g_check_match
        assign check_hits[i] = entries[i].valid
                            && (entries[i].hart_id == res_hart_id)
                            && (entries[i].line_addr == res_line_addr);
    end
    assign res_check = |check_hits;

    // ============================================================
    // Reserve victim selection (combinational)
    // ============================================================
    // Same-hart re-reserve: overwrite that entry. Else find a free
    // slot. Else evict the LRU one. Both invariants are RVA-conformant
    // (RVA permits spurious failure; LRU eviction merely produces it).
    wire [NUM_RES_ENTRIES-1:0] same_hart;
    wire [NUM_RES_ENTRIES-1:0] free_slot;
    for (genvar i = 0; i < NUM_RES_ENTRIES; ++i) begin : g_reserve_match
        assign same_hart[i] = entries[i].valid && (entries[i].hart_id == res_hart_id);
        assign free_slot[i] = ~entries[i].valid;
    end

    // Find a victim row index. Priority: same_hart > free_slot > LRU.
    // Implemented as nested priority encoders. NUM_RES_ENTRIES is
    // expected small (~4-16) so the logic depth is fine for one cycle.
    function automatic logic [$clog2(NUM_RES_ENTRIES)-1:0]
            select_victim_idx(
                logic [NUM_RES_ENTRIES-1:0] sh,
                logic [NUM_RES_ENTRIES-1:0] fs);
        logic [$clog2(NUM_RES_ENTRIES)-1:0] r;
        logic                               found_sh, found_fs;
        logic [LRU_BITS-1:0]                oldest_lru;
        logic [$clog2(NUM_RES_ENTRIES)-1:0] lru_idx;
        r = '0;
        found_sh = 1'b0;
        for (int i = 0; i < NUM_RES_ENTRIES; i++) begin
            if (sh[i] && !found_sh) begin r = i[$clog2(NUM_RES_ENTRIES)-1:0]; found_sh = 1'b1; end
        end
        if (found_sh) return r;
        found_fs = 1'b0;
        for (int i = 0; i < NUM_RES_ENTRIES; i++) begin
            if (fs[i] && !found_fs) begin r = i[$clog2(NUM_RES_ENTRIES)-1:0]; found_fs = 1'b1; end
        end
        if (found_fs) return r;
        // LRU pick: lowest-LRU among valid entries.
        oldest_lru = '1;
        lru_idx = '0;
        for (int i = 0; i < NUM_RES_ENTRIES; i++) begin
            if (entries[i].valid && (entries[i].lru < oldest_lru)) begin
                oldest_lru = entries[i].lru;
                lru_idx = i[$clog2(NUM_RES_ENTRIES)-1:0];
            end
        end
        return lru_idx;
    endfunction

    wire [$clog2(NUM_RES_ENTRIES)-1:0] reserve_victim_idx =
        select_victim_idx(same_hart, free_slot);

    // ============================================================
    // Sequential update
    // ============================================================
    integer j;
    always @(posedge clk) begin
        if (reset) begin
            for (j = 0; j < NUM_RES_ENTRIES; j = j + 1) begin
                entries[j].valid     <= 1'b0;
                entries[j].hart_id   <= '0;
                entries[j].line_addr <= '0;
                entries[j].lru       <= '0;
            end
            lru_clock <= '0;
        end else begin
            // The three operations are independent — they touch
            // disjoint rows by construction (reserve picks a victim,
            // clear targets the same hart, invalidate targets other
            // harts on the same line). They MUST be allowed to fire
            // together on the same cycle: a successful SC drives both
            // res_clear (own reservation) AND res_invalidate (other
            // harts'), and an if/elseif chain would silently drop the
            // invalidate. Mirrors SimX AmoUnit::clear + invalidate
            // both being called for SC commit.
            if (res_reserve) begin
                lru_clock <= lru_clock + 1'b1;
                entries[reserve_victim_idx].valid     <= 1'b1;
                entries[reserve_victim_idx].hart_id   <= res_hart_id;
                entries[reserve_victim_idx].line_addr <= res_line_addr;
                entries[reserve_victim_idx].lru       <= lru_clock + 1'b1;
            end
            if (res_clear) begin
                for (j = 0; j < NUM_RES_ENTRIES; j = j + 1) begin
                    if (entries[j].valid
                     && (entries[j].hart_id   == res_hart_id)
                     && (entries[j].line_addr == res_line_addr)) begin
                        entries[j].valid <= 1'b0;
                    end
                end
            end
            if (res_invalidate) begin
                for (j = 0; j < NUM_RES_ENTRIES; j = j + 1) begin
                    if (entries[j].valid
                     && (entries[j].line_addr == res_line_addr)
                     && (entries[j].hart_id != res_hart_id)) begin
                        entries[j].valid <= 1'b0;
                    end
                end
            end
        end
    end

endmodule
