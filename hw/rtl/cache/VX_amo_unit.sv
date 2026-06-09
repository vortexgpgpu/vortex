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

// Per-LLC-bank AMO helper: the RVA RMW kernel + a small reservation table.
//
// Single port for reservation activity: at most one of
// {reserve, clear, invalidate} fires per cycle, matching the bank's
// one-commit-per-cycle invariant. The reservation `check` lookup is
// combinational so the SC outcome can be computed in the same S1
// cycle as the commit decision.
//
// Reservations are per hart: one slot per hart_id, directly indexed.
// LR semantics: install (or refresh) the requesting hart's reservation
// at line_addr — never disturbs another hart's slot.
// SC check: the requesting hart's slot holds line_addr.
// invalidate: drop reservations on `line_addr` belonging to harts
// other than `except_hart_id`. Triggered by every committed write
// reaching the LLC bank's tag array.
// A hart's reservation is broken only by such a write (never by another
// hart's LR), which guarantees LR/SC forward progress under contention.
module VX_amo_unit import VX_gpu_pkg::*; #(
    parameter NUM_RES_ENTRIES = 4,
    parameter LINE_ADDR_BITS  = 32
) (
    input  wire                          clk,
    input  wire                          reset,

    // Combinational compute kernel.
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

    // Per-hart reservation: one directly-indexed slot per hart_id. Each
    // hart owns its reservation, so another hart's LR never displaces it;
    // only a committed write to the same line breaks it. This guarantees
    // LR/SC forward progress under contention (one hart wins each retry
    // round). Storage is (1 << HART_ID_WIDTH) slots of {valid, line_addr}.
    localparam NUM_HARTS = 1 << HART_ID_WIDTH;
    `UNUSED_PARAM (NUM_RES_ENTRIES)

    reg                      res_valid [NUM_HARTS];
    reg [LINE_ADDR_BITS-1:0] res_line  [NUM_HARTS];

    // ============================================================
    // SC check (combinational)
    // ============================================================
    // The requesting hart's own slot holds the reserved line.
    assign res_check = res_valid[res_hart_id]
                    && (res_line[res_hart_id] == res_line_addr);

    // ============================================================
    // Sequential update
    // ============================================================
    integer h;
    always @(posedge clk) begin
        if (reset) begin
            for (h = 0; h < NUM_HARTS; h = h + 1) begin
                res_valid[h] <= 1'b0;
            end
        end else begin
            // reserve/clear touch only the requesting hart's own slot;
            // invalidate touches other harts on the same line. They are
            // independent and may fire together: a successful SC drives
            // both res_clear (own reservation) AND res_invalidate (other
            // harts'), so an if/elseif chain would drop the invalidate.
            if (res_reserve) begin
                res_valid[res_hart_id] <= 1'b1;
                res_line [res_hart_id] <= res_line_addr;
            end
            if (res_clear) begin
                if (res_valid[res_hart_id]
                 && (res_line[res_hart_id] == res_line_addr)) begin
                    res_valid[res_hart_id] <= 1'b0;
                end
            end
            if (res_invalidate) begin
                for (h = 0; h < NUM_HARTS; h = h + 1) begin
                    if (res_valid[h]
                     && (res_line[h] == res_line_addr)
                     && (HART_ID_WIDTH'(h) != res_hart_id)) begin
                        res_valid[h] <= 1'b0;
                    end
                end
            end
        end
    end

endmodule
