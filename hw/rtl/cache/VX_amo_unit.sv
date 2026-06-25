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

// Per-LLC-bank AMO helper: the RVA RMW kernel + a small reservation cache.
//
// Reservations are tracked by a bounded, fixed-size set of stations per bank
// (NUM_RS, default load-matched to system concurrency) rather than a slot per
// hart. The cache is line-indexed (direct-mapped on the reserved line's low
// address bits): the {hart,tag} payload lives in a synchronous block RAM read
// one stage ahead (look-ahead address, the VX_cache_tags pattern) so the
// registered output lands the same cycle the SC commit decision is made; the
// valid bits live in resettable flops (BRAM contents are not reset).
//
//   LR  : claim the slot for {hart, line}  (overwrites any prior occupant).
//   SC  : succeeds iff the slot still holds {hart, line}; the success store
//         then clears it through the write path below.
//   write: a committed store/RMW to a line clears the slot iff it holds that
//          line (tag match) — breaks the reserver, any hart.
//
// A bounded set with conflict/capacity eviction is RISC-V-legal: SC may fail
// spuriously for any reason, and forward progress is a system property (some
// hart's SC wins each round). This is how real GPUs/CPUs implement LR/SC, and
// it removes the per-hart table's O(NUM_HARTS) storage and CAM.
module VX_amo_unit import VX_gpu_pkg::*; #(
    parameter NUM_RES_ENTRIES = 4,   // reservation stations per bank (NUM_RS)
    parameter LINE_ADDR_BITS  = 32,
    parameter DATA_WIDTH      = 64   // ALU operand width (cache word, capped at 64)
) (
    input  wire                          clk,
    input  wire                          reset,
    input  wire                          pipe_stall,

    // Combinational compute kernel.
    input  amo_op_e                      compute_op,
    input  wire                          compute_unsigned,
    input  wire [1:0]                    compute_width,
    input  wire [63:0]                   compute_old,
    input  wire [63:0]                   compute_rhs,
    output wire [63:0]                   compute_new_word,
    output wire [63:0]                   compute_ret_word,

    // Reservation activity (single-fire per cycle).
    input  wire                          res_reserve,    // LR commit
    input  wire                          res_clear,      // SC commit (success or fail)
    input  wire                          res_invalidate, // committed write to res_line_addr
    input  wire [HART_ID_WIDTH-1:0]      res_hart_id,
    input  wire [LINE_ADDR_BITS-1:0]     res_line_addr,   // committed line (stC)
    input  wire [LINE_ADDR_BITS-1:0]     res_line_addr_n, // line entering stC next cycle
    output wire                          res_check        // SC outcome (1 = match)
);

    // Pure ALU (no state, no clock).
    VX_amo_alu #(
        .DATA_WIDTH (DATA_WIDTH)
    ) alu (
        .op       (compute_op),
        .is_unsigned (compute_unsigned),
        .width    (compute_width),
        .old_word (compute_old),
        .rhs      (compute_rhs),
        .new_word (compute_new_word),
        .ret_word (compute_ret_word)
    );

    // ============================================================
    // Reservation cache: NUM_RS stations, line-indexed.
    // ============================================================
    // Effective capacity is the next power-of-two >= NUM_RS (>=2), so the line
    // index fully covers the storage depth for any requested NUM_RS.
    localparam RS_ADDRW    = `UP(`CLOG2(NUM_RES_ENTRIES));
    localparam RS_DEPTH    = 1 << RS_ADDRW;
    localparam RS_TAG_BITS = LINE_ADDR_BITS - RS_ADDRW;   // {tag,idx} = full line
    localparam RS_DATA_W   = HART_ID_WIDTH + RS_TAG_BITS;  // BRAM payload: {hart, tag}

    wire en = ~pipe_stall;

    // index by the reserved line's low bits; the rest is the stored tag.
    // (only the index bits of the look-ahead address are needed)
    wire [RS_ADDRW-1:0]    rs_idx   = res_line_addr  [RS_ADDRW-1:0];
    wire [RS_ADDRW-1:0]    rs_idx_n = res_line_addr_n[RS_ADDRW-1:0];
    wire [RS_TAG_BITS-1:0] rs_tag   = res_line_addr[LINE_ADDR_BITS-1:RS_ADDRW];
    `UNUSED_VAR (res_line_addr_n)

    // Look-ahead BRAM payload (cache_tags pattern): read at the next index so
    // the registered output presents the entry at the commit cycle (stC); the
    // LR installs {hart,tag} at the committed index.
    wire                 rs_we;
    wire [RS_DATA_W-1:0] rs_wdata = {res_hart_id, rs_tag};
    wire [RS_DATA_W-1:0] rs_rdata;
    VX_dp_ram #(
        .DATAW    (RS_DATA_W),
        .SIZE     (RS_DEPTH),
        .OUT_REG  (1),
        .RDW_MODE ("R")
    ) rs_store (
        .clk   (clk),
        .reset (reset),
        .read  (en),
        .write (rs_we),
        .wren  (1'b1),
        .waddr (rs_idx),
        .raddr (rs_idx_n),
        .wdata (rs_wdata),
        .rdata (rs_rdata)
    );

    // Read-during-write forward: an LR installs at the committed index the same
    // cycle the next op prefetches it; the registered read would miss it, so
    // forward the just-written payload for one cycle.
    wire                 rdw_set = rs_we && (rs_idx == rs_idx_n);
    reg                  rdw_valid_r;
    reg [RS_DATA_W-1:0]  rdw_data_r;
    always @(posedge clk) begin
        if (reset) begin
            rdw_valid_r <= 1'b0;
        end else if (en) begin
            rdw_valid_r <= rdw_set;
            rdw_data_r  <= rs_wdata;
        end
    end
    wire [RS_DATA_W-1:0] rs_data = rdw_valid_r ? rdw_data_r : rs_rdata;

    // current entry at the committed line index: valid from flops, payload from BRAM
    reg  [RS_DEPTH-1:0]      rs_valid;
    wire                     e_valid = rs_valid[rs_idx];
    wire [HART_ID_WIDTH-1:0] e_hart  = rs_data[RS_TAG_BITS +: HART_ID_WIDTH];
    wire [RS_TAG_BITS-1:0]   e_tag   = rs_data[RS_TAG_BITS-1:0];

    wire line_match = e_valid && (e_tag == rs_tag);            // slot holds this line
    wire own_match  = line_match && (e_hart == res_hart_id);   // ...reserved by this hart

    // SC outcome: this hart's reservation on this line is still live.
    assign res_check = own_match;

    // LR installs the payload; a matching SC/store clears the valid bit.
    assign rs_we = res_reserve && en;
    wire   rs_clr = en && ((res_invalidate && line_match)      // any write breaks the reserver
                        || (res_clear && own_match));          // SC clears its own

    always @(posedge clk) begin
        if (reset) begin
            rs_valid <= '0;
        end else begin
            if (rs_we) begin
                rs_valid[rs_idx] <= 1'b1;
            end else if (rs_clr) begin
                rs_valid[rs_idx] <= 1'b0;
            end
        end
    end

endmodule
