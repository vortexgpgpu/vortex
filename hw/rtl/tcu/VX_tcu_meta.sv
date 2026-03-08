// Copyright 2019-2023
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

`include "VX_define.vh"

`ifdef TCU_SPARSE_ENABLE

/* verilator lint_off UNUSEDSIGNAL */

module VX_tcu_meta import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter META_BLOCK_WIDTH = 64,
    parameter PER_WARP_DEPTH = 4
) (
    input wire          clk,
    input wire          reset,

    // Read port (from FEDP path)
    input wire [`LOG2UP(`NUM_WARPS)-1:0] raddr_wid,
    input wire [3:0]    step_m,
    input wire [3:0]    step_k,
    output wire [META_BLOCK_WIDTH-1:0] vld_meta_block,

    // Write port (meta_store instruction)
    input wire          wr_en,
    input wire [`LOG2UP(`NUM_WARPS)-1:0] wr_wid,
    input wire [3:0]    wr_col_idx,
    input wire [PER_WARP_DEPTH-1:0][31:0] wr_data
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // Local parameters
    localparam HALF_K_STEPS = TCU_K_STEPS / 2;
    localparam ADDRW_PW     = `CLOG2(PER_WARP_DEPTH);
    localparam NUM_COLS     = META_BLOCK_WIDTH / 32;
    localparam BANK_DEPTH   = `NUM_WARPS;
    localparam BANK_ADDRW   = `LOG2UP(BANK_DEPTH);

    // Bank select: same generate-if as original per_warp_raddr
    localparam M_STEP_BITS = `CLOG2(TCU_M_STEPS);
    localparam K_STEP_BITS = `CLOG2(HALF_K_STEPS);

    wire [ADDRW_PW-1:0] bank_sel;
    generate
        if (K_STEP_BITS > 0 && M_STEP_BITS > 0) begin : g_addr_mk
            assign bank_sel = {step_m[M_STEP_BITS-1:0], step_k[K_STEP_BITS-1:0]};
        end else if (K_STEP_BITS > 0) begin : g_addr_k
            assign bank_sel = step_k[K_STEP_BITS-1:0];
        end else if (M_STEP_BITS > 0) begin : g_addr_m
            assign bank_sel = step_m[M_STEP_BITS-1:0];
        end else begin : g_addr_zero
            assign bank_sel = '0;
        end
    endgenerate

    // Post-reset init FSM: runs NUM_WARPS cycles (one per warp)
    reg [BANK_ADDRW:0] init_counter;
    wire init_active = ~init_counter[BANK_ADDRW];
    wire [BANK_ADDRW-1:0] init_addr = init_counter[BANK_ADDRW-1:0];

    always_ff @(posedge clk) begin
        if (reset) begin
            init_counter <= 0;
        end else if (init_active) begin
            init_counter <= init_counter + 1;
        end
    end

    // Column write-enable (one-hot from wr_col_idx)
    wire [NUM_COLS-1:0] col_wren;
    for (genvar c = 0; c < NUM_COLS; ++c) begin : g_col_wren
        assign col_wren[c] = (c[3:0] == wr_col_idx);
    end

    // Per-column RAMs avoid WRENW partial writes (FPGA LUTRAM byte-enable bug).
    // Each column gets its own WRENW=1 RAM — no partial writes needed.
    wire [META_BLOCK_WIDTH-1:0] bank_rdata [PER_WARP_DEPTH];

    for (genvar b = 0; b < PER_WARP_DEPTH; ++b) begin : g_meta_banks
        localparam [31:0] COL_INIT_EVEN = {8{4'b0101}};
        localparam [31:0] COL_INIT_ODD  = {8{4'b1010}};
        localparam [31:0] COL_INIT = (b % 2 == 0) ? COL_INIT_EVEN : COL_INIT_ODD;

        wire                   bank_wr = init_active || wr_en;
        wire [BANK_ADDRW-1:0] bank_wa = init_active ? init_addr : wr_wid;

        for (genvar c = 0; c < NUM_COLS; ++c) begin : g_col
            wire col_wr = init_active ? bank_wr : (bank_wr && col_wren[c]);
            wire [31:0] col_wd = init_active ? COL_INIT : wr_data[b];
            wire [31:0] col_rd;

            VX_dp_ram #(
                .DATAW    (32),
                .SIZE     (BANK_DEPTH),
                .WRENW    (1),
                .OUT_REG  (0),
                .RDW_MODE ("W")
            ) meta_col_ram (
                .clk   (clk),
                .reset (reset),
                .read  (1'b1),
                .write (col_wr),
                .wren  (1'b1),
                .waddr (bank_wa),
                .wdata (col_wd),
                .raddr (raddr_wid),
                .rdata (col_rd)
            );

            assign bank_rdata[b][c*32 +: 32] = col_rd;
        end
    end

    // Read output MUX: select bank based on {step_m, step_k}
    assign vld_meta_block = bank_rdata[bank_sel];

endmodule

/* verilator lint_on UNUSEDSIGNAL */

`endif // TCU_SPARSE_ENABLE
