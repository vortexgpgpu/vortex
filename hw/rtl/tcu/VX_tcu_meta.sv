// Copyright 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

`ifdef TCU_SPARSE_ENABLE

module VX_tcu_meta import VX_gpu_pkg::*, VX_tcu_pkg::*;
#(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire          clk,
    input wire          reset,

    // Capture hidden f14/f15 metadata sidebands on the first WMMA compute uop.
    input wire          capture_en,
    input wire [`LOG2UP(`NUM_WARPS)-1:0] capture_wid,
    input wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] meta0_data,
    input wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] meta1_data,

    // Read port (from FEDP path)
    input wire [`LOG2UP(`NUM_WARPS)-1:0] rd_wid,
    input wire [3:0]    step_m,
    input wire [3:0]    step_k,
    output wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_block
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (reset)

    localparam PER_WARP_DEPTH  = TCU_META_PER_WARP_DEPTH;
    localparam META_BLOCK_WIDTH = TCU_MAX_META_BLOCK_WIDTH;
    localparam COLS_PER_LOAD   = TCU_META_COLS_PER_LOAD;
    localparam BANKS_PER_STORE = TCU_BANKS_PER_STORE;
    localparam STORES_PER_COL  = TCU_STORES_PER_COL;

    localparam HALF_K_STEPS = TCU_K_STEPS / 2;
    localparam ADDRW_PW     = `CLOG2(PER_WARP_DEPTH);
    localparam NUM_COLS     = META_BLOCK_WIDTH / 32;
    localparam BANK_DEPTH   = `NUM_WARPS;
    localparam TOTAL_COLS   = PER_WARP_DEPTH * NUM_COLS;
    localparam PACKED_WIDTH = TOTAL_COLS * 32;

    localparam M_STEP_BITS = `CLOG2(TCU_M_STEPS);
    localparam K_STEP_BITS = `CLOG2(HALF_K_STEPS);
    `UNUSED_VAR (step_m)
    `UNUSED_VAR (step_k)

    wire [ADDRW_PW-1:0] bank_sel;
    if (K_STEP_BITS > 0 && M_STEP_BITS > 0) begin : g_addr_mk
        assign bank_sel = {step_m[M_STEP_BITS-1:0], step_k[K_STEP_BITS-1:0]};
    end else if (K_STEP_BITS > 0) begin : g_addr_k
        assign bank_sel = step_k[K_STEP_BITS-1:0];
    end else if (M_STEP_BITS > 0) begin : g_addr_m
        assign bank_sel = step_m[M_STEP_BITS-1:0];
    end else begin : g_addr_zero
        assign bank_sel = '0;
    end

    wire [PACKED_WIDTH-1:0] packed_wdata;
    wire [PACKED_WIDTH-1:0] packed_rdata;

    for (genvar b = 0; b < PER_WARP_DEPTH; ++b) begin : g_meta_banks
        for (genvar c = 0; c < NUM_COLS; ++c) begin : g_col
            localparam STORE_IN_COL   = b / BANKS_PER_STORE;
            localparam THREAD_IN_STORE= b % BANKS_PER_STORE;
            localparam FLAT_STORE     = c * STORES_PER_COL + STORE_IN_COL;
            localparam LOAD_IDX       = FLAT_STORE / COLS_PER_LOAD;
            localparam STORE_IN_LOAD  = FLAT_STORE % COLS_PER_LOAD;
            localparam SRC_THREAD     = STORE_IN_LOAD * BANKS_PER_STORE + THREAD_IN_STORE;

            assign packed_wdata[(b * NUM_COLS + c) * 32 +: 32] =
                (LOAD_IDX == 0) ? 32'(meta0_data[SRC_THREAD]) : 32'(meta1_data[SRC_THREAD]);
        end
    end

    VX_dp_ram #(
        .DATAW    (PACKED_WIDTH),
        .SIZE     (BANK_DEPTH),
        .WRENW    (1),
        .LUTRAM   (1),
        .OUT_REG  (0),
        .RDW_MODE ("W"),
        .RADDR_REG(1)
    ) meta_col_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (capture_en),
        .wren  (1'b1),
        .waddr (capture_wid),
        .wdata (packed_wdata),
        .raddr (rd_wid),
        .rdata (packed_rdata)
    );

    wire same_warp_capture = capture_en && (capture_wid == rd_wid);
    wire [PACKED_WIDTH-1:0] live_packed_data = same_warp_capture ? packed_wdata : packed_rdata;

    wire [PER_WARP_DEPTH-1:0][META_BLOCK_WIDTH-1:0] bank_rdata;
    for (genvar b = 0; b < PER_WARP_DEPTH; ++b) begin : g_unpack
        assign bank_rdata[b] = live_packed_data[b * META_BLOCK_WIDTH +: META_BLOCK_WIDTH];
    end

    assign vld_block = bank_rdata[bank_sel];

endmodule

`endif // TCU_SPARSE_ENABLE
