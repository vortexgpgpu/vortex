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

    // Write port (meta_store instruction)
    input wire          wr_en,
    input wire [`LOG2UP(`NUM_WARPS)-1:0] wr_wid,
    input wire [3:0]    wr_idx, // flat store index within the warp's metadata block
    input wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] wr_data,

    // Read port (from FEDP path)
    input wire [`LOG2UP(`NUM_WARPS)-1:0] rd_wid,
    input wire [3:0]    step_m,
    input wire [3:0]    step_k,
    output wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_block
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // Local parameters
    localparam PER_WARP_DEPTH  = TCU_META_PER_WARP_DEPTH;
    localparam META_BLOCK_WIDTH = TCU_MAX_META_BLOCK_WIDTH;
    localparam COLS_PER_LOAD   = TCU_META_COLS_PER_LOAD;
    localparam BANKS_PER_STORE = TCU_BANKS_PER_STORE;
    localparam STORES_PER_COL  = TCU_STORES_PER_COL;

    localparam HALF_K_STEPS = TCU_K_STEPS / 2;
    localparam ADDRW_PW     = `CLOG2(PER_WARP_DEPTH);
    localparam NUM_COLS     = META_BLOCK_WIDTH / 32;
    localparam BANK_DEPTH   = `NUM_WARPS;
    localparam BANK_ADDRW   = `LOG2UP(BANK_DEPTH);
    `UNUSED_PARAM (BANK_ADDRW)

    localparam TOTAL_COLS   = PER_WARP_DEPTH * NUM_COLS;
    localparam PACKED_WIDTH = TOTAL_COLS * 32;

    localparam LG_CPL = $clog2((COLS_PER_LOAD > 1) ? COLS_PER_LOAD : 2);
    localparam LG_PD  = $clog2(PER_WARP_DEPTH);
    localparam LG_SPC = (STORES_PER_COL > 1) ? $clog2(STORES_PER_COL) : 1;

    // Bank select: same generate-if as original per_warp_raddr
    localparam M_STEP_BITS = `CLOG2(TCU_M_STEPS);
    localparam K_STEP_BITS = `CLOG2(HALF_K_STEPS);
    if (M_STEP_BITS < 4) `UNUSED_VAR (step_m[3:M_STEP_BITS])
    if (K_STEP_BITS < 4) `UNUSED_VAR (step_k[3:K_STEP_BITS])

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

    // Write decode: flat store index → actual column, sub-store, bank-enable, write data
    wire [3:0] meta_actual_col_idx;
    wire [LG_SPC-1:0] sub_store_idx;
    if (STORES_PER_COL > 1) begin : g_meta_spc
        assign meta_actual_col_idx = 4'(wr_idx >> LG_SPC);
        assign sub_store_idx = wr_idx[LG_SPC-1:0];
    end else begin : g_meta_spc
        assign meta_actual_col_idx = wr_idx;
        assign sub_store_idx = '0;
    end
    `UNUSED_VAR (sub_store_idx)

    wire [PER_WARP_DEPTH-1:0] meta_wr_bank_en;
    for (genvar b = 0; b < PER_WARP_DEPTH; ++b) begin : g_bank_en
        if (STORES_PER_COL > 1) begin : g_partial
            assign meta_wr_bank_en[b] = (LG_SPC'(b / BANKS_PER_STORE) == sub_store_idx);
        end else begin : g_partial
            assign meta_wr_bank_en[b] = 1'b1;
        end
    end

    wire [PER_WARP_DEPTH-1:0][31:0] meta_wr_data;
    if (STORES_PER_COL > 1) begin : g_meta_wr_mode
        for (genvar r = 0; r < PER_WARP_DEPTH; ++r) begin : g_meta_wr
            assign meta_wr_data[r] = 32'(wr_data[r % BANKS_PER_STORE]);
        end
    end else begin : g_meta_wr_mode
        wire [$clog2(TCU_BLOCK_CAP)-1:0] meta_thread_offset;
        if (COLS_PER_LOAD > 1) begin : g_meta_off
            assign meta_thread_offset = {meta_actual_col_idx[LG_CPL-1:0], {LG_PD{1'b0}}};
        end else begin : g_meta_off
            assign meta_thread_offset = '0;
        end
        for (genvar r = 0; r < PER_WARP_DEPTH; ++r) begin : g_meta_wr
            assign meta_wr_data[r] = 32'(wr_data[meta_thread_offset + r]);
        end
    end

    // Column write-enable (one-hot from meta_actual_col_idx)
    wire [NUM_COLS-1:0] col_wren;
    for (genvar c = 0; c < NUM_COLS; ++c) begin : g_col_wren
        assign col_wren[c] = (c[3:0] == meta_actual_col_idx);
    end

    // Pack write data and enables for unified RAM
    wire [TOTAL_COLS-1:0]   packed_wren;
    wire [PACKED_WIDTH-1:0] packed_wdata;
    wire [PACKED_WIDTH-1:0] packed_rdata;

    for (genvar b = 0; b < PER_WARP_DEPTH; ++b) begin : g_meta_banks
        for (genvar c = 0; c < NUM_COLS; ++c) begin : g_col
            assign packed_wren[b * NUM_COLS + c] = wr_en && col_wren[c] && meta_wr_bank_en[b];
            assign packed_wdata[(b * NUM_COLS + c) * 32 +: 32] = meta_wr_data[b];
        end
    end

    VX_dp_ram #(
        .DATAW    (PACKED_WIDTH),
        .SIZE     (BANK_DEPTH),
        .WRENW    (TOTAL_COLS),
        .LUTRAM   (1),
        .OUT_REG  (0),
        .RDW_MODE ("W"),
        .RADDR_REG(1)
    ) meta_col_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (|packed_wren),
        .wren  (packed_wren),
        .waddr (wr_wid),
        .wdata (packed_wdata),
        .raddr (rd_wid),
        .rdata (packed_rdata)
    );

    // Read output MUX: select bank based on {step_m, step_k}, then split into per-row slices
    wire [PER_WARP_DEPTH-1:0][META_BLOCK_WIDTH-1:0] bank_rdata;
    for (genvar b = 0; b < PER_WARP_DEPTH; ++b) begin : g_unpack
        assign bank_rdata[b] = packed_rdata[b * META_BLOCK_WIDTH +: META_BLOCK_WIDTH];
    end

    assign vld_block = bank_rdata[bank_sel];

endmodule

`endif // TCU_SPARSE_ENABLE
