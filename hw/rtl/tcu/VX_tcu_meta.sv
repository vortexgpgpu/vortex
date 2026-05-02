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
    input wire [NW_WIDTH-1:0] wid,
    input wire [3:0]    wr_idx, // group index: each group writes COLS_PER_LOAD columns
    input wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] wr_data,

    // Read port (from FEDP path)
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

    localparam TOTAL_COLS   = PER_WARP_DEPTH * NUM_COLS;
    localparam PACKED_WIDTH = TOTAL_COLS * 32;

    localparam LG_SPC = (STORES_PER_COL > 1) ? $clog2(STORES_PER_COL) : 1;

    // Bank select: same generate-if as original per_warp_raddr
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

    // Write decode: wr_idx is a group index — each group writes COLS_PER_LOAD
    // columns in parallel, using thread data from a single register read.

    wire [3:0] group_idx;
    wire [LG_SPC-1:0] sub_store_idx;
    if (STORES_PER_COL > 1) begin : g_meta_spc
        assign group_idx = 4'(wr_idx >> LG_SPC);
        assign sub_store_idx = wr_idx[LG_SPC-1:0];
    end else begin : g_meta_spc
        assign group_idx = wr_idx;
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

    // Column group enable: all COLS_PER_LOAD columns in the selected group
    wire [NUM_COLS-1:0] col_wren;
    for (genvar c = 0; c < NUM_COLS; ++c) begin : g_col_wren
        assign col_wren[c] = (4'(c / COLS_PER_LOAD) == group_idx);
    end

    // Pack write data and enables for unified RAM
    wire [TOTAL_COLS-1:0]   packed_wren;
    wire [PACKED_WIDTH-1:0] packed_wdata;
    wire [PACKED_WIDTH-1:0] packed_rdata;

    for (genvar b = 0; b < PER_WARP_DEPTH; ++b) begin : g_meta_banks
        for (genvar c = 0; c < NUM_COLS; ++c) begin : g_col
            if (STORES_PER_COL > 1) begin : g_wr
                assign packed_wdata[(b * NUM_COLS + c) * 32 +: 32] =
                    32'(wr_data[b % BANKS_PER_STORE]);
            end else begin : g_wr
                // Thread offset for column c within its group: (c % CPL) * PER_WARP_DEPTH + b
                localparam int COL_IN_GROUP = c % COLS_PER_LOAD;
                localparam int THREAD_OFF = COL_IN_GROUP * PER_WARP_DEPTH + b;
                assign packed_wdata[(b * NUM_COLS + c) * 32 +: 32] =
                    32'(wr_data[THREAD_OFF]);
            end
            assign packed_wren[b * NUM_COLS + c] = wr_en && col_wren[c] && meta_wr_bank_en[b];
        end
    end

    localparam META_DEPTH = `NUM_WARPS;
    localparam META_ADDRW = `CLOG2(META_DEPTH);

    wire [`UP(META_ADDRW)-1:0] meta_addr = `UP(META_ADDRW)'(wid);

    VX_dp_ram #(
        .DATAW    (PACKED_WIDTH),
        .SIZE     (META_DEPTH),
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
        .waddr (meta_addr),
        .wdata (packed_wdata),
        .raddr (meta_addr),
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
