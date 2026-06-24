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

`include "VX_define.vh"

`ifdef VX_CFG_TCU_META_ENABLE

// Unified TCU metadata SRAM. Merges the sparse (2:4) lane-validity store and
// the MX scale-factor stores (A/B) behind a single warp-indexed block RAM.
//
// All regions share one warp-indexed address space (SIZE = NUM_WARPS) and the
// one broadcast write port from VX_tcu_agu. wr_idx[4] selects the namespace
// (0 = sparse, 1 = MX); within MX wr_idx[0] selects the axis (0 = A, 1 = B);
// within sparse wr_idx[3:0] is the column-group index. This matches the
// TCU_LD destination-register encoding emitted by vx_tensor.h (x0/x1 = SP,
// x16/x17 = MX-A/MX-B). Exactly one region is written per TCU_LD.
//
// Read latency contract: the read address (rd_wid) is the already-registered
// execute_if header wid, so the RAM is read combinationally in the issue
// cycle (OUT_REG=0, RADDR_REG=1). The sparse mux / MX scale extraction in
// VX_tcu_core consume the result in that same cycle.

module VX_tcu_meta import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire          clk,
    input wire          reset,

    // Unified broadcast write port (from VX_tcu_agu).
    input wire                     wr_en,
    input wire [NW_WIDTH-1:0]      wr_wid,
    input wire [4:0]               wr_idx,
    input wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] wr_data,

    // Unified read port (warp id from the FEDP path).
    input wire [NW_WIDTH-1:0]      rd_wid
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    , input wire [3:0]             step_m
    , input wire [3:0]             step_k
    , output wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_block
`endif
`ifdef VX_CFG_TCU_MX_ENABLE
    , output wire [TCU_BLOCK_CAP-1:0][31:0] meta_a
    , output wire [TCU_BLOCK_CAP-1:0][31:0] meta_b
`endif
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // Region layout (32-bit columns). The merged RAM concatenates the enabled
    // regions: [ sparse banks | MX-A scales | MX-B scales ].
    // -----------------------------------------------------------------------
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    localparam SP_COLS = TCU_META_PER_WARP_DEPTH * (TCU_MAX_META_BLOCK_WIDTH / 32);
`else
    localparam SP_COLS = 0;
`endif
`ifdef VX_CFG_TCU_MX_ENABLE
    localparam MX_COLS = TCU_BLOCK_CAP;
`else
    localparam MX_COLS = 0;
`endif

    localparam MERGED_COLS  = SP_COLS + 2 * MX_COLS;
    localparam MERGED_BITS  = MERGED_COLS * 32;

    localparam META_DEPTH = `VX_CFG_NUM_WARPS;
    localparam META_ADDRW = `CLOG2(META_DEPTH);

    // Namespace decode (mirrors the AGU / vx_tensor.h slot encoding). The
    // per-region write strobes are defined inside each region's block below so
    // that nothing is left unused when a region is compiled out.

    wire [MERGED_COLS-1:0]  merged_wren;
    wire [MERGED_BITS-1:0]  merged_wdata;
    wire [MERGED_BITS-1:0]  merged_rdata;

    wire [`UP(META_ADDRW)-1:0] wr_addr = `UP(META_ADDRW)'(wr_wid);
    wire [`UP(META_ADDRW)-1:0] rd_addr = `UP(META_ADDRW)'(rd_wid);

    // -----------------------------------------------------------------------
    // Sparse region: per-thread metadata packed into PER_WARP_DEPTH banks of
    // NUM_COLS columns each, written COLS_PER_LOAD columns at a time.
    // -----------------------------------------------------------------------
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    localparam PER_WARP_DEPTH   = TCU_META_PER_WARP_DEPTH;
    localparam META_BLOCK_WIDTH = TCU_MAX_META_BLOCK_WIDTH;
    localparam COLS_PER_LOAD    = TCU_META_COLS_PER_LOAD;
    localparam BANKS_PER_STORE  = TCU_BANKS_PER_STORE;
    localparam STORES_PER_COL   = TCU_STORES_PER_COL;

    localparam HALF_K_STEPS = TCU_K_STEPS / 2;
    localparam NUM_COLS     = META_BLOCK_WIDTH / 32;
    localparam LG_SPC       = (STORES_PER_COL > 1) ? $clog2(STORES_PER_COL) : 1;

    // Bank select: composed from step_m and step_k bit widths.
    localparam M_STEP_BITS = `CLOG2(TCU_M_STEPS);
    localparam K_STEP_BITS = `CLOG2(HALF_K_STEPS);
    `UNUSED_VAR (step_m)
    `UNUSED_VAR (step_k)

    wire sp_wr = wr_en && !wr_idx[4];

    localparam ADDRW_PW = `CLOG2(PER_WARP_DEPTH);
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
    // columns in parallel from a single register read.
    wire [3:0] group_idx;
    wire [LG_SPC-1:0] sub_store_idx;
    if (STORES_PER_COL > 1) begin : g_meta_spc
        assign group_idx = 4'(wr_idx[3:0] >> LG_SPC);
        assign sub_store_idx = wr_idx[LG_SPC-1:0];
    end else begin : g_meta_spc
        assign group_idx = wr_idx[3:0];
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

    // Column group enable: all COLS_PER_LOAD columns in the selected group.
    wire [NUM_COLS-1:0] col_wren;
    for (genvar c = 0; c < NUM_COLS; ++c) begin : g_col_wren
        assign col_wren[c] = (4'(c / COLS_PER_LOAD) == group_idx);
    end

    // Pack sparse write data/enables into the merged column space (base 0).
    // Sparse region occupies the low columns of the merged RAM (base 0).
    for (genvar b = 0; b < PER_WARP_DEPTH; ++b) begin : g_meta_banks
        for (genvar c = 0; c < NUM_COLS; ++c) begin : g_col
            localparam int COL = b * NUM_COLS + c;
            if (STORES_PER_COL > 1) begin : g_wr
                assign merged_wdata[COL * 32 +: 32] = 32'(wr_data[b % BANKS_PER_STORE]);
            end else begin : g_wr
                // Thread offset for column c within its group: (c % CPL) * PER_WARP_DEPTH + b
                localparam int COL_IN_GROUP = c % COLS_PER_LOAD;
                localparam int THREAD_OFF   = COL_IN_GROUP * PER_WARP_DEPTH + b;
                assign merged_wdata[COL * 32 +: 32] = 32'(wr_data[THREAD_OFF]);
            end
            assign merged_wren[COL] = sp_wr && col_wren[c] && meta_wr_bank_en[b];
        end
    end

    // Read output mux: select bank by {step_m, step_k}, slice per-row.
    wire [PER_WARP_DEPTH-1:0][META_BLOCK_WIDTH-1:0] bank_rdata;
    for (genvar b = 0; b < PER_WARP_DEPTH; ++b) begin : g_unpack
        assign bank_rdata[b] = merged_rdata[b * META_BLOCK_WIDTH +: META_BLOCK_WIDTH];
    end
    assign vld_block = bank_rdata[bank_sel];
`endif

    // -----------------------------------------------------------------------
    // MX region: one full warp-wide scale row per axis, whole-region write.
    // -----------------------------------------------------------------------
`ifdef VX_CFG_TCU_MX_ENABLE
`ifndef VX_CFG_TCU_SPARSE_ENABLE
    `UNUSED_VAR (wr_idx[3:1])
`endif
    // MX A/B scale regions follow the sparse region (base SP_COLS).
    localparam MXA_COL_BASE = SP_COLS;
    localparam MXB_COL_BASE = SP_COLS + MX_COLS;
    wire mx_wr_a = wr_en && wr_idx[4] && !wr_idx[0];
    wire mx_wr_b = wr_en && wr_idx[4] &&  wr_idx[0];

    for (genvar i = 0; i < TCU_BLOCK_CAP; ++i) begin : g_mx_pack
        assign merged_wdata[(MXA_COL_BASE + i) * 32 +: 32] = 32'(wr_data[i]);
        assign merged_wdata[(MXB_COL_BASE + i) * 32 +: 32] = 32'(wr_data[i]);
        assign merged_wren[MXA_COL_BASE + i] = mx_wr_a;
        assign merged_wren[MXB_COL_BASE + i] = mx_wr_b;
    end
    assign meta_a = merged_rdata[(MXA_COL_BASE * 32) +: (TCU_BLOCK_CAP * 32)];
    assign meta_b = merged_rdata[(MXB_COL_BASE * 32) +: (TCU_BLOCK_CAP * 32)];
`endif

    // -----------------------------------------------------------------------
    // Merged warp-indexed block RAM.
    // -----------------------------------------------------------------------
    VX_dp_ram #(
        .DATAW    (MERGED_BITS),
        .SIZE     (META_DEPTH),
        .WRENW    (MERGED_COLS),
        .LUTRAM   (0),
        .OUT_REG  (0),
        .RDW_MODE ("W"),
        .RADDR_REG(1) // rd_wid is registered!
    ) meta_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (|merged_wren),
        .wren  (merged_wren),
        .waddr (wr_addr),
        .wdata (merged_wdata),
        .raddr (rd_addr),
        .rdata (merged_rdata)
    );

endmodule

`endif // VX_CFG_TCU_META_ENABLE
