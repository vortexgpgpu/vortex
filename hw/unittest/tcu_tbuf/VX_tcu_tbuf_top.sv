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

`ifdef TCU_WGMMA_ENABLE

// Flat-port testbench wrapper around VX_tcu_tbuf.
// All VX_tcu_lmem_if signals are exposed as individual ports so that
// the Verilator testbench can drive LMEM read responses directly.

module VX_tcu_tbuf_top import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    // Bank address width for LMEM (word-addressed within one bank)
    parameter BANK_ADDR_WIDTH = `LMEM_LOG_SIZE - `CLOG2(`XLEN / 8) - `CLOG2(`LMEM_NUM_BANKS)
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [`PERF_CTR_BITS-1:0] tbuf_fetch_stalls,
    output wire [`PERF_CTR_BITS-1:0] lmem_reads,
`endif

    // Execute-side observation (driven by testbench)
    input  wire                     req_valid,
    input  wire                     req_fire,
    input  wire [NW_WIDTH-1:0]      req_wid,
    input  wire                     req_is_sparse,
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_n,
    input  wire [3:0]               req_step_k,
    input  wire [3:0]               req_fmt_s,
    input  wire [`XLEN-1:0]         req_desc_a,
    input  wire [`XLEN-1:0]         req_desc_b,

    // LMEM read request (master → testbench memory model)
    output wire                                        lmem_req_valid,
    output wire [BANK_ADDR_WIDTH-1:0]         lmem_req_addr,
    input  wire                                        lmem_req_ready,

    // LMEM read response (testbench memory model → master)
    input  wire                                        lmem_rsp_valid,
    input  wire [`LMEM_NUM_BANKS-1:0][`XLEN-1:0]      lmem_rsp_data,

    // Tile buffer outputs
    output wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0]        tbuf_rs1_data,
    output wire [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] tbuf_rs2_data,
`ifdef TCU_SPARSE_ENABLE
    output wire [TCU_MAX_META_BLOCK_WIDTH-1:0]         tbuf_sp_meta,
`endif
    output wire                                        tbuf_ready
);

    localparam NUM_BANKS       = `LMEM_NUM_BANKS;
    localparam LMEM_DATA_WIDTH = NUM_BANKS * `XLEN;

    VX_tcu_lmem_if #(
        .DATA_WIDTH (LMEM_DATA_WIDTH),
        .ADDR_WIDTH (BANK_ADDR_WIDTH)
    ) tcu_lmem_if();

    assign lmem_req_valid          = tcu_lmem_if.req_valid;
    assign lmem_req_addr           = tcu_lmem_if.req_addr;
    assign tcu_lmem_if.req_ready   = lmem_req_ready;
    assign tcu_lmem_if.rsp_valid   = lmem_rsp_valid;
    assign tcu_lmem_if.rsp_data    = lmem_rsp_data;

    VX_tcu_tbuf #(
        .INSTANCE_ID     ("tcu_tbuf_top"),
        .TCU_TBUF_SIZE   (`NUM_WARPS),
        .NUM_BANKS       (NUM_BANKS),
        .BANK_ADDR_WIDTH (BANK_ADDR_WIDTH)
    ) tbuf (
        .clk             (clk),
        .reset           (reset),
`ifdef PERF_ENABLE
        .tbuf_fetch_stalls (tbuf_fetch_stalls),
        .lmem_reads        (lmem_reads),
`endif
        .req_valid       (req_valid),
        .req_fire        (req_fire),
        .req_wid         (req_wid),
        .req_is_sparse   (req_is_sparse),
        .req_step_m      (req_step_m),
        .req_step_n      (req_step_n),
        .req_step_k      (req_step_k),
        .req_fmt_s       (req_fmt_s),
        .req_desc_a      (req_desc_a),
        .req_desc_b      (req_desc_b),
        .tcu_lmem_if     (tcu_lmem_if),
        .tbuf_rs1_data   (tbuf_rs1_data),
        .tbuf_rs2_data   (tbuf_rs2_data),
`ifdef TCU_SPARSE_ENABLE
        .tbuf_sp_meta    (tbuf_sp_meta),
`endif
        .tbuf_ready      (tbuf_ready)
    );

endmodule

`endif // TCU_WGMMA_ENABLE
