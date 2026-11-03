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

// VX_tcu_agu with flat ports: the TCU_LD operands in, the LSU client port and
// the metadata RAM write port out.
module VX_tcu_agu_top import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter NUM_LANES = `VX_CFG_NUM_TCU_LANES,
    parameter LSU_LANES = `VX_CFG_NUM_LSU_LANES
) (
    input wire                              clk,
    input wire                              reset,

    input wire                              ld_valid,
    input wire [NW_WIDTH-1:0]               ld_wid,
    input wire [4:0]                        ld_slot,
    input wire [`VX_CFG_XLEN-1:0]           ld_addr,
    output wire                             ld_ready,

    output wire                             req_valid,
    output wire                             req_rw,
    output wire [LSU_LANES-1:0]             req_mask,
    output wire [LSU_LANES*LSU_ADDR_WIDTH-1:0] req_addr,
    output wire [LSU_CLIENT_TAG_WIDTH-1:0]  req_tag,
    input wire                              req_ready,

    input wire                              rsp_valid,
    input wire [LSU_LANES-1:0]              rsp_mask,
    input wire [LSU_LANES*LSU_WORD_SIZE*8-1:0] rsp_data,
    input wire [LSU_CLIENT_TAG_WIDTH-1:0]   rsp_tag,
    input wire                              rsp_sop,
    input wire                              rsp_eop,
    output wire                             rsp_ready,

    output wire                             meta_wr_en,
    output wire [NW_WIDTH-1:0]              meta_wr_wid,
    output wire [4:0]                       meta_wr_idx,
    output wire [TCU_BLOCK_CAP-1:0]         meta_wr_mask,
    output wire [TCU_BLOCK_CAP*32-1:0]      meta_wr_data,

    output wire                             result_valid,
    output wire [NW_WIDTH-1:0]              result_wid,
    input wire                              result_ready
);
    VX_lsu_sched_if client_if();

    tcu_execute_t ld_data;
    always_comb begin
        ld_data = '0;
        ld_data.header.wid = ld_wid;
        ld_data.op_args.tcu.fmt_d = ld_slot;
        ld_data.rs1_data[0] = ld_addr;
    end

    wire [TCU_BLOCK_CAP-1:0][31:0] meta_wr_data_w;
    tcu_result_t result_data;

    VX_tcu_agu #(
        .INSTANCE_ID ("agu"),
        .NUM_LANES   (NUM_LANES)
    ) agu (
        .clk          (clk),
        .reset        (reset),
        .ld_valid     (ld_valid),
        .ld_data      (ld_data),
        .ld_ready     (ld_ready),
        .client_if    (client_if),
        .meta_wr_en   (meta_wr_en),
        .meta_wr_wid  (meta_wr_wid),
        .meta_wr_idx  (meta_wr_idx),
        .meta_wr_mask (meta_wr_mask),
        .meta_wr_data (meta_wr_data_w),
        .result_valid (result_valid),
        .result_data  (result_data),
        .result_ready (result_ready)
    );

    assign meta_wr_data = meta_wr_data_w;
    assign result_wid   = result_data.header.wid;
    `UNUSED_VAR (result_data)

    assign req_valid = client_if.req_valid;
    assign req_rw    = client_if.req_data.rw;
    assign req_mask  = client_if.req_data.mask;
    assign req_addr  = client_if.req_data.addr;
    assign req_tag   = client_if.req_data.tag;
    assign client_if.req_ready = req_ready;
    `UNUSED_VAR (client_if.req_data)

    assign client_if.rsp_valid     = rsp_valid;
    assign client_if.rsp_data.mask = rsp_mask;
    assign client_if.rsp_data.data = rsp_data;
    assign client_if.rsp_data.tag  = rsp_tag;
    assign client_if.rsp_data.sop  = rsp_sop;
    assign client_if.rsp_data.eop  = rsp_eop;
    assign rsp_ready = client_if.rsp_ready;

endmodule
