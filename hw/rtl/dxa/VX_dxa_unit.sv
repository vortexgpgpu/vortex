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

module VX_dxa_unit import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = 1
) (
    input wire              clk,
    input wire              reset,

    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,
    VX_dxa_req_bus_if.master    dxa_req_bus_if,
    VX_txbar_bus_if.master  txbar_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (execute_if.data.rs3_data)
    localparam LANE_BITS = `CLOG2(NUM_LANES);

    sfu_header_t header_out;
    wire issue_ready_in;
    wire issue_valid_in;
    wire [`UP(LANE_BITS)-1:0] issue_tid;
    wire [`XLEN-1:0] issue_rs1_data;
    wire [`XLEN-1:0] issue_rs2_data;

    if (LANE_BITS != 0) begin : g_issue_tid
        VX_priority_encoder #(
            .N (NUM_LANES),
            .REVERSE (1)
        ) issue_tid_select (
            .data_in   (execute_if.data.header.tmask),
            .index_out (issue_tid),
            `UNUSED_PIN (onehot_out),
            `UNUSED_PIN (valid_out)
        );
    end else begin : g_issue_tid_w0
        assign issue_tid = 0;
    end

    assign issue_rs1_data = execute_if.data.rs1_data[issue_tid];
    assign issue_rs2_data = execute_if.data.rs2_data[issue_tid];

    VX_elastic_buffer #(
        .DATAW ($bits(sfu_header_t)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (issue_valid_in),
        .ready_in  (issue_ready_in),
        .data_in   (execute_if.data.header),
        .data_out  (header_out),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

    if (`EXT_DXA_CLUSTER_LEVEL_ENABLED) begin : g_cluster_level
        wire is_setup0_req;
        wire req_tx_ready;
        wire req_fire;
        wire [BAR_ADDR_W-1:0] setup0_bar_addr;
        wire tx_setup_valid;
        wire tx_rsp_valid;
        wire tx_emit_setup;
        wire tx_emit_rsp;

        assign is_setup0_req = (execute_if.data.op_args.dxa.op == DXA_OP_SETUP0);
        assign req_tx_ready = ~is_setup0_req || txbar_if.ready;
        assign issue_valid_in = execute_if.valid;
        assign execute_if.ready = issue_ready_in && dxa_req_bus_if.req_ready && req_tx_ready;

        assign dxa_req_bus_if.req_valid = issue_valid_in && issue_ready_in && req_tx_ready;
        assign dxa_req_bus_if.req_data.core_id = NC_WIDTH'(CORE_ID);
        assign dxa_req_bus_if.req_data.uuid    = execute_if.data.header.uuid;
        assign dxa_req_bus_if.req_data.wid     = execute_if.data.header.wid;
        assign dxa_req_bus_if.req_data.op      = execute_if.data.op_args.dxa.op;
        assign dxa_req_bus_if.req_data.rs1     = issue_rs1_data;
        assign dxa_req_bus_if.req_data.rs2     = issue_rs2_data;

        if (`NUM_WARPS > 1) begin : g_setup_bar_addr_w
            assign setup0_bar_addr = {issue_rs2_data[NW_BITS-1:0], issue_rs2_data[16 +: NB_BITS]};
        end else begin : g_setup_bar_addr_wo
            assign setup0_bar_addr = issue_rs2_data[16 +: NB_BITS];
        end

        assign req_fire = dxa_req_bus_if.req_valid && dxa_req_bus_if.req_ready;
        assign tx_setup_valid = req_fire && is_setup0_req;
        assign tx_rsp_valid = dxa_req_bus_if.rsp_valid && dxa_req_bus_if.rsp_data.notify_barrier;

        assign tx_emit_setup = tx_setup_valid;
        assign tx_emit_rsp = tx_rsp_valid && ~tx_emit_setup;

        assign txbar_if.valid        = tx_emit_setup || tx_emit_rsp;
        assign txbar_if.data.addr    = tx_emit_setup ? setup0_bar_addr : dxa_req_bus_if.rsp_data.bar_addr;
        assign txbar_if.data.is_done = tx_emit_setup ? 1'b0 : dxa_req_bus_if.rsp_data.done;

        assign dxa_req_bus_if.rsp_ready = ~tx_rsp_valid || (txbar_if.ready && ~tx_emit_setup);

        `UNUSED_VAR (dxa_req_bus_if.rsp_data.core_id)
        `UNUSED_VAR (dxa_req_bus_if.rsp_data.uuid)
        `UNUSED_VAR (dxa_req_bus_if.rsp_data.wid)
    end else begin : g_cluster_level_off
        assign issue_valid_in = execute_if.valid;
        assign execute_if.ready = issue_ready_in;

        assign dxa_req_bus_if.req_valid = 1'b0;
        assign dxa_req_bus_if.req_data  = '0;
        assign dxa_req_bus_if.rsp_ready = 1'b1;
        assign txbar_if.valid       = 1'b0;
        assign txbar_if.data        = '0;

        `UNUSED_VAR (execute_if.data.op_type)
        `UNUSED_VAR (execute_if.data.op_args)
        `UNUSED_VAR (dxa_req_bus_if.req_ready)
        `UNUSED_VAR (dxa_req_bus_if.rsp_valid)
        `UNUSED_VAR (dxa_req_bus_if.rsp_data)
        `UNUSED_VAR (txbar_if.ready)
    end

    assign result_if.data.header = header_out;
    assign result_if.data.data = '0;

endmodule
