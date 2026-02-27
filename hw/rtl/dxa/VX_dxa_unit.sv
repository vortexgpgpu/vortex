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
    VX_tx_bar_bus_if.master tx_bar_if
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

    if (1) begin : g_cluster_level
        reg [1:0] boot_guard_r;
        wire boot_ready = boot_guard_r[1];
        wire issue_valid_safe;
        wire is_setup0_req;
        wire req_tx_ready;
        wire req_fire;
        wire [BAR_ADDR_W-1:0] setup0_bar_addr;
        wire tx_setup_valid;
        wire setup0_is_packed;
        wire [NB_BITS-1:0] setup0_bar_slot;
        wire [NW_BITS-1:0] setup0_bar_owner;

        // Treat X/unknown valid as 0 and add a short boot guard to avoid
        // reset-exit ghost traffic on DXA request/response channels.
        assign issue_valid_safe = boot_ready && (execute_if.valid === 1'b1);
        assign is_setup0_req = (execute_if.data.op_args.dxa.op == DXA_OP_SETUP);
        assign req_tx_ready = ~is_setup0_req || tx_bar_if.ready;
        assign issue_valid_in = issue_valid_safe;
        assign execute_if.ready = issue_ready_in && dxa_req_bus_if.req_ready && req_tx_ready && ~reset;

        assign dxa_req_bus_if.req_valid = issue_valid_safe && issue_ready_in && req_tx_ready;
        assign dxa_req_bus_if.req_data.core_id = NC_WIDTH'(CORE_ID);
        assign dxa_req_bus_if.req_data.uuid    = execute_if.data.header.uuid;
        assign dxa_req_bus_if.req_data.wid     = execute_if.data.header.wid;
        assign dxa_req_bus_if.req_data.op      = execute_if.data.op_args.dxa.op;
        assign dxa_req_bus_if.req_data.rs1     = issue_rs1_data;
        assign dxa_req_bus_if.req_data.rs2     = issue_rs2_data;

        // Packed launch metadata: [3:0]=desc, [30:4]=barrier payload, [31]=marker.
        assign setup0_is_packed = issue_rs2_data[31];
        assign setup0_bar_slot = setup0_is_packed ? issue_rs2_data[20 +: NB_BITS]
                                                  : issue_rs2_data[16 +: NB_BITS];
        if (`NUM_WARPS > 1) begin : g_setup_bar_owner
            assign setup0_bar_owner = setup0_is_packed ? issue_rs2_data[4 +: NW_BITS]
                                                       : issue_rs2_data[NW_BITS-1:0];
        end else begin : g_setup_bar_owner_wo
            assign setup0_bar_owner = '0;
        end

        if (`NUM_WARPS > 1) begin : g_setup_bar_addr_w
            assign setup0_bar_addr = {setup0_bar_owner, setup0_bar_slot};
        end else begin : g_setup_bar_addr_wo
            assign setup0_bar_addr = setup0_bar_slot;
        end

        assign req_fire = boot_ready && dxa_req_bus_if.req_valid && dxa_req_bus_if.req_ready;
        assign tx_setup_valid = req_fire && is_setup0_req;
        assign tx_bar_if.valid        = tx_setup_valid;
        assign tx_bar_if.data.addr    = setup0_bar_addr;
        assign tx_bar_if.data.is_done = 1'b0;

        // Completion path is no longer carried over dxa_req_bus_if.rsp.
        assign dxa_req_bus_if.rsp_ready = 1'b1;

        always @(posedge clk) begin
            if (reset) begin
                boot_guard_r <= 2'b00;
            end else begin
                if (~boot_guard_r[1]) begin
                    boot_guard_r <= boot_guard_r + 2'b01;
                end
            end
        end

        `UNUSED_VAR (dxa_req_bus_if.rsp_valid)
        `UNUSED_VAR (dxa_req_bus_if.rsp_data)

    `ifdef DBG_TRACE_DXA_TX_BAR
        always @(posedge clk) begin
            if (~reset) begin
                if (req_fire) begin
                    `TRACE(1, ("%t: %s dxa-req: op=%0d wid=%0d rs1=0x%0h rs2=0x%0h setup0=%b\n",
                        $time, INSTANCE_ID, dxa_req_bus_if.req_data.op, execute_if.data.header.wid, issue_rs1_data, issue_rs2_data, is_setup0_req))
                end
                if (tx_setup_valid) begin
                    `TRACE(1, ("%t: %s tx-setup: addr=%0d packed=%b slot=%0d owner=%0d\n",
                        $time, INSTANCE_ID, setup0_bar_addr, setup0_is_packed, setup0_bar_slot, setup0_bar_owner))
                end
                if (tx_bar_if.valid && tx_bar_if.ready) begin
                    `TRACE(1, ("%t: %s tx-bar-fire: addr=%0d done=%b\n",
                        $time, INSTANCE_ID, tx_bar_if.data.addr, tx_bar_if.data.is_done))
                end
            end
        end
    `endif
    end

    assign result_if.data.header = header_out;
    assign result_if.data.data = '0;

endmodule
