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
    VX_dxa_req_bus_if.master dxa_req_bus_if,
    VX_txbar_bus_if.master txbar_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (execute_if.data.rs3_data)

    // ── Per-lane register reads ───────────────────────────────────────────────
    // Wgather-based layout (lane index = thread_id & 3):
    //   Lane 0: rs1=smem_addr, rs2=coord2
    //   Lane 1: rs1=meta,      rs2=coord3
    //   Lane 2: rs1=coord0,    rs2=coord4
    //   Lane 3: rs1=coord1,    rs2=0
    wire [`XLEN-1:0] lane0_rs1 = execute_if.data.rs1_data[0];  // smem_addr
    wire [`XLEN-1:0] lane1_rs1 = execute_if.data.rs1_data[1];  // meta
    wire [`XLEN-1:0] lane2_rs1 = execute_if.data.rs1_data[2];  // coord0
    wire [`XLEN-1:0] lane3_rs1 = execute_if.data.rs1_data[3];  // coord1
    wire [`XLEN-1:0] lane0_rs2 = execute_if.data.rs2_data[0];  // coord2
    wire [`XLEN-1:0] lane1_rs2 = execute_if.data.rs2_data[1];  // coord3
    wire [`XLEN-1:0] lane2_rs2 = execute_if.data.rs2_data[2];  // coord4

    sfu_header_t header_out;
    wire issue_ready_in;
    wire issue_valid_in;

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
        wire req_tx_ready;
        wire req_fire;
        wire [BAR_ADDR_W-1:0] setup_bar_addr;
        wire tx_setup_valid;
        wire setup_is_packed;
        wire [NB_BITS-1:0] setup_bar_slot;
        wire [NW_BITS-1:0] setup_bar_owner;

        // Treat X/unknown valid as 0 and add a short boot guard to avoid
        // reset-exit ghost traffic on DXA request/response channels.
        assign issue_valid_safe = boot_ready && (execute_if.valid === 1'b1);

        // Every DXA instruction is a combined setup+issue: always send txbar.
        assign req_tx_ready = txbar_bus_if.ready;

        // Gate response buffer accept on ALL downstream ready signals.
        assign issue_valid_in = issue_valid_safe && dxa_req_bus_if.req_ready && req_tx_ready;
        assign execute_if.ready = issue_ready_in && dxa_req_bus_if.req_ready && req_tx_ready && ~reset;

        assign dxa_req_bus_if.req_valid = issue_valid_safe && issue_ready_in && req_tx_ready;

        // Populate bus with all decoded fields from 4 lanes
        assign dxa_req_bus_if.req_data.core_id = NC_WIDTH'(CORE_ID);
        assign dxa_req_bus_if.req_data.uuid    = execute_if.data.header.uuid;
        assign dxa_req_bus_if.req_data.wid     = execute_if.data.header.wid;
        assign dxa_req_bus_if.req_data.smem_addr = lane0_rs1;
        assign dxa_req_bus_if.req_data.meta      = lane1_rs1;
        assign dxa_req_bus_if.req_data.coords[0] = lane2_rs1;
        assign dxa_req_bus_if.req_data.coords[1] = lane3_rs1;
        assign dxa_req_bus_if.req_data.coords[2] = lane0_rs2;
        assign dxa_req_bus_if.req_data.coords[3] = lane1_rs2;
        assign dxa_req_bus_if.req_data.coords[4] = lane2_rs2;

        // Barrier address from meta (always packed: meta[31]=1).
        // meta[3:0]  = desc_slot
        // meta[30:4] = raw barrier payload:
        //   raw[NW_BITS-1:0]         = bar_owner (warp id)
        //   raw[16 +: NB_BITS]       = bar_slot (barrier index)
        assign setup_is_packed = lane1_rs1[31];
        assign setup_bar_slot = setup_is_packed ? lane1_rs1[20 +: NB_BITS]
                                                : lane1_rs1[16 +: NB_BITS];
        if (`NUM_WARPS > 1) begin : g_setup_bar_owner
            assign setup_bar_owner = setup_is_packed ? lane1_rs1[4 +: NW_BITS]
                                                     : lane1_rs1[NW_BITS-1:0];
        end else begin : g_setup_bar_owner_wo
            assign setup_bar_owner = '0;
        end

        if (`NUM_WARPS > 1) begin : g_setup_bar_addr_w
            assign setup_bar_addr = {setup_bar_owner, setup_bar_slot};
        end else begin : g_setup_bar_addr_wo
            assign setup_bar_addr = setup_bar_slot;
        end

        assign req_fire = boot_ready && dxa_req_bus_if.req_valid && dxa_req_bus_if.req_ready;
        assign tx_setup_valid = req_fire;  // every DXA instruction registers a barrier
        assign txbar_bus_if.valid        = tx_setup_valid;
        assign txbar_bus_if.data.addr    = setup_bar_addr;
        assign txbar_bus_if.data.is_done = 1'b0;

        always @(posedge clk) begin
            if (reset) begin
                boot_guard_r <= 2'b00;
            end else begin
                if (~boot_guard_r[1]) begin
                    boot_guard_r <= boot_guard_r + 2'b01;
                end
            end
        end

    `ifdef DBG_TRACE_DXA
        always @(posedge clk) begin
            if (~reset) begin
                if (req_fire) begin
                    `TRACE(1, ("%t: %s dxa-req: wid=%0d smem=0x%0h meta=0x%0h c0=%0d c1=%0d c2=%0d c3=%0d c4=%0d\n",
                        $time, INSTANCE_ID, execute_if.data.header.wid,
                        lane0_rs1, lane1_rs1, lane2_rs1, lane3_rs1,
                        lane0_rs2, lane1_rs2, lane2_rs2))
                end
                if (tx_setup_valid) begin
                    `TRACE(1, ("%t: %s tx-setup: addr=%0d packed=%b slot=%0d owner=%0d\n",
                        $time, INSTANCE_ID, setup_bar_addr, setup_is_packed, setup_bar_slot, setup_bar_owner))
                end
                if (txbar_bus_if.valid && txbar_bus_if.ready) begin
                    `TRACE(1, ("%t: %s tx-bar-fire: addr=%0d done=%b\n",
                        $time, INSTANCE_ID, txbar_bus_if.data.addr, txbar_bus_if.data.is_done))
                end
            end
        end
    `endif
    end

    assign result_if.data.header = header_out;
    assign result_if.data.data = '0;

endmodule
