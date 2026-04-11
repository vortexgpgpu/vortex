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
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,
    VX_dxa_req_bus_if.master dxa_req_bus_if,
    VX_txbar_bus_if.master  txbar_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (execute_if.data.rs3_data)

    // Wgather-based layout (lane index = thread_id & 3):
    //   Lane 0: rs1=smem_addr, rs2=coord2
    //   Lane 1: rs1=meta,      rs2=coord3
    //   Lane 2: rs1=coord0,    rs2=coord4
    //   Lane 3: rs1=coord1,    rs2=cta_mask (multicast)
    wire [`XLEN-1:0] lane0_rs1 = execute_if.data.rs1_data[0];
    wire [`XLEN-1:0] lane1_rs1 = execute_if.data.rs1_data[1];
    wire [`XLEN-1:0] lane2_rs1 = execute_if.data.rs1_data[2];
    wire [`XLEN-1:0] lane3_rs1 = execute_if.data.rs1_data[3];
    wire [`XLEN-1:0] lane0_rs2 = execute_if.data.rs2_data[0];
    wire [`XLEN-1:0] lane1_rs2 = execute_if.data.rs2_data[1];
    wire [`XLEN-1:0] lane2_rs2 = execute_if.data.rs2_data[2];
    wire [`XLEN-1:0] lane3_rs2 = execute_if.data.rs2_data[3];
    `UNUSED_VAR (lane3_rs2)

    // Build dxa_req payload
    dxa_req_data_t dxa_req_data_in;
    assign dxa_req_data_in.core_id   = NC_WIDTH'(CORE_ID);
    assign dxa_req_data_in.uuid      = execute_if.data.header.uuid;
    assign dxa_req_data_in.wid       = execute_if.data.header.wid;
    assign dxa_req_data_in.smem_addr = lane0_rs1;
    assign dxa_req_data_in.meta      = lane1_rs1;
    assign dxa_req_data_in.coords[0] = lane2_rs1;
    assign dxa_req_data_in.coords[1] = lane3_rs1;
    assign dxa_req_data_in.coords[2] = lane0_rs2;
    assign dxa_req_data_in.coords[3] = lane1_rs2;
    assign dxa_req_data_in.coords[4] = lane2_rs2;
    assign dxa_req_data_in.cta_mask  = lane3_rs2[`NUM_WARPS-1:0];

    // Build txbar payload
    wire [BAR_ADDR_W-1:0] bar_addr;
    if (`NUM_WARPS > 1) begin : g_bar_addr
        assign bar_addr = {lane1_rs1[4 +: NW_BITS], lane1_rs1[(4 + BAR_ID_SHIFT) +: NB_BITS]};
    end else begin : g_bar_addr_wo
        assign bar_addr = lane1_rs1[(4 + BAR_ID_SHIFT) +: NB_BITS];
    end

    txbar_t txbar_data_in;
    assign txbar_data_in.addr    = bar_addr;
    assign txbar_data_in.is_done = 1'b0;

    // Output elastic buffers break the UNOPTFLAT combinatorial loop
    // between dxa_req_arb and txbar_arb through this unit.
    wire dxa_buf_ready, txbar_buf_ready, wb_ready;
    wire accept = dxa_buf_ready && txbar_buf_ready && wb_ready;
    wire fire   = execute_if.valid && accept;

    assign execute_if.ready = accept;

    VX_elastic_buffer #(
        .DATAW ($bits(dxa_req_data_t)),
        .SIZE  (2)
    ) dxa_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fire),
        .ready_in  (dxa_buf_ready),
        .data_in   (dxa_req_data_in),
        .valid_out (dxa_req_bus_if.req_valid),
        .ready_out (dxa_req_bus_if.req_ready),
        .data_out  (dxa_req_bus_if.req_data)
    );

    VX_elastic_buffer #(
        .DATAW ($bits(txbar_t)),
        .SIZE  (2)
    ) txbar_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fire),
        .ready_in  (txbar_buf_ready),
        .data_in   (txbar_data_in),
        .valid_out (txbar_bus_if.valid),
        .ready_out (txbar_bus_if.ready),
        .data_out  (txbar_bus_if.data)
    );

    sfu_header_t header_out;

    VX_elastic_buffer #(
        .DATAW ($bits(sfu_header_t)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fire),
        .ready_in  (wb_ready),
        .data_in   (execute_if.data.header),
        .data_out  (header_out),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

    assign result_if.data.header = header_out;
    assign result_if.data.data   = '0;

`ifdef DBG_TRACE_DXA
    always_ff @(posedge clk) begin
        if (~reset && dxa_req_bus_if.req_valid && dxa_req_bus_if.req_ready) begin
            `TRACE(1, ("%t: %s dxa-req: wid=%0d, smem=0x%0h, meta=0x%0h, c0=%0d, c1=%0d, c2=%0d, c3=%0d, c4=%0d\n",
                $time, INSTANCE_ID, dxa_req_bus_if.req_data.wid,
                dxa_req_bus_if.req_data.smem_addr, dxa_req_bus_if.req_data.meta,
                dxa_req_bus_if.req_data.coords[0], dxa_req_bus_if.req_data.coords[1],
                dxa_req_bus_if.req_data.coords[2], dxa_req_bus_if.req_data.coords[3],
                dxa_req_bus_if.req_data.coords[4]))
        end
        if (~reset && txbar_bus_if.valid && txbar_bus_if.ready) begin
            `TRACE(1, ("%t: %s txbar-fire: addr=%0d, done=%b\n",
                $time, INSTANCE_ID, txbar_bus_if.data.addr, txbar_bus_if.data.is_done))
        end
    end
`endif

endmodule
