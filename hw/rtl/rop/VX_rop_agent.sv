//!/bin/bash

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

`include "VX_rop_define.vh"

module VX_rop_agent import VX_rop_pkg::*; #(
    parameter CORE_ID = 0,
    parameter NUM_LANES = 1
) (
    input wire clk, 
    input wire reset,

    // Inputs    
    VX_execute_if.slave     execute_if,
    VX_sfu_csr_if.slave     rop_csr_if, 

    // Outputs    
    VX_rop_bus_if.master    rop_bus_if,
    VX_commit_if.master     commit_if
);
    `UNUSED_PARAM (CORE_ID)
    localparam PID_BITS   = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH  = `UP(PID_BITS);

    wire [NUM_LANES-1:0][`VX_ROP_DIM_BITS-1:0] sfu_exe_pos_x;
    wire [NUM_LANES-1:0][`VX_ROP_DIM_BITS-1:0] sfu_exe_pos_y;
    wire [NUM_LANES-1:0]                       sfu_exe_face;
    wire [NUM_LANES-1:0][31:0]                 sfu_exe_color;
    wire [NUM_LANES-1:0][`VX_ROP_DEPTH_BITS-1:0] sfu_exe_depth;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign sfu_exe_face[i]  = execute_if.data.rs1_data[i][0];
        assign sfu_exe_pos_x[i] = execute_if.data.rs1_data[i][1 +: `VX_ROP_DIM_BITS];
        assign sfu_exe_pos_y[i] = execute_if.data.rs1_data[i][16 +: `VX_ROP_DIM_BITS];
        assign sfu_exe_color[i] = execute_if.data.rs2_data[i][31:0];
        assign sfu_exe_depth[i] = execute_if.data.rs3_data[i][`VX_ROP_DEPTH_BITS-1:0];
    end

    // CSRs access

    rop_csrs_t rop_csrs;

    VX_rop_csr #(
        .CORE_ID   (CORE_ID),
        .NUM_LANES (NUM_LANES)
    ) rop_csr (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .rop_csr_if (rop_csr_if),

        // outputs
        .rop_csrs   (rop_csrs)
    );

    `UNUSED_VAR (rop_csrs)

    wire rop_req_valid, rop_req_ready;
    wire rop_rsp_valid, rop_rsp_ready;

    // it is possible to have ready = f(valid) when using arbiters, 
    // because of that we need to decouple execute_if and commit_if handshake with a pipe register

    VX_elastic_buffer #(
        .DATAW   (`UUID_WIDTH + NUM_LANES * (1 + 2 * `VX_ROP_DIM_BITS + 32 + `VX_ROP_DEPTH_BITS + 1)),
        .SIZE    (2),
        .OUT_REG (2) // external bus should be registered
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rop_req_valid),
        .ready_in  (rop_req_ready),
        .data_in   ({execute_if.data.uuid, execute_if.data.tmask, sfu_exe_pos_x, sfu_exe_pos_y, sfu_exe_color, sfu_exe_depth, sfu_exe_face}),
        .data_out  ({rop_bus_if.req_data.uuid, rop_bus_if.req_data.mask, rop_bus_if.req_data.pos_x, rop_bus_if.req_data.pos_y, rop_bus_if.req_data.color, rop_bus_if.req_data.depth, rop_bus_if.req_data.face}),
        .valid_out (rop_bus_if.req_valid),
        .ready_out (rop_bus_if.req_ready)
    );

    assign rop_req_valid = execute_if.valid && rop_rsp_ready;
    assign execute_if.ready = rop_req_ready && rop_rsp_ready;
    assign rop_rsp_valid = execute_if.valid && rop_req_ready;

    VX_elastic_buffer #(
        .DATAW (`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `XLEN + PID_WIDTH + 1 + 1),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rop_rsp_valid),
        .ready_in  (rop_rsp_ready),
        .data_in   ({execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.PC, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop}),
        .data_out  ({commit_if.data.uuid, commit_if.data.wid, commit_if.data.tmask, commit_if.data.PC, commit_if.data.pid, commit_if.data.sop, commit_if.data.eop}),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready)
    );

    assign commit_if.data.data = '0;
    assign commit_if.data.rd   = '0;
    assign commit_if.data.wb   = 0;

`ifdef DBG_TRACE_ROP
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%d: core%0d-rop-req: wid=%0d, PC=0x%0h, tmask=%b, x=", $time, CORE_ID, execute_if.data.wid, execute_if.data.PC, execute_if.data.tmask));
            `TRACE_ARRAY1D(1, sfu_exe_pos_x, NUM_LANES);
            `TRACE(1, (", y="));
            `TRACE_ARRAY1D(1, sfu_exe_pos_y, NUM_LANES);
            `TRACE(1, (", face="));
            `TRACE_ARRAY1D(1, sfu_exe_face, NUM_LANES);
            `TRACE(1, (", color="));
            `TRACE_ARRAY1D(1, sfu_exe_color, NUM_LANES);
            `TRACE(1, (", depth="));
            `TRACE_ARRAY1D(1, sfu_exe_depth, NUM_LANES);
            `TRACE(1, (", face=%b (#%0d)\n", sfu_exe_face, execute_if.data.uuid));
        end
    end
`endif

endmodule
