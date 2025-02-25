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

module VX_pe_switch import VX_gpu_pkg::*; #(
    parameter PE_COUNT        = 0,
    parameter NUM_LANES       = 0,
    parameter REQ_OUT_BUF     = 0,
    parameter RSP_OUT_BUF     = 0,
    parameter `STRING ARBITER = "R",
    parameter PE_SEL_BITS = `CLOG2(PE_COUNT)
) (
    input wire          clk,
    input wire          reset,
    input wire [`UP(PE_SEL_BITS)-1:0] pe_sel,
    VX_execute_if.slave execute_in_if,
    VX_commit_if.master commit_out_if,
    VX_execute_if.master execute_out_if[PE_COUNT],
    VX_commit_if .slave commit_in_if[PE_COUNT]
);
    localparam PID_BITS    = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH   = `UP(PID_BITS);
    localparam REQ_DATAW   = `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `INST_ALU_BITS + $bits(op_args_t) + 1 + `NR_BITS + `NT_WIDTH + (3 * NUM_LANES * `XLEN) + PID_WIDTH + 1 + 1;
    localparam RSP_DATAW   = `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + 1 + NUM_LANES * `XLEN + PID_WIDTH + 1 + 1;

    wire [PE_COUNT-1:0] pe_req_valid;
    wire [PE_COUNT-1:0][REQ_DATAW-1:0] pe_req_data;
    wire [PE_COUNT-1:0] pe_req_ready;

    VX_stream_switch #(
        .DATAW       (REQ_DATAW),
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (PE_COUNT),
        .OUT_BUF     (REQ_OUT_BUF)
    ) req_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (pe_sel),
        .valid_in  (execute_in_if.valid),
        .ready_in  (execute_in_if.ready),
        .data_in   (execute_in_if.data),
        .data_out  (pe_req_data),
        .valid_out (pe_req_valid),
        .ready_out (pe_req_ready)
    );

    for (genvar i = 0; i < PE_COUNT; ++i) begin : g_execute_out_if
        assign execute_out_if[i].valid = pe_req_valid[i];
        assign execute_out_if[i].data = pe_req_data[i];
        assign pe_req_ready[i] = execute_out_if[i].ready;
    end

    ///////////////////////////////////////////////////////////////////////////

    wire [PE_COUNT-1:0] pe_rsp_valid;
    wire [PE_COUNT-1:0][RSP_DATAW-1:0] pe_rsp_data;
    wire [PE_COUNT-1:0] pe_rsp_ready;

    for (genvar i = 0; i < PE_COUNT; ++i) begin : g_commit_in_if
        assign pe_rsp_valid[i] = commit_in_if[i].valid;
        assign pe_rsp_data[i] = commit_in_if[i].data;
        assign commit_in_if[i].ready = pe_rsp_ready[i];
    end

    VX_stream_arb #(
        .NUM_INPUTS (PE_COUNT),
        .DATAW      (RSP_DATAW),
        .ARBITER    (ARBITER),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (pe_rsp_valid),
        .ready_in  (pe_rsp_ready),
        .data_in   (pe_rsp_data),
        .data_out  (commit_out_if.data),
        .valid_out (commit_out_if.valid),
        .ready_out (commit_out_if.ready),
        `UNUSED_PIN (sel_out)
    );

endmodule
