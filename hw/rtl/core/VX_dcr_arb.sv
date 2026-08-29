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

module VX_dcr_arb import VX_gpu_pkg::*; #(
    parameter NUM_REQS    = 1,
    parameter REQ_OUT_BUF = 0,
    parameter RSP_OUT_BUF = 0
) (
    input wire clk,
    input wire reset,

    VX_dcr_bus_if.slave  bus_in_if,
    VX_dcr_bus_if.master bus_out_if [NUM_REQS]
);
    localparam REQ_DATAW = $bits(dcr_req_t);
    localparam RSP_DATAW = $bits(dcr_rsp_t);

    // broadcast request

    wire [NUM_REQS-1:0]                req_valid_out;
    wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_out;

    VX_stream_fork #(
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (REQ_DATAW),
        .OUT_BUF     (REQ_OUT_BUF)
    ) req_fork (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_in_if.req_valid),
        .data_in   (bus_in_if.req_data),
        `UNUSED_PIN (ready_in),
        .valid_out (req_valid_out),
        .data_out  (req_data_out),
        .ready_out ({NUM_REQS{1'b1}})
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_out
        assign bus_out_if[i].req_valid = req_valid_out[i];
        assign bus_out_if[i].req_data  = req_data_out[i];
    end

    // arbitrate response

    wire [NUM_REQS-1:0]                rsp_valid_in;
    wire [NUM_REQS-1:0][RSP_DATAW-1:0] rsp_data_in;

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_rsp_data_in
        assign rsp_valid_in[i] = bus_out_if[i].rsp_valid;
        assign rsp_data_in[i]  = bus_out_if[i].rsp_data;
    end

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (1),
        .DATAW       (RSP_DATAW),
        .OUT_BUF     (RSP_OUT_BUF)
    ) rsp_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (rsp_valid_in),
        .data_in    (rsp_data_in),
        `UNUSED_PIN (ready_in),
        .valid_out  (bus_in_if.rsp_valid),
        .data_out   (bus_in_if.rsp_data),
        .ready_out  (1'b1),
        `UNUSED_PIN (sel_out)
    );

endmodule
