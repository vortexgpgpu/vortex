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

module VX_txbar_arb #(
    parameter NUM_REQS = 1,
    parameter OUT_BUF = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire             clk,
    input wire             reset,

    VX_txbar_bus_if.slave  bus_in_if [NUM_REQS],
    VX_txbar_bus_if.master bus_out_if
);

    localparam DATAW = $bits(txbar_t);

    wire [NUM_REQS-1:0]            valid_in;
    wire [NUM_REQS-1:0][DATAW-1:0] data_in;
    wire [NUM_REQS-1:0]            ready_in;

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_data_in
        assign valid_in[i] = bus_in_if[i].valid;
        assign data_in[i]  = bus_in_if[i].data;
        assign bus_in_if[i].ready = ready_in[i];
    end

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (1),
        .DATAW       (DATAW),
        .ARBITER     (ARBITER),
        .OUT_BUF     (OUT_BUF)
    ) req_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_in),
        .ready_in  (ready_in),
        .data_in   (data_in),
        .data_out  (bus_out_if.data),
        .valid_out (bus_out_if.valid),
        .ready_out (bus_out_if.ready),
        `UNUSED_PIN (sel_out)
    );

endmodule
