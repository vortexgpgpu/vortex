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

module VX_kmu_arb import VX_gpu_pkg::*; #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter NUM_LANES      = 1,
    parameter OUT_BUF        = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    // input request
    VX_kmu_bus_if.slave  bus_in_if [NUM_INPUTS],

    // output requests
    VX_kmu_bus_if.master bus_out_if [NUM_OUTPUTS]
);
    localparam DATAW = NUM_LANES * $bits(kmu_req_t);

    wire [NUM_INPUTS-1:0]             valid_in;
    wire [NUM_INPUTS-1:0][DATAW-1:0]  data_in;
    wire [NUM_INPUTS-1:0]             ready_in;

    wire [NUM_OUTPUTS-1:0]            valid_out;
    wire [NUM_OUTPUTS-1:0][DATAW-1:0] data_out;
    wire [NUM_OUTPUTS-1:0]            ready_out;

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_raster_valid
        assign valid_in[i] = bus_in_if[i].valid;
        assign data_in[i]  = bus_in_if[i].data;
        assign bus_in_if[i].ready = ready_in[i];
    end

    VX_stream_arb #(
        .NUM_INPUTS (NUM_INPUTS),
        .NUM_OUTPUTS(NUM_OUTPUTS),
        .DATAW      (DATAW),
        .ARBITER    (ARBITER),
        .OUT_BUF    (OUT_BUF)
    ) arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .ready_in   (ready_in),
        .data_in    (data_in),
        .data_out   (data_out),
        .valid_out  (valid_out),
        .ready_out  (ready_out),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_raster_bus_out
        assign bus_out_if[i].valid = valid_out[i];
        assign bus_out_if[i].data  = data_out[i];
        assign ready_out[i] = bus_out_if[i].ready;
    end

endmodule
