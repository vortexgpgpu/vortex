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

// DXA dispatch: route request+descriptor to first idle worker.

module VX_dxa_dispatch import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_INPUTS  = 1,
    parameter NUM_OUTPUTS = 1,
    parameter BUFFERED    = 0
) (
    input wire clk,
    input wire reset,
    VX_dxa_worker_req_if.slave  req_in  [NUM_INPUTS],
    VX_dxa_worker_req_if.master req_out [NUM_OUTPUTS]
);

    localparam DATAW = $bits(dxa_req_data_t) + $bits(dxa_desc_t);

    wire [NUM_INPUTS-1:0]              valid_in;
    wire [NUM_INPUTS-1:0][DATAW-1:0]   data_in;
    wire [NUM_INPUTS-1:0]              ready_in;
    wire [NUM_OUTPUTS-1:0]             valid_out;
    wire [NUM_OUTPUTS-1:0][DATAW-1:0]  data_out;
    wire [NUM_OUTPUTS-1:0]             ready_out;

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_inputs
        assign valid_in[i]     = req_in[i].valid;
        assign data_in[i]      = {req_in[i].req_data, req_in[i].desc_data};
        assign req_in[i].ready = ready_in[i];
    end

    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_outputs
        assign req_out[i].valid = valid_out[i];
        assign {req_out[i].req_data, req_out[i].desc_data} = data_out[i];
        assign ready_out[i] = req_out[i].ready;
    end

    VX_stream_dispatch #(
        .NUM_INPUTS  (NUM_INPUTS),
        .NUM_OUTPUTS (NUM_OUTPUTS),
        .DATAW       (DATAW),
        .ARBITER     ("R"),
        .BUFFERED    (BUFFERED)
    ) dispatch (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_in),
        .data_in   (data_in),
        .ready_in  (ready_in),
        .valid_out (valid_out),
        .data_out  (data_out),
        .ready_out (ready_out)
    );

endmodule
