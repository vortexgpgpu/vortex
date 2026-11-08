// Copyright 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Pipelined elastic buffer: full-bandwidth throughput, one register per stage, registered output.
// Limitation: ready_in and ready_out are combinationally coupled.

`include "VX_platform.vh"

`TRACING_OFF
module VX_pipe_buffer #(
    parameter DATAW  = 1,
    parameter RESETW = 0,
    parameter DEPTH  = 1
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             valid_in,
    output wire             ready_in,
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out,
    input  wire             ready_out,
    output wire             valid_out
);
    if (DEPTH == 0) begin : g_passthru
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        assign ready_in  = ready_out;
        assign valid_out = valid_in;
        assign data_out  = data_in;
    end else begin : g_register
        // Stage outputs only: the input stays out of these vectors, so an
        // unsplit vector (Verilator -O0) does not join valid_in to valid_out.
        wire [DEPTH-1:0] valid_r;
        wire [DEPTH-1:0] ready;
        wire [DEPTH-1:0][DATAW-1:0] data_r;

        assign ready_in = ready[0];

        for (genvar i = 0; i < DEPTH; ++i) begin : g_pipe_regs
            wire             stage_valid = (i == 0) ? valid_in : valid_r[`MAX(i, 1)-1];
            wire [DATAW-1:0] stage_data  = (i == 0) ? data_in  : data_r[`MAX(i, 1)-1];
            // Stage i advances when any later stage has a hole or the output drains.
            assign ready[i] = ready_out || ~(& valid_r[DEPTH-1:i]);
            VX_pipe_register #(
                .DATAW  (1 + DATAW),
                .RESETW (1 + RESETW)
            ) pipe_register (
                .clk      (clk),
                .reset    (reset),
                .enable   (ready[i]),
                .data_in  ({stage_valid, stage_data}),
                .data_out ({valid_r[i], data_r[i]})
            );
        end

        assign valid_out = valid_r[DEPTH-1];
        assign data_out = data_r[DEPTH-1];
    end

endmodule
`TRACING_ON
