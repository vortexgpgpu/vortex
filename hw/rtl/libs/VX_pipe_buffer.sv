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

// A pipelined elastic buffer operates at full bandwidth where push can happen if the buffer is not empty but is going empty
// It has the following benefits:
// + Full-bandwidth throughput
// + use only one register for storage
// + data_out is fully registered
// It has the following limitations:
// + ready_in and ready_out are coupled

`include "VX_platform.vh"

`TRACING_OFF
module VX_pipe_buffer #(
    parameter DATAW = 1,
    parameter DEPTH = 1
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
    if (DEPTH == 0) begin
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        assign ready_in  = ready_out;
        assign valid_out = valid_in;
        assign data_out  = data_in;
    end else begin
        wire [DEPTH:0] valid;
    `IGNORE_UNOPTFLAT_BEGIN
        wire [DEPTH:0] ready;
    `IGNORE_UNOPTFLAT_END
        wire [DEPTH:0][DATAW-1:0] data;

        assign valid[0] = valid_in;
        assign data[0]  = data_in;
        assign ready_in = ready[0];

        for (genvar i = 0; i < DEPTH; ++i) begin
            assign ready[i] = (ready[i+1] || ~valid[i+1]);
            VX_pipe_register #(
                .DATAW  (1 + DATAW),
                .RESETW (1)
            ) pipe_register (
                .clk      (clk),
                .reset    (reset),
                .enable   (ready[i]),
                .data_in  ({valid[i], data[i]}),
                .data_out ({valid[i+1], data[i+1]})
            );
        end

        assign valid_out = valid[DEPTH];
        assign data_out = data[DEPTH];
        assign ready[DEPTH] = ready_out;

    end

endmodule
`TRACING_ON
