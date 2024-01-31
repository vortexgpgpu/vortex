// Copyright 2024 blaise
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
    parameter DATAW    = 1,
    parameter PASSTHRU = 0
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
    if (PASSTHRU != 0) begin
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        assign ready_in  = ready_out;
        assign valid_out = valid_in;        
        assign data_out  = data_in;
    end else begin
        wire stall = valid_out && ~ready_out;

        VX_pipe_register #(
            .DATAW	(1 + DATAW),
            .RESETW (1)
        ) pipe_register (
            .clk      (clk),
            .reset    (reset),
            .enable	  (~stall),
            .data_in  ({valid_in,  data_in}),
            .data_out ({valid_out, data_out})
        );

        assign ready_in = ~stall;
    end

endmodule
`TRACING_ON
