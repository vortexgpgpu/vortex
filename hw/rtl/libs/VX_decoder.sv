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

`include "VX_platform.vh"

// Fast encoder using parallel prefix computation
// Adapted from BaseJump STL: http://bjump.org/data_out.html

`TRACING_OFF
module VX_decoder #(
    parameter N = 1,
    parameter M = 1,
`ifdef VIVADO
    parameter MODEL = 1,
`else
    parameter MODEL = 0,
`endif
    parameter D = 1 << N
) (
    input wire [N-1:0] shift_in,
    input wire [M-1:0] data_in,
    output wire [D-1:0][M-1:0] data_out
);
    if (MODEL == 1) begin
        reg [D-1:0][M-1:0] data_out_w;
        always @(*) begin
            data_out_w = '0;
            data_out_w[shift_in] = data_in;
        end
        assign data_out = data_out_w;
    end else begin
        assign data_out = (D*M)'(data_in) << (shift_in * M);
    end

endmodule
`TRACING_ON
