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
    parameter MODEL = 0,
    parameter D = 1 << N
) (
    input wire [N-1:0] data_in,
    input wire [M-1:0] valid_in,
    output wire [D-1:0][M-1:0] data_out
);
    logic [D-1:0][M-1:0] shift;
    if (MODEL == 1) begin : g_model1
        always @(*) begin
            shift = '0;
            shift[data_in] = 1'b1;
        end
    end else begin : g_model0
        assign shift = (D*M)'(1'b1) << (data_in * M);
    end
    assign data_out = {D{valid_in}} & shift;

endmodule
`TRACING_ON
