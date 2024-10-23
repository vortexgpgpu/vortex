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
module VX_demux #(
    parameter N = 0,
    parameter M = 1,
    parameter MODEL = 0,
    parameter D = 1 << N
) (
    input wire [`UP(N)-1:0] sel_in,
    input wire [M-1:0] data_in,
    output wire [D-1:0][M-1:0] data_out
);
    if (N != 0) begin : g_decoder
        logic [D-1:0][M-1:0] shift;
        if (MODEL == 1) begin : g_model1
            always @(*) begin
                shift = '0;
                shift[sel_in] = {M{1'b1}};
            end
        end else begin : g_model0
            assign shift = ((D*M)'({M{1'b1}})) << (sel_in * M);
        end
        assign data_out = {D{data_in}} & shift;
    end else begin : g_passthru
        `UNUSED_VAR (sel_in)
        assign data_out = data_in;
    end

endmodule
`TRACING_ON
