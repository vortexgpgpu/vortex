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

`TRACING_OFF
module VX_reset_relay #(
    parameter N          = 1,
    parameter MAX_FANOUT = 0
) (
    input wire          clk,
    input wire          reset,
    output wire [N-1:0] reset_o
);
    if (MAX_FANOUT >= 0 && N > (MAX_FANOUT + MAX_FANOUT/2)) begin
        localparam F = `UP(MAX_FANOUT);
        localparam R = N / F;
        `PRESERVE_NET reg [R-1:0] reset_r;
        for (genvar i = 0; i < R; ++i) begin
            always @(posedge clk) begin
                reset_r[i] <= reset;
            end
        end
        for (genvar i = 0; i < N; ++i) begin
            assign reset_o[i] = reset_r[i / F];
        end
    end else begin
        `UNUSED_VAR (clk)
        assign reset_o = {N{reset}};
    end

endmodule
`TRACING_ON
