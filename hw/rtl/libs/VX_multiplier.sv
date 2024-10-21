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
module VX_multiplier #(
    parameter A_WIDTH = 1,
    parameter B_WIDTH = A_WIDTH,
    parameter R_WIDTH = A_WIDTH + B_WIDTH,
    parameter SIGNED  = 0,
    parameter LATENCY = 0
) (
    input wire clk,
    input wire enable,
    input wire [A_WIDTH-1:0]  dataa,
    input wire [B_WIDTH-1:0]  datab,
    output wire [R_WIDTH-1:0] result
);
    wire [R_WIDTH-1:0] prod_w;

    if (SIGNED != 0) begin : g_prod_s
        assign prod_w = R_WIDTH'($signed(dataa) * $signed(datab));
    end else begin : g_prod_u
        assign prod_w = R_WIDTH'(dataa * datab);
    end

    if (LATENCY == 0) begin : g_passthru
        assign result = prod_w;
    end else begin : g_latency
        reg [LATENCY-1:0][R_WIDTH-1:0] prod_r;
        always @(posedge clk) begin
            if (enable) begin
                prod_r[0] <= prod_w;
                for (integer i = 1; i < LATENCY; ++i) begin
                    prod_r[i] <= prod_r[i-1];
                end
            end
        end
        assign result = prod_r[LATENCY-1];
    end

endmodule
`TRACING_ON
