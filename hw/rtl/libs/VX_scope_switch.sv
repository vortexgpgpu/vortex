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
module VX_scope_switch #(
    parameter N = 0
) (
    input wire  clk,
    input wire  reset,
    input wire  req_in,
    output wire [N-1:0] req_out,
    input wire  [N-1:0] rsp_in,
    output wire rsp_out
);
    if (N > 1) begin : g_switch
        reg req_out_r [N];
        reg rsp_out_r;

        always @(posedge clk) begin
            if (reset) begin
                for (integer i = 0; i < N; ++i) begin
                    req_out_r[i] <= 0;
                end
                rsp_out_r <= 0;
            end else begin
                for (integer i = 0; i < N; ++i) begin
                    req_out_r[i] <= req_in;
                end
                rsp_out_r <= 0;
                for (integer i = 0; i < N; ++i) begin
                    if (rsp_in[i])
                        rsp_out_r <= 1;
                end
            end
        end

        for (genvar i = 0; i < N; ++i) begin : g_req_out
            assign req_out[i] = req_out_r[i];
        end
        
        assign rsp_out = rsp_out_r;

    end else begin : g_passthru

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign req_out[0] = req_in;
        assign rsp_out = rsp_in[0];

    end

endmodule
`TRACING_ON
