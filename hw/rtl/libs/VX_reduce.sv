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
module VX_reduce #(
    parameter DATAW_IN   = 1,
    parameter DATAW_OUT  = DATAW_IN,
    parameter N          = 1,
    parameter `STRING OP = "+"
) (
    input wire [N-1:0][DATAW_IN-1:0] data_in,
    output wire [DATAW_OUT-1:0]      data_out
);
    if (N == 1) begin : g_passthru
        assign data_out = DATAW_OUT'(data_in[0]);
    end else begin : g_reduce
        localparam int N_A = N / 2;
        localparam int N_B = N - N_A;

        wire [N_A-1:0][DATAW_IN-1:0] in_A;
        wire [N_B-1:0][DATAW_IN-1:0] in_B;
        wire [DATAW_OUT-1:0] out_A, out_B;

        for (genvar i = 0; i < N_A; i++) begin : g_in_A
            assign in_A[i] = data_in[i];
        end

        for (genvar i = 0; i < N_B; i++) begin : g_in_B
            assign in_B[i] = data_in[N_A + i];
        end

        VX_reduce #(
            .DATAW_IN  (DATAW_IN),
            .DATAW_OUT (DATAW_OUT),
            .N  (N_A),
            .OP (OP)
        ) reduce_A (
            .data_in  (in_A),
            .data_out (out_A)
        );

        VX_reduce #(
            .DATAW_IN  (DATAW_IN),
            .DATAW_OUT (DATAW_OUT),
            .N  (N_B),
            .OP (OP)
        ) reduce_B (
            .data_in  (in_B),
            .data_out (out_B)
        );

        if (OP == "+") begin : g_plus
            assign data_out = out_A + out_B;
        end else if (OP == "^") begin : g_xor
            assign data_out = out_A ^ out_B;
        end else if (OP == "&") begin : g_and
            assign data_out = out_A & out_B;
        end else if (OP == "|") begin : g_or
            assign data_out = out_A | out_B;
        end else begin : g_error
            `ERROR(("invalid parameter"));
        end
    end

endmodule
`TRACING_ON
