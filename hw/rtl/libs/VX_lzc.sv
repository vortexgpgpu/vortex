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
module VX_lzc #(
    parameter N       = 2,
    parameter REVERSE = 0,  // 0 -> leading zero, 1 -> trailing zero,
    parameter LOGN    = `LOG2UP(N)
) (
    input  wire [N-1:0]    data_in,
    output wire [LOGN-1:0] data_out,
    output wire            valid_out
);
    if (N == 1) begin : g_passthru

        `UNUSED_PARAM (REVERSE)

        assign data_out  = '0;
        assign valid_out = data_in;

    end else begin : g_lzc

        wire [N-1:0][LOGN-1:0] indices;

        for (genvar i = 0; i < N; ++i) begin : g_indices
            assign indices[i] = REVERSE ? LOGN'(i) : LOGN'(N-1-i);
        end

        VX_find_first #(
            .N       (N),
            .DATAW   (LOGN),
            .REVERSE (!REVERSE)
        ) find_first (
            .data_in   (indices),
            .valid_in  (data_in),
            .data_out  (data_out),
            .valid_out (valid_out)
        );

    end

endmodule
`TRACING_ON
