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
module VX_bits_remove #(
    parameter N   = 2,
    parameter S   = 1,
    parameter POS = 0
) (
    input wire [N-1:0]    data_in,
    output wire [`UP(S)-1:0] sel_out,
    output wire [N-S-1:0] data_out
);
    `STATIC_ASSERT (((0 == S) || ((POS + S) <= N)), ("invalid parameter"))

    if (S == 0) begin : g_passthru
        assign sel_out = 0;
        assign data_out = data_in;
    end else if (POS == 0) begin : g_pos_0
        assign sel_out = data_in[0 +: S];
        assign data_out = data_in[N-1:S];
    end else if ((POS + S) == N) begin : g_pos_N
        assign sel_out = data_in[POS +: S];
        assign data_out = data_in[POS-1:0];
    end else begin : g_pos
        assign sel_out = data_in[POS +: S];
        assign data_out = {data_in[N-1:(POS+S)], data_in[POS-1:0]};
    end

    `UNUSED_VAR (data_in)

endmodule
`TRACING_ON
