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
module VX_bits_concat #(
    parameter L = 1,
    parameter R = 1
) (
    input wire [`UP(L)-1:0] left_in,
    input wire [`UP(R)-1:0] right_in,
    output wire [(L+R)-1:0] data_out
);
    if (L == 0) begin : g_right_only
        `UNUSED_VAR (left_in)
        assign data_out = right_in;
    end else if (R == 0) begin : g_left_only
        `UNUSED_VAR (right_in)
        assign data_out = left_in;
    end else begin : g_concat
        assign data_out = {left_in, right_in};
    end

endmodule
`TRACING_ON
