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
module VX_onehot_shift #(
    parameter N = 1,
    parameter M = 1
) (
    input wire [N-1:0] data_in0,
    input wire [M-1:0] data_in1,
    output wire [N*M-1:0] data_out
);
    for (genvar i = 0; i < M; ++i) begin : g_i
        for (genvar j = 0; j < N; ++j) begin : g_j
            assign data_out[i*N + j] = data_in1[i] & data_in0[j];
        end
    end

endmodule
`TRACING_ON
