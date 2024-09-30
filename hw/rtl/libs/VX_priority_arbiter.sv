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
module VX_priority_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input  wire [NUM_REQS-1:0]      requests,
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,
    output wire                     grant_valid
);
    if (NUM_REQS == 1) begin : g_passthru

        assign grant_index  = '0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin : g_encoder

        VX_priority_encoder #(
            .N (NUM_REQS)
        ) priority_encoder (
            .data_in    (requests),
            .index_out  (grant_index),
            .onehot_out (grant_onehot),
            .valid_out  (grant_valid)
        );

    end

endmodule
`TRACING_ON
