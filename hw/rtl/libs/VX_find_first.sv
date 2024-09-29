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
module VX_find_first #(
    parameter N       = 1,
    parameter DATAW   = 1,
    parameter REVERSE = 0
) (
    input  wire [N-1:0][DATAW-1:0] data_in,
    input  wire [N-1:0]            valid_in,
    output wire [DATAW-1:0]        data_out,
    output wire                    valid_out
);
    localparam LOGN = `CLOG2(N);
    localparam TL   = (1 << LOGN) - 1;
    localparam TN   = (1 << (LOGN+1)) - 1;

`IGNORE_UNOPTFLAT_BEGIN
    wire s_n [TN];
    wire [DATAW-1:0] d_n [TN];
`IGNORE_UNOPTFLAT_END

    for (genvar i = 0; i < N; ++i) begin : g_reverse
        assign s_n[TL+i] = REVERSE ? valid_in[N-1-i] : valid_in[i];
        assign d_n[TL+i] = REVERSE ? data_in[N-1-i] : data_in[i];
    end

    if (TL < (TN-N)) begin : g_fill
        for (genvar i = TL+N; i < TN; ++i) begin : g_i
            assign s_n[i] = 0;
            assign d_n[i] = '0;
        end
    end

    for (genvar j = 0; j < LOGN; ++j) begin : g_scan
        localparam I = 1 << j;
        for (genvar i = 0; i < I; ++i) begin : g_i
            localparam K = I+i-1;
            assign s_n[K] = s_n[2*K+1] | s_n[2*K+2];
            assign d_n[K] = s_n[2*K+1] ? d_n[2*K+1] : d_n[2*K+2];
        end
    end

    assign valid_out = s_n[0];
    assign data_out  = d_n[0];

endmodule
`TRACING_ON
