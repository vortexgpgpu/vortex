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
    parameter REVERSE = 0 // 0 -> first valid, 1 -> last valid
) (
    input  wire [N-1:0][DATAW-1:0] data_in,
    input  wire [N-1:0]            valid_in,
    output wire [DATAW-1:0]        data_out,
    output wire                    valid_out
);
    localparam LOGN = `CLOG2(N);
    localparam TL   = (1 << LOGN) - 1;
    localparam TN   = (1 << (LOGN+1)) - 1;

    reg             valid_r;
    reg [DATAW-1:0] data_r;

    // Binary tree: node k reduces children 2k+1 (first) and 2k+2; leaves start at TL.
    // The tree is local to one block so no node variable feeds itself.
    always @(*) begin : g_scan
        reg [TN-1:0]       s_n;
        reg [TN*DATAW-1:0] d_n;
        s_n = '0;
        d_n = '0;
        for (integer i = 0; i < N; ++i) begin
            s_n[TL+i] = REVERSE ? valid_in[N-1-i] : valid_in[i];
            d_n[(TL+i)*DATAW +: DATAW] = REVERSE ? data_in[N-1-i] : data_in[i];
        end
        for (integer k = TL-1; k >= 0; --k) begin
            s_n[k] = s_n[2*k+2] | s_n[2*k+1];
            d_n[k*DATAW +: DATAW] = s_n[2*k+1] ? d_n[(2*k+1)*DATAW +: DATAW] : d_n[(2*k+2)*DATAW +: DATAW];
        end
        valid_r = s_n[0];
        data_r  = d_n[0 +: DATAW];
    end

    assign valid_out = valid_r;
    assign data_out  = data_r;

endmodule
`TRACING_ON
