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

// Fast one-hot encoder using parallel prefix computation
// Adapted from BaseJump STL: http://bjump.org/data_out.html

`TRACING_OFF
module VX_onehot_encoder #(
    parameter N       = 1,
    parameter REVERSE = 0,
    parameter MODEL   = 1,
    parameter LN      = `LOG2UP(N)
) (
    input wire [N-1:0]   data_in,
    output wire [LN-1:0] data_out,
    output wire          valid_out
);
    if (N == 1) begin : g_n1

        assign data_out  = 0;
        assign valid_out = data_in;

    end else if (N == 2) begin : g_n2

        assign data_out  = data_in[!REVERSE];
        assign valid_out = (| data_in);

    end else if (MODEL == 1) begin : g_model1
        localparam M = 1 << LN;

        reg [LN-1:0] addr_r;
        reg          valid_r;

        // Level lvl merges pairs of level lvl-1 groups of size 2^(lvl-1); each
        // level has its own slice of v/addr, local to one block, so no
        // variable feeds itself.
        always @(*) begin : g_scan
            reg [(LN+1)*M-1:0] v;
            reg [LN*M-1:0]     addr;
            reg [1:0]          vs;
            v    = '0;
            addr = '0;
            // base case, also handle padding for non-power of two inputs
            v[0 +: M] = REVERSE ? (M'(data_in) << (M - N)) : M'(data_in);
            for (integer lvl = 1; lvl <= LN; ++lvl) begin
                for (integer s = 0; s < (1 << (LN - lvl)); ++s) begin
                    vs = {v[(lvl-1)*M + s*(1 << lvl) + (1 << (lvl-1))], v[(lvl-1)*M + s*(1 << lvl)]};
                    v[lvl*M + s*(1 << lvl)] = (| vs);
                    addr[(lvl-1)*M + s*(1 << lvl) + (lvl-1)] = vs[!REVERSE];
                    for (integer b = 0; b < lvl-1; ++b) begin
                        addr[(lvl-1)*M + s*(1 << lvl) + b] = addr[(lvl-2)*M + s*(1 << lvl) + b]
                                                           | addr[(lvl-2)*M + s*(1 << lvl) + (1 << (lvl-1)) + b];
                    end
                end
            end
            addr_r  = addr[(LN-1)*M +: LN];
            valid_r = v[LN*M];
        end

        assign data_out  = addr_r;
        assign valid_out = valid_r;

    end else if (MODEL == 2 && REVERSE == 0) begin : g_model2

        for (genvar j = 0; j < LN; ++j) begin : g_data_out
            wire [N-1:0] mask;
            for (genvar i = 0; i < N; ++i) begin : g_mask
                assign mask[i] = i[j];
            end
            assign data_out[j] = | (mask & data_in);
        end

        assign valid_out = (| data_in);

    end else begin : g_model0

        reg [LN-1:0] index_w;

        if (REVERSE != 0) begin : g_msb
            always @(*) begin
                index_w = 'x;
                for (integer i = N-1; i >= 0; --i) begin
                    if (data_in[i]) begin
                        index_w = LN'(N-1-i);
                    end
                end
            end
        end else begin : g_lsb
            always @(*) begin
                index_w = 'x;
                for (integer i = 0; i < N; ++i) begin
                    if (data_in[i]) begin
                        index_w = LN'(i);
                    end
                end
            end
        end

        assign data_out  = index_w;
        assign valid_out = (| data_in);
    end

endmodule
`TRACING_ON
