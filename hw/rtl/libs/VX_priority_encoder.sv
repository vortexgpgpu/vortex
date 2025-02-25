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
module VX_priority_encoder #(
    parameter N       = 1,
    parameter REVERSE = 0,
    parameter MODEL   = 1,
    parameter LN      = `LOG2UP(N)
) (
    input  wire [N-1:0]  data_in,
    output wire [N-1:0]  onehot_out,
    output wire [LN-1:0] index_out,
    output wire          valid_out
);
    wire [N-1:0] reversed;

    if (REVERSE != 0) begin : g_reverse
        for (genvar i = 0; i < N; ++i) begin : g_i
            assign reversed[N-i-1] = data_in[i];
        end
    end else begin : g_no_reverse
        assign reversed = data_in;
    end

    if (N == 1) begin : g_n1

        assign onehot_out = reversed;
        assign index_out  = '0;
        assign valid_out  = reversed;

    end else if (N == 2) begin : g_n2

        assign onehot_out = {reversed[1] && ~reversed[0], reversed[0]};
        assign index_out  = ~reversed[0];
        assign valid_out  = (| reversed);

    end else if (MODEL == 1) begin : g_model1

    `IGNORE_UNOPTFLAT_BEGIN
        wire [N-1:0] higher_pri_regs;
    `IGNORE_UNOPTFLAT_END

        assign higher_pri_regs[0] = 1'b0;
        for (genvar i = 1; i < N; ++i) begin : g_higher_pri_regs
            assign higher_pri_regs[i] = higher_pri_regs[i-1] | reversed[i-1];
        end
        assign onehot_out[N-1:0] = reversed[N-1:0] & ~higher_pri_regs[N-1:0];

        VX_lzc #(
            .N       (N),
            .REVERSE (1)
        ) lzc (
            .data_in   (reversed),
            .data_out  (index_out),
            .valid_out (valid_out)
        );

    end else if (MODEL == 2) begin : g_model2

        wire [N-1:0] scan_lo;

        VX_scan #(
            .N  (N),
            .OP ("|")
        ) scan (
            .data_in  (reversed),
            .data_out (scan_lo)
        );

        VX_lzc #(
            .N       (N),
            .REVERSE (1)
        ) lzc (
            .data_in  (reversed),
            .data_out (index_out),
            .valid_out(valid_out)
        );

        assign onehot_out = scan_lo & {(~scan_lo[N-2:0]), 1'b1};

    end else if (MODEL == 3) begin : g_model3

        assign onehot_out = reversed & -reversed;

        VX_lzc #(
            .N       (N),
            .REVERSE (1)
        ) lzc (
            .data_in   (reversed),
            .data_out  (index_out),
            .valid_out (valid_out)
        );

    end else begin : g_model0

        reg [LN-1:0] index_w;
        reg [N-1:0]  onehot_w;

        always @(*) begin
            index_w  = 'x;
            onehot_w = 'x;
            for (integer i = N-1; i >= 0; --i) begin
                if (reversed[i]) begin
                    index_w  = LN'(i);
                    onehot_w = N'(1) << i;
                end
            end
        end

        assign index_out  = index_w;
        assign onehot_out = onehot_w;
        assign valid_out  = (| reversed);

    end

endmodule
`TRACING_ON
