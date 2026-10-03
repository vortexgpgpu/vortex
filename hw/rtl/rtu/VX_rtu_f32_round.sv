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

// VX_rtu_f32_round — rounds the positive value (mag + sticky) * 2^exp to F32,
// nearest even, subnormals and overflow to infinity included (combinational).
// `sticky` stands for a nonzero remainder below mag's LSB; callers keep at
// least two bits of mag below the result's LSB whenever it is set. Returns the
// magnitude bits {exponent, fraction}; mag == 0 gives +0.

`include "VX_define.vh"

module VX_rtu_f32_round #(
    parameter WB = 44,      // mag width
    parameter EW = 11       // signed exponent width
) (
    input  wire [WB-1:0]        mag,
    input  wire signed [EW-1:0] exp,
    input  wire                 sticky,
    output wire [30:0]          result
);
    localparam IW = `CLOG2(WB);
    localparam BW = WB + 12;

    // index of mag's leading one
    reg [IW-1:0] msb;
    always @(*) begin
        msb = '0;
        for (integer i = 0; i < WB; ++i) begin
            if (mag[i]) msb = IW'(i);
        end
    end

    // the result LSB's exponent, clamped at the subnormal LSB 2^-149
    wire signed [EW+1:0] lsb_raw = (EW+2)'(exp) + (EW+2)'($signed({1'b0, msb})) - (EW+2)'(23);
    wire signed [EW+1:0] lsb     = (lsb_raw < -(EW+2)'(149)) ? -(EW+2)'(149) : lsb_raw;
    wire signed [EW+1:0] sh      = lsb - (EW+2)'(exp);

    // sh <= 0: exact, shifted up; sh > 0: drop sh bits, guard + sticky
    wire           exact = (sh <= 0);
    wire [IW-1:0]  ush   = exact ? IW'(-sh) : IW'(sh);
    wire [IW-1:0]  gi    = exact ? '0 : IW'(sh - (EW+2)'(1));
    wire [WB-1:0]  q     = mag >> ush;
    wire [WB-1:0]  low   = mag & ((WB'(1) << gi) - WB'(1));
    wire           g     = mag[gi];
    wire           st    = (low != '0) || sticky;
    wire [BW-1:0]  sig   = exact ? (BW'(mag) << ush)
                                 : (BW'(q) + BW'(g && (st || q[0])));

    // {biased exponent of the LSB position, significand}: a carry out of the
    // significand bumps the exponent, a subnormal reaching 2^23 becomes normal
    wire [BW-1:0] bits = (BW'(lsb + (EW+2)'(149)) << 23) + sig;
    assign result = (mag == '0) ? 31'd0
                  : (bits >= BW'(32'h7f800000)) ? 31'h7f800000
                  : bits[30:0];

endmodule
