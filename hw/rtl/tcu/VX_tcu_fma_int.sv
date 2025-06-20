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

`include "VX_define.vh"

module VX_tcu_fma_int #(
    parameter DATAW = 32
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,
    input  wire [2:0] fmt_s,   // 0=i32, 1=i16, 2=i8
    input  wire [2:0] fmt_d,   // 0=i32, 1=i16, 2=i8
    input  wire [DATAW-1:0] a, // packed i32, i16, i8
    input  wire [DATAW-1:0] b, // packed i32, i16, i8
    input  wire [DATAW-1:0] c, // i32
    output reg  [DATAW-1:0] y  // i32
);
    `UNUSED_VAR(reset);
    `UNUSED_VAR(fmt_d);

    function automatic [DATAW-1:0] sext16;
        input logic [15:0] in;
      begin
        sext16 = {{(DATAW-16){in[15]}}, in};
      end
    endfunction

    function automatic signed [DATAW-1:0] sext8;
        input logic [7:0] in;
      begin
        sext8 = $signed({{(DATAW-8){in[7]}}, in});
      end
    endfunction

    always @(posedge clk) begin
        if (enable) begin
            case (fmt_s)
            3'b000: begin // i32
                y <= $signed(a) * $signed(b) + $signed(c);
            end
            3'b001: begin // i16
                y <= sext16(a[15:0]) + sext16(b[15:0]) +
                     sext16(a[31:16]) + sext16(b[31:16]) + $signed(c);
            end
            3'b010: begin // i8
                y <= sext8(a[7:0]) * sext8(b[7:0]) +
                     sext8(a[15:8]) * sext8(b[15:8]) +
                     sext8(a[23:16]) * sext8(b[23:16]) +
                     sext8(a[31:24]) * sext8(b[31:24]) + $signed(c);
            end
            default:;
            endcase
        end
    end

endmodule
