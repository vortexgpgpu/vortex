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

`ifdef FPU_DPI

`include "VX_define.vh"

module VX_tcu_fma_dpi #(
    parameter DATAW = 32
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,
    input  wire [2:0] fmt_s,
    input  wire [2:0] fmt_d,
    input  wire [DATAW-1:0] a,
    input  wire [DATAW-1:0] b,
    input  wire [DATAW-1:0] c,
    output reg  [DATAW-1:0] y
);
    `UNUSED_VAR(reset);

    reg [1:0][4:0] fflags;
    reg [1:0][63:0] a_h, a_b;
    reg [1:0][63:0] b_h, b_b;
    reg [2:0][63:0] fmadd;
    reg [63:0] result;

    `UNUSED_VAR(fflags);
    `UNUSED_VAR(result);

    always @(*) begin
        case (fmt_s)
        3'b000: begin // fp32
            dpi_fmadd(enable, int'(0), {32'hffffffff,a}, {32'hffffffff,b}, {32'hffffffff,c}, 3'b0, fmadd[0], fflags[0]);
            result = fmadd[0];
        end
        3'b010: begin // fp16
            fmadd[0] = {32'hffffffff,c};
            for (integer i = 0; i < 2; i++) begin
                dpi_f2f(enable, int'(0), int'(2), {48'hffffffffffff,a[i*16 +: 16]}, a_h[i]);
                dpi_f2f(enable, int'(0), int'(2), {48'hffffffffffff,b[i*16 +: 16]}, b_h[i]);
                dpi_fmadd(enable, int'(0), a_h[i], b_h[i], fmadd[i], 3'b0, fmadd[i+1], fflags[i]);
            end
            case (fmt_d)
            3'b000: begin // fp32
                result = fmadd[2];
            end
            3'b010: begin // fp16
                dpi_f2f(enable, int'(2), int'(0), fmadd[2], result);
            end
            default:;
            endcase
        end
        3'b011: begin // bf16
            fmadd[0] = {32'hffffffff,c};
            for (integer i = 0; i < 2; i++) begin
                dpi_f2f(enable, int'(0), int'(3), {48'hffffffffffff,a[i*16 +: 16]}, a_b[i]);
                dpi_f2f(enable, int'(0), int'(3), {48'hffffffffffff,b[i*16 +: 16]}, b_b[i]);
                dpi_fmadd(enable, int'(0), a_b[i], b_b[i], fmadd[i], 3'b0, fmadd[i+1], fflags[i]);
            end
            case (fmt_d)
            3'b000: begin // f32
                result = fmadd[2];
            end
            3'b011: begin // bf16
                dpi_f2f(enable, int'(3), int'(0), fmadd[2], result);
            end
            default:;
            endcase
        end
        default:;
        endcase
    end

    VX_shift_register #(
        .DATAW (DATAW),
        .DEPTH (`LATENCY_FMA)
    ) shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (result[DATAW-1:0]),
        .data_out (y)
    );

endmodule

`endif
