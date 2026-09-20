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
// dpi_float.vh is simulation-only; only the DPI backend below uses it.
`ifdef VX_CFG_TCU_TYPE_DPI
`include "dpi_float.vh"
`endif

`ifdef TCU_OP
// The exact-product datapath (VX_CFG_TCU_TYPE_TFR) uses VX_tcu_op_mul and
// VX_tcu_op_accu instead; this module's BHF/DPI-only parameter lists do not
// elaborate without one of those backends selected.
`ifndef VX_CFG_TCU_TYPE_TFR

module VX_tcu_feop import VX_tcu_pkg::*; #(
    parameter N            = 16,
    parameter FREC_LATENCY = 0,
    parameter FMUL_LATENCY = 2,
    parameter FRND_LATENCY = 1,
    parameter ID           = 0  // instance index, used in traces
) (
    input  wire clk,
    input  wire reset,

    input  wire enable,
    input  wire valid_in,
    input  wire [N-1:0] valid_in_bitmap, // Bitmap indicating which elements in the block are valid for processing

    input  wire [3:0] fmt_s,
    input  wire [3:0] fmt_d,

    input  wire [`VX_CFG_XLEN-1:0]        a_elem,
    input  wire [N-1:0][`VX_CFG_XLEN-1:0] b_row,
    
    output wire [N-1:0][`VX_CFG_XLEN-1:0] d_block // Output D block to the tcu_core
);
    localparam TOTAL_LATENCY = FREC_LATENCY + FMUL_LATENCY + FRND_LATENCY;
`ifdef VX_CFG_TCU_TYPE_BHF
    `UNUSED_PARAM (TOTAL_LATENCY)
`endif

    `UNUSED_PARAM (ID)
    `UNUSED_VAR (fmt_d)

    wire [15:0] a_16b = a_elem[15:0];

    reg  [63:0] feop_output [N];
    wire [N-1:0][31:0] feop_output_delayed;

    // multiplication stage
    for (genvar j = 0; j < N; j++) begin : g_prod

`ifdef VX_CFG_TCU_TYPE_DPI
        reg [63:0] a_f, b_f;
        reg [63:0] xprod;
        reg [4:0] fflags;

        `UNUSED_VAR({fflags, xprod[63:32]});
`endif

        wire [15:0] b_16b = b_row[j >> 1][5'(j[0] << 4) +: 16];
        wire [15:0] b_16b_gated = valid_in_bitmap[j] ? b_16b : 16'h0;
        wire [7:0]  b_8b = b_row[j >> 2][5'(j[1:0] << 3) +: 8];
        wire [7:0]  b_8b_gated = valid_in_bitmap[j] ? b_8b : 8'h0;
        wire signed [7:0]  a_i8 = $signed(a_elem[7:0]);
        wire signed [7:0]  b_i8 = valid_in_bitmap[j] ? $signed(b_8b) : 8'sd0;
        wire signed [31:0] prod_i32 = a_i8 * b_i8;

`ifdef VX_CFG_TCU_TYPE_BHF
        wire signed [31:0] prod_i32_delayed;
        wire [31:0] bhf_prod_fp16;
        wire [31:0] bhf_prod_fp32;
        wire [31:0] bhf_prod_fp8;
        VX_tcu_bhf_fmul #(
            .IN_EXPW      (5),    // fp16 exponent size
            .IN_SIGW      (10+1), // fp16 significand size (+1 for hidden bit)
            .OUT_EXPW     (8),    // fp32 exponent size
            .OUT_SIGW     (24),   // fp32 significand size
            .IN_REC       (0),    // input is IEEE754  (not in recoded format)
            .OUT_REC      (0),    // output is IEEE754 (not in recoded format)
            .REC_LATENCY  (FREC_LATENCY),
            .MUL_LATENCY  (FMUL_LATENCY),
            .RND_LATENCY  (FRND_LATENCY)
        ) fp16_mul (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .frm    (3'b000),
            .a      (a_16b),
            .b      (b_16b_gated),
            .y      (bhf_prod_fp16),
            `UNUSED_PIN(fflags)
        );

        VX_tcu_bhf_fmul #(
            .IN_EXPW      (4),    // fp8 (e4m3) exponent size
            .IN_SIGW      (3+1),  // fp8 (e4m3) significand size (+1 for hidden bit)
            .OUT_EXPW     (8),    // fp32 exponent size
            .OUT_SIGW     (24),   // fp32 significand size
            .IN_REC       (0),    // input is IEEE754  (not in recoded format)
            .OUT_REC      (0),    // output is IEEE754 (not in recoded format)
            .REC_LATENCY  (FREC_LATENCY),
            .MUL_LATENCY  (FMUL_LATENCY),
            .RND_LATENCY  (FRND_LATENCY)
        ) fp8_mul (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .frm    (3'b000),
            .a      (a_elem[7:0]),
            .b      (b_8b_gated),
            .y      (bhf_prod_fp8),
            `UNUSED_PIN(fflags)
        );

        wire [31:0] a_32b = a_elem[31:0];
        wire [31:0] b_32b = b_row[j][31:0];
        wire [31:0] b_32b_gated = valid_in_bitmap[j] ? b_32b : 32'h0;

        VX_tcu_bhf_fmul #(
            .IN_EXPW      (8),    // fp32 exponent size
            .IN_SIGW      (23+1), // fp32 significand size (+1 for hidden bit)
            .OUT_EXPW     (8),    // fp32 exponent size
            .OUT_SIGW     (24),   // fp32 significand size
            .IN_REC       (0),    // input is IEEE754  (not in recoded format)
            .OUT_REC      (0),    // output is IEEE754 (not in recoded format)
            .REC_LATENCY  (FREC_LATENCY),
            .MUL_LATENCY  (FMUL_LATENCY),
            .RND_LATENCY  (FRND_LATENCY)
        ) fp32_mul (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .frm    (3'b000),
            .a      (a_32b),
            .b      (b_32b_gated),
            .y      (bhf_prod_fp32),
            `UNUSED_PIN(fflags)
        );

        // BHF FP mul units are internally pipelined, but int8 multiply is combinational.
        // Delay int8 products to keep alignment with FEOP control timing.
        VX_pipe_register #(
            .DATAW  (32),
            .RESETW (32),
            .DEPTH  (TOTAL_LATENCY)
        ) pipe_imul (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (prod_i32),
            .data_out (prod_i32_delayed)
        );
`endif

        always @(*) begin
            feop_output[j] = '0;
`ifdef VX_CFG_TCU_TYPE_DPI
            // Default assignments avoid inferred latches in combinational logic.
            a_f    = {32'hffffffff, 32'h0};
            b_f    = {32'hffffffff, 32'h0};
            xprod  = 64'hffffffff00000000;
            fflags = '0;
`endif
            case (fmt_s)
            // Format ids are VX_tcu_pkg's, truncated to the 4-bit fmt field.
            4'(TCU_FP32_ID): begin // fp32
`ifdef VX_CFG_TCU_TYPE_BHF
                feop_output[j] = {32'hffffffff, bhf_prod_fp32};
`elsif VX_CFG_TCU_TYPE_DPI
                a_f = {32'hffffffff, a_elem};
                b_f = valid_in_bitmap[j] ? {32'hffffffff, b_row[j][31:0]} : {32'hffffffff, 32'h0};
                dpi_fmadd(enable, int'(0), a_f, b_f, xprod, 3'b0, feop_output[j], fflags);
`endif
            end
            4'(TCU_FP16_ID): begin // fp16
`ifdef VX_CFG_TCU_TYPE_BHF
                feop_output[j] = {32'hffffffff, bhf_prod_fp16};
`elsif VX_CFG_TCU_TYPE_DPI
                dpi_f2f(enable, int'(0), int'(2), {48'hffffffffffff, a_16b},       3'b0, a_f, fflags);
                dpi_f2f(enable, int'(0), int'(2), {48'hffffffffffff, b_16b_gated}, 3'b0, b_f, fflags);
                dpi_fmadd(enable, int'(0), a_f, b_f, xprod, 3'b0, feop_output[j], fflags);
`endif
            end
            4'(TCU_FP8_ID): begin // fp8
`ifdef VX_CFG_TCU_TYPE_BHF
                feop_output[j] = {32'hffffffff, bhf_prod_fp8};
`elsif VX_CFG_TCU_TYPE_DPI
                dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffffff, a_elem[7:0]}, 3'b0, a_f, fflags);
                dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffffff, b_8b_gated}, 3'b0, b_f, fflags);
                dpi_fmadd(enable, int'(0), a_f, b_f, xprod, 3'b0, feop_output[j], fflags);
`endif
            end
            4'(TCU_I8_ID): begin // int8
                // lower 32 bits are what pipe_mult forwards; upper 32 just sign-extend
`ifdef VX_CFG_TCU_TYPE_BHF
                feop_output[j] = {{32{prod_i32_delayed[31]}}, prod_i32_delayed};
`elsif VX_CFG_TCU_TYPE_DPI
                feop_output[j] = {{32{prod_i32[31]}}, prod_i32};
`endif
            end
            default: begin
                feop_output[j] = '0;
            end
            endcase
        end

        // Models FMUL latency for feop_output (BHF already pipelines internally)
        VX_pipe_register #(
            .DATAW  (32),
            .RESETW (32),
`ifdef VX_CFG_TCU_TYPE_BHF
            .DEPTH  (0)
`elsif VX_CFG_TCU_TYPE_DPI
            .DEPTH  (TOTAL_LATENCY)
`endif
        ) pipe_mult (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (feop_output[j][31:0]),
            .data_out (feop_output_delayed[j])
        );
    end

    always @(posedge clk) begin
        if (~reset && valid_in && enable) begin
            `TRACE(1, ("%t: [feop %0d]: Started\n", $time, ID))
        end
    end

    assign d_block = feop_output_delayed;

endmodule

`endif // !VX_CFG_TCU_TYPE_TFR
`endif // TCU_OP
