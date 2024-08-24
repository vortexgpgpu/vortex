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

// Modified port of noncomp module from fpnew Libray
// reference: https://github.com/pulp-platform/fpnew

`include "VX_fpu_define.vh"

`ifdef FPU_DSP

module VX_fncp_unit import VX_fpu_pkg::*; #(
    parameter LATENCY  = 1,
    parameter EXP_BITS = 8,
    parameter MAN_BITS = 23,
    parameter OUT_REG  = 0
) (
    input wire clk,
    input wire reset,

    input wire enable,

    input wire [`INST_FPU_BITS-1:0] op_type,
    input wire [`INST_FRM_BITS-1:0] frm,

    input wire [31:0]  dataa,
    input wire [31:0]  datab,
    output wire [31:0] result,

    output wire [`FP_FLAGS_BITS-1:0] fflags
);
    localparam  NEG_INF     = 32'h00000001,
                NEG_NORM    = 32'h00000002,
                NEG_SUBNORM = 32'h00000004,
                NEG_ZERO    = 32'h00000008,
                POS_ZERO    = 32'h00000010,
                POS_SUBNORM = 32'h00000020,
                POS_NORM    = 32'h00000040,
                POS_INF     = 32'h00000080,
                //SIG_NAN   = 32'h00000100,
                QUT_NAN     = 32'h00000200;

    wire        a_sign, b_sign;
    wire [7:0]  a_exponent, b_exponent;
    wire [22:0] a_mantissa, b_mantissa;
    fclass_t    a_fclass, b_fclass;
    wire        a_smaller, ab_equal;

    // Setup
    assign     a_sign = dataa[31];
    assign a_exponent = dataa[30:23];
    assign a_mantissa = dataa[22:0];

    assign     b_sign = datab[31];
    assign b_exponent = datab[30:23];
    assign b_mantissa = datab[22:0];

    VX_fp_classifier #(
        .EXP_BITS (EXP_BITS),
        .MAN_BITS (MAN_BITS)
    ) fp_class_a (
        .exp_i  (a_exponent),
        .man_i  (a_mantissa),
        .clss_o (a_fclass)
    );

    VX_fp_classifier #(
        .EXP_BITS (EXP_BITS),
        .MAN_BITS (MAN_BITS)
    ) fp_class_b (
        .exp_i  (b_exponent),
        .man_i  (b_mantissa),
        .clss_o (b_fclass)
    );

    assign a_smaller = (dataa < datab) ^ (a_sign || b_sign);
    assign ab_equal  = (dataa == datab)
                    || (a_fclass.is_zero && b_fclass.is_zero); // +0 == -0

    // Pipeline stage0

    wire [3:0]   op_mod_s0;
    wire [31:0]  dataa_s0, datab_s0;
    wire         a_sign_s0, b_sign_s0;
    wire [7:0]   a_exponent_s0;
    wire [22:0]  a_mantissa_s0;
    fclass_t     a_fclass_s0, b_fclass_s0;
    wire         a_smaller_s0, ab_equal_s0;

    `UNUSED_VAR (b_fclass_s0)

    wire [3:0] op_mod = {(op_type == `INST_FPU_CMP), frm};

    VX_pipe_register #(
        .DATAW (4 + 2 * 32 + 1 + 1 + 8 + 23 + 2 * $bits(fclass_t) + 1 + 1),
        .DEPTH (LATENCY > 0)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({op_mod,    dataa,    datab,    a_sign,    b_sign,    a_exponent,    a_mantissa,    a_fclass,    b_fclass,    a_smaller,    ab_equal}),
        .data_out ({op_mod_s0, dataa_s0, datab_s0, a_sign_s0, b_sign_s0, a_exponent_s0, a_mantissa_s0, a_fclass_s0, b_fclass_s0, a_smaller_s0, ab_equal_s0})
    );

    // FCLASS
    reg [31:0] fclass_mask_s0;  // generate a 10-bit mask for integer reg
    always @(*) begin
        if (a_fclass_s0.is_normal) begin
            fclass_mask_s0 = a_sign_s0 ? NEG_NORM : POS_NORM;
        end
        else if (a_fclass_s0.is_inf) begin
            fclass_mask_s0 = a_sign_s0 ? NEG_INF : POS_INF;
        end
        else if (a_fclass_s0.is_zero) begin
            fclass_mask_s0 = a_sign_s0 ? NEG_ZERO : POS_ZERO;
        end
        else if (a_fclass_s0.is_subnormal) begin
            fclass_mask_s0 = a_sign_s0 ? NEG_SUBNORM : POS_SUBNORM;
        end
        else if (a_fclass_s0.is_nan) begin
            fclass_mask_s0 = {22'h0, a_fclass_s0.is_quiet, a_fclass_s0.is_signaling, 8'h0};
        end
        else begin
            fclass_mask_s0 = QUT_NAN;
        end
    end

    // Min/Max
    reg [31:0] fminmax_res_s0;
    always @(*) begin
        if (a_fclass_s0.is_nan && b_fclass_s0.is_nan)
            fminmax_res_s0 = {1'b0, 8'hff, 1'b1, 22'd0}; // canonical qNaN
        else if (a_fclass_s0.is_nan)
            fminmax_res_s0 = datab_s0;
        else if (b_fclass_s0.is_nan)
            fminmax_res_s0 = dataa_s0;
        else begin
            // FMIN, FMAX
            fminmax_res_s0 = (op_mod_s0[0] ^ a_smaller_s0) ? dataa_s0 : datab_s0;
        end
    end

    // Sign injection
    reg [31:0] fsgnj_res_s0;    // result of sign injection
    always @(*) begin
        case (op_mod_s0[1:0])
            0: fsgnj_res_s0 = { b_sign_s0, a_exponent_s0, a_mantissa_s0};
            1: fsgnj_res_s0 = {~b_sign_s0, a_exponent_s0, a_mantissa_s0};
        default: fsgnj_res_s0 = { a_sign_s0 ^ b_sign_s0, a_exponent_s0, a_mantissa_s0};
        endcase
    end

    // Comparison
    reg fcmp_res_s0;        // result of comparison
    reg fcmp_fflags_NV_s0;  // comparison fflags
    always @(*) begin
        case (op_mod_s0[1:0])
            0: begin // LE
                if (a_fclass_s0.is_nan || b_fclass_s0.is_nan) begin
                    fcmp_res_s0       = 0;
                    fcmp_fflags_NV_s0 = 1;
                end else begin
                    fcmp_res_s0       = (a_smaller_s0 | ab_equal_s0);
                    fcmp_fflags_NV_s0 = 0;
                end
            end
            1: begin // LT
                if (a_fclass_s0.is_nan || b_fclass_s0.is_nan) begin
                    fcmp_res_s0       = 0;
                    fcmp_fflags_NV_s0 = 1;
                end else begin
                    fcmp_res_s0       = (a_smaller_s0 & ~ab_equal_s0);
                    fcmp_fflags_NV_s0 = 0;
                end
            end
            2: begin // EQ
                if (a_fclass_s0.is_nan || b_fclass_s0.is_nan) begin
                    fcmp_res_s0       = 0;
                    fcmp_fflags_NV_s0 = a_fclass_s0.is_signaling | b_fclass_s0.is_signaling;
                end else begin
                    fcmp_res_s0       = ab_equal_s0;
                    fcmp_fflags_NV_s0 = 0;
                end
            end
            default: begin
                fcmp_res_s0       = 'x;
                fcmp_fflags_NV_s0 = 'x;
            end
        endcase
    end

    // outputs
    reg [31:0] result_s0;
    reg fflags_NV_s0;
    always @(*) begin
        case (op_mod_s0[2:0])
            0,1,2: begin
                // SGNJ, CMP
                result_s0 = op_mod_s0[3] ? 32'(fcmp_res_s0) : fsgnj_res_s0;
                fflags_NV_s0 = fcmp_fflags_NV_s0;
            end
            3: begin
                // CLASS
                result_s0 = fclass_mask_s0;
                fflags_NV_s0 = 0;
            end
            4,5: begin
                // FMV
                result_s0 = dataa_s0;
                fflags_NV_s0 = 0;
            end
            6,7: begin
                // MIN/MAX
                result_s0 = fminmax_res_s0;
                fflags_NV_s0 = a_fclass_s0.is_signaling | b_fclass_s0.is_signaling;
            end
        endcase
    end

    wire fflags_NV;

    VX_pipe_register #(
        .DATAW (32 + 1),
        .DEPTH (OUT_REG)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({result_s0, fflags_NV_s0}),
        .data_out ({result,    fflags_NV})
    );
                    // NV, DZ, OF, UF, NX
    assign fflags = {fflags_NV, 1'b0, 1'b0, 1'b0, 1'b0};

endmodule
`endif
