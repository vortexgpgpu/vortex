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

`include "VX_fpu_define.vh"

`ifdef FPU_DSP

/// Modified port of noncomp module from fpnew Libray 
/// reference: https://github.com/pulp-platform/fpnew

module VX_fpu_ncomp import VX_fpu_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter TAGW = 1
) (
    input wire clk,
    input wire reset,

    output wire ready_in,
    input wire  valid_in,

    input wire [NUM_LANES-1:0] lane_mask,

    input wire [TAGW-1:0] tag_in,

    input wire [`INST_FPU_BITS-1:0] op_type,
    input wire [`INST_FRM_BITS-1:0] frm,

    input wire [NUM_LANES-1:0][31:0]  dataa,
    input wire [NUM_LANES-1:0][31:0]  datab,
    output wire [NUM_LANES-1:0][31:0] result, 

    output wire has_fflags,
    output wire [`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);  
    localparam  EXP_BITS = 8;
    localparam  MAN_BITS = 23;
        
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

    wire [NUM_LANES-1:0]        a_sign, b_sign;
    wire [NUM_LANES-1:0][7:0]   a_exponent, b_exponent;
    wire [NUM_LANES-1:0][22:0]  a_mantissa, b_mantissa;
    fclass_t [NUM_LANES-1:0]    a_fclass, b_fclass;
    wire [NUM_LANES-1:0]        a_smaller, ab_equal;

    // Setup
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign     a_sign[i] = dataa[i][31]; 
        assign a_exponent[i] = dataa[i][30:23];
        assign a_mantissa[i] = dataa[i][22:0];

        assign     b_sign[i] = datab[i][31]; 
        assign b_exponent[i] = datab[i][30:23];
        assign b_mantissa[i] = datab[i][22:0];

        VX_fpu_class #( 
            .EXP_BITS (EXP_BITS),
            .MAN_BITS (MAN_BITS)
        ) fp_class_a (
            .exp_i  (a_exponent[i]),
            .man_i  (a_mantissa[i]),
            .clss_o (a_fclass[i])
        );

        VX_fpu_class #( 
            .EXP_BITS (EXP_BITS),
            .MAN_BITS (MAN_BITS)
        ) fp_class_b (
            .exp_i  (b_exponent[i]),
            .man_i  (b_mantissa[i]),
            .clss_o (b_fclass[i])
        );

        assign a_smaller[i] = (dataa[i] < datab[i]) ^ (a_sign[i] || b_sign[i]);
        assign ab_equal[i]  = (dataa[i] == datab[i]) 
                           || (a_fclass[i].is_zero && b_fclass[i].is_zero); // +0 == -0
    end  

    // Pipeline stage0

    wire                        valid_in_s0;
    wire [NUM_LANES-1:0]        lane_mask_s0;
    wire [TAGW-1:0]             tag_in_s0;
    wire [3:0]                  op_mod_s0;
    wire [NUM_LANES-1:0][31:0]  dataa_s0, datab_s0;
    wire [NUM_LANES-1:0]        a_sign_s0, b_sign_s0;
    wire [NUM_LANES-1:0][7:0]   a_exponent_s0;
    wire [NUM_LANES-1:0][22:0]  a_mantissa_s0;
    fclass_t [NUM_LANES-1:0]    a_fclass_s0, b_fclass_s0;
    wire [NUM_LANES-1:0]        a_smaller_s0, ab_equal_s0;

    wire stall;

    wire [3:0] op_mod = {(op_type == `INST_FPU_CMP), frm};

    VX_pipe_register #(
        .DATAW  (1 + NUM_LANES + TAGW + 4 + NUM_LANES * (2 * 32 + 1 + 1 + 8 + 23 + 2 * $bits(fclass_t) + 1 + 1)),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_in, lane_mask, tag_in, op_mod, dataa, datab, a_sign, b_sign, a_exponent, a_mantissa, a_fclass, b_fclass, a_smaller, ab_equal}),
        .data_out ({valid_in_s0, lane_mask_s0, tag_in_s0, op_mod_s0, dataa_s0, datab_s0, a_sign_s0, b_sign_s0, a_exponent_s0, a_mantissa_s0, a_fclass_s0, b_fclass_s0, a_smaller_s0, ab_equal_s0})
    ); 

    // FCLASS
    reg [NUM_LANES-1:0][31:0] fclass_mask_s0;  // generate a 10-bit mask for integer reg
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        always @(*) begin 
            if (a_fclass_s0[i].is_normal) begin
                fclass_mask_s0[i] = a_sign_s0[i] ? NEG_NORM : POS_NORM;
            end 
            else if (a_fclass_s0[i].is_inf) begin
                fclass_mask_s0[i] = a_sign_s0[i] ? NEG_INF : POS_INF;
            end 
            else if (a_fclass_s0[i].is_zero) begin
                fclass_mask_s0[i] = a_sign_s0[i] ? NEG_ZERO : POS_ZERO;
            end 
            else if (a_fclass_s0[i].is_subnormal) begin
                fclass_mask_s0[i] = a_sign_s0[i] ? NEG_SUBNORM : POS_SUBNORM;
            end 
            else if (a_fclass_s0[i].is_nan) begin
                fclass_mask_s0[i] = {22'h0, a_fclass_s0[i].is_quiet, a_fclass_s0[i].is_signaling, 8'h0};
            end 
            else begin                     
                fclass_mask_s0[i] = QUT_NAN;
            end
        end
    end

    // Min/Max    
    reg [NUM_LANES-1:0][31:0] fminmax_res_s0;
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        always @(*) begin
            if (a_fclass_s0[i].is_nan && b_fclass_s0[i].is_nan)
                fminmax_res_s0[i] = {1'b0, 8'hff, 1'b1, 22'd0}; // canonical qNaN
            else if (a_fclass_s0[i].is_nan) 
                fminmax_res_s0[i] = datab_s0[i];
            else if (b_fclass_s0[i].is_nan) 
                fminmax_res_s0[i] = dataa_s0[i];
            else begin 
                // FMIN, FMAX
                fminmax_res_s0[i] = (op_mod_s0[0] ^ a_smaller_s0[i]) ? dataa_s0[i] : datab_s0[i];
            end
        end
    end

    // Sign injection    
    reg [NUM_LANES-1:0][31:0] fsgnj_res_s0;    // result of sign injection
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        always @(*) begin
            case (op_mod_s0[1:0])
                0: fsgnj_res_s0[i] = { b_sign_s0[i], a_exponent_s0[i], a_mantissa_s0[i]};
                1: fsgnj_res_s0[i] = {~b_sign_s0[i], a_exponent_s0[i], a_mantissa_s0[i]};
          default: fsgnj_res_s0[i] = { a_sign_s0[i] ^ b_sign_s0[i], a_exponent_s0[i], a_mantissa_s0[i]};
            endcase
        end
    end

    // Comparison    
    reg [NUM_LANES-1:0] fcmp_res_s0;        // result of comparison
    reg [NUM_LANES-1:0] fcmp_fflags_NV_s0;  // comparison fflags
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        always @(*) begin
            case (op_mod_s0[1:0])
                0: begin // LE                    
                    if (a_fclass_s0[i].is_nan || b_fclass_s0[i].is_nan) begin
                        fcmp_res_s0[i]       = 0;
                        fcmp_fflags_NV_s0[i] = 1;
                    end else begin
                        fcmp_res_s0[i]       = (a_smaller_s0[i] | ab_equal_s0[i]);
                        fcmp_fflags_NV_s0[i] = 0;
                    end
                end
                1: begin // LT
                    if (a_fclass_s0[i].is_nan || b_fclass_s0[i].is_nan) begin
                        fcmp_res_s0[i]       = 0;
                        fcmp_fflags_NV_s0[i] = 1;
                    end else begin
                        fcmp_res_s0[i]       = (a_smaller_s0[i] & ~ab_equal_s0[i]);
                        fcmp_fflags_NV_s0[i] = 0;
                    end                    
                end
                2: begin // EQ
                    if (a_fclass_s0[i].is_nan || b_fclass_s0[i].is_nan) begin
                        fcmp_res_s0[i]       = 0;
                        fcmp_fflags_NV_s0[i] = a_fclass_s0[i].is_signaling | b_fclass_s0[i].is_signaling; 
                    end else begin
                        fcmp_res_s0[i]       = ab_equal_s0[i];
                        fcmp_fflags_NV_s0[i] = 0;
                    end
                end
                default: begin
                    fcmp_res_s0[i]       = 'x;
                    fcmp_fflags_NV_s0[i] = 'x;                        
                end
            endcase
        end
    end

    // outputs

    reg [NUM_LANES-1:0][31:0] result_s0;
    reg [NUM_LANES-1:0] fflags_NV_s0;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        always @(*) begin
            case (op_mod_s0[2:0])
                0,1,2: begin
                    // SGNJ, CMP
                    result_s0[i] = op_mod_s0[3] ? 32'(fcmp_res_s0[i]) : fsgnj_res_s0[i];
                    fflags_NV_s0[i] = fcmp_fflags_NV_s0[i];
                end
                3: begin
                    // CLASS
                    result_s0[i] = fclass_mask_s0[i];
                    fflags_NV_s0[i] = 'x;
                end
                4,5: begin
                    // FMV
                    result_s0[i] = dataa_s0[i];
                    fflags_NV_s0[i] = 'x;
                end                
                6,7: begin
                    // MIN/MAX
                    result_s0[i] = fminmax_res_s0[i];
                    fflags_NV_s0[i] = a_fclass_s0[i].is_signaling | b_fclass_s0[i].is_signaling;
                end
            endcase
        end
    end

    // only MIN/MAX and CMP return status flags
    wire has_fflags_s0 = (op_mod_s0[2:0] >= 6) || op_mod_s0[3];

    assign stall = ~ready_out && valid_out;

    wire fflags_NV;
    reg fflags_merged;

    always @(*) begin
        fflags_merged = 0;
        for (integer i = 0; i < NUM_LANES; ++i) begin
            if (lane_mask_s0[i]) begin
                fflags_merged |= fflags_NV_s0[i];
            end
        end
    end

    VX_pipe_register #(
        .DATAW  (1 + TAGW + (NUM_LANES * 32) + 1 + 1),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_in_s0, tag_in_s0, result_s0, has_fflags_s0, fflags_merged}),
        .data_out ({valid_out, tag_out, result, has_fflags, fflags_NV})
    );

    assign ready_in = ~stall;

                  // NV, DZ, OF, UF, NX
    assign fflags = {fflags_NV, 1'b0, 1'b0, 1'b0, 1'b0};

endmodule
`endif
