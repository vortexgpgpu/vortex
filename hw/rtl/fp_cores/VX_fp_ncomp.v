`include "VX_define.vh"

/// Modified port of noncomp module from fpnew Libray 
/// reference: https://github.com/pulp-platform/fpnew

module VX_fp_ncomp #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,
    
    input wire [`FPU_BITS-1:0] op_type,
    input wire [`FRM_BITS-1:0] frm,

    input wire [LANES-1:0][31:0]  dataa,
    input wire [LANES-1:0][31:0]  datab,
    output wire [LANES-1:0][31:0] result, 

    output wire has_fflags,
    output fflags_t [LANES-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);  
    localparam  NEG_INF     = 32'h00000001,
                NEG_NORM    = 32'h00000002,
                NEG_SUBNORM = 32'h00000004,
                NEG_ZERO    = 32'h00000008,
                POS_ZERO    = 32'h00000010,
                POS_SUBNORM = 32'h00000020,
                POS_NORM    = 32'h00000040,
                POS_INF     = 32'h00000080,
                SIG_NAN     = 32'h00000100,
                QUT_NAN     = 32'h00000200;

    wire [LANES-1:0]        tmp_a_sign, tmp_b_sign;
    wire [LANES-1:0][7:0]   tmp_a_exponent, tmp_b_exponent;
    wire [LANES-1:0][22:0]  tmp_a_mantissa, tmp_b_mantissa;
    fp_type_t [LANES-1:0]   tmp_a_type, tmp_b_type;
    wire [LANES-1:0]        tmp_a_smaller, tmp_ab_equal;

    // Setup
    for (genvar i = 0; i < LANES; i++) begin
        assign     tmp_a_sign[i] = dataa[i][31]; 
        assign tmp_a_exponent[i] = dataa[i][30:23];
        assign tmp_a_mantissa[i] = dataa[i][22:0];

        assign     tmp_b_sign[i] = datab[i][31]; 
        assign tmp_b_exponent[i] = datab[i][30:23];
        assign tmp_b_mantissa[i] = datab[i][22:0];

        VX_fp_type fp_type_a (
            .exp_i  (tmp_a_exponent[i]),
            .man_i  (tmp_a_mantissa[i]),
            .type_o (tmp_a_type[i])
        );

        VX_fp_type fp_type_b (
            .exp_i  (tmp_b_exponent[i]),
            .man_i  (tmp_b_mantissa[i]),
            .type_o (tmp_b_type[i])
        );

        assign tmp_a_smaller[i] = $signed(dataa[i]) < $signed(datab[i]);
        assign tmp_ab_equal[i]  = (dataa[i] == datab[i]) | (tmp_a_type[i].is_zero & tmp_b_type[i].is_zero);
    end  

    // Pipeline stage0

    wire                    valid_in_s0;
    wire [TAGW-1:0]         tag_in_s0;
    wire [`FPU_BITS-1:0]    op_type_s0;
    wire [`FRM_BITS-1:0]    frm_s0;
    wire [LANES-1:0][31:0]  dataa_s0, datab_s0;
    wire [LANES-1:0]        a_sign_s0, b_sign_s0;
    wire [LANES-1:0][7:0]   a_exponent_s0;
    wire [LANES-1:0][22:0]  a_mantissa_s0;
    fp_type_t [LANES-1:0]   a_type_s0, b_type_s0;
    wire [LANES-1:0]        a_smaller_s0, ab_equal_s0;

    wire stall;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + `FPU_BITS + `FRM_BITS + LANES * (2 * 32 + 1 + 1 + 8 + 23 + 2 * $bits(fp_type_t) + 1 + 1)),
        .RESETW (1),
        .DEPTH  (0)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_in,    tag_in,    op_type,    frm,    dataa,    datab,    tmp_a_sign, tmp_b_sign, tmp_a_exponent, tmp_a_mantissa, tmp_a_type, tmp_b_type, tmp_a_smaller, tmp_ab_equal}),
        .data_out ({valid_in_s0, tag_in_s0, op_type_s0, frm_s0, dataa_s0, datab_s0, a_sign_s0,  b_sign_s0,  a_exponent_s0,  a_mantissa_s0,  a_type_s0,  b_type_s0,  a_smaller_s0,  ab_equal_s0})
    ); 

    // FCLASS
    reg [LANES-1:0][31:0] fclass_mask;  // generate a 10-bit mask for integer reg
    for (genvar i = 0; i < LANES; i++) begin
        always @(*) begin 
            if (a_type_s0[i].is_normal) begin
                fclass_mask[i] = a_sign_s0[i] ? NEG_NORM : POS_NORM;
            end 
            else if (a_type_s0[i].is_inf) begin
                fclass_mask[i] = a_sign_s0[i] ? NEG_INF : POS_INF;
            end 
            else if (a_type_s0[i].is_zero) begin
                fclass_mask[i] = a_sign_s0[i] ? NEG_ZERO : POS_ZERO;
            end 
            else if (a_type_s0[i].is_subnormal) begin
                fclass_mask[i] = a_sign_s0[i] ? NEG_SUBNORM : POS_SUBNORM;
            end 
            else if (a_type_s0[i].is_nan) begin
                fclass_mask[i] = {22'h0, a_type_s0[i].is_quiet, a_type_s0[i].is_signaling, 8'h0};
            end 
            else begin                     
                fclass_mask[i] = QUT_NAN;
            end
        end
    end

    // Min/Max    
    reg [LANES-1:0][31:0] fminmax_res;  // result of fmin/fmax
    for (genvar i = 0; i < LANES; i++) begin
        always @(*) begin
            if (a_type_s0[i].is_nan && b_type_s0[i].is_nan)
                fminmax_res[i] = {1'b0, 8'hff, 1'b1, 22'd0}; // canonical qNaN
            else if (a_type_s0[i].is_nan) 
                fminmax_res[i] = datab_s0[i];
            else if (b_type_s0[i].is_nan) 
                fminmax_res[i] = dataa_s0[i];
            else begin 
                case (frm_s0) // use LSB to distinguish MIN and MAX
                    3: fminmax_res[i] = a_smaller_s0[i] ? dataa_s0[i] : datab_s0[i];
                    4: fminmax_res[i] = a_smaller_s0[i] ? datab_s0[i] : dataa_s0[i];
              default: fminmax_res[i] = 'x;  // don't care value
                endcase
            end
        end
    end

    // Sign injection    
    reg [LANES-1:0][31:0] fsgnj_res;    // result of sign injection
    for (genvar i = 0; i < LANES; i++) begin
        always @(*) begin
            case (frm_s0)
                0: fsgnj_res[i] = { b_sign_s0[i], a_exponent_s0[i], a_mantissa_s0[i]};
                1: fsgnj_res[i] = {~b_sign_s0[i], a_exponent_s0[i], a_mantissa_s0[i]};
                2: fsgnj_res[i] = { a_sign_s0[i] ^ b_sign_s0[i], a_exponent_s0[i], a_mantissa_s0[i]};
          default: fsgnj_res[i] = 'x;  // don't care value
            endcase
        end
    end

    // Comparison    
    reg [LANES-1:0][31:0] fcmp_res;     // result of comparison
    fflags_t [LANES-1:0]   fcmp_fflags;  // comparison fflags
    for (genvar i = 0; i < LANES; i++) begin
        always @(*) begin
            case (frm_s0)
                `FRM_RNE: begin // LE
                    fcmp_fflags[i] = 5'h0;
                    if (a_type_s0[i].is_nan || b_type_s0[i].is_nan) begin
                        fcmp_res[i]       = 32'h0;
                        fcmp_fflags[i].NV = 1'b1;
                    end else begin
                        fcmp_res[i] = {31'h0, (a_smaller_s0[i] | ab_equal_s0[i])};
                    end
                end
                `FRM_RTZ: begin // LS
                    fcmp_fflags[i] = 5'h0;
                    if (a_type_s0[i].is_nan || b_type_s0[i].is_nan) begin
                        fcmp_res[i]       = 32'h0;
                        fcmp_fflags[i].NV = 1'b1;
                    end else begin
                        fcmp_res[i] = {31'h0, (a_smaller_s0[i] & ~ab_equal_s0[i])};
                    end                    
                end
                `FRM_RDN: begin // EQ
                    fcmp_fflags[i] = 5'h0;
                    if (a_type_s0[i].is_nan || b_type_s0[i].is_nan) begin
                        fcmp_res[i]       = 32'h0;
                        fcmp_fflags[i].NV = a_type_s0[i].is_signaling | b_type_s0[i].is_signaling; 
                    end else begin
                        fcmp_res[i] = {31'h0, ab_equal_s0[i]};
                    end
                end
                default: begin
                    fcmp_res[i]    = 'x;
                    fcmp_fflags[i] = 'x;                        
                end
            endcase
        end
    end

    // outputs

    reg [LANES-1:0][31:0] tmp_result;
    fflags_t [LANES-1:0] tmp_fflags;

    for (genvar i = 0; i < LANES; i++) begin
        always @(*) begin
            case (op_type_s0)
                `FPU_CLASS: begin
                    tmp_result[i] = fclass_mask[i];
                    tmp_fflags[i] = 'x;
                end   
                `FPU_CMP: begin 
                    tmp_result[i] = fcmp_res[i];
                    tmp_fflags[i] = fcmp_fflags[i];
                end      
                //`FPU_MISC:
                default: begin
                    case (frm_s0)
                        0,1,2: begin
                            tmp_result[i] = fsgnj_res[i];
                            tmp_fflags[i] = 'x;
                        end
                        3,4: begin
                            tmp_result[i] = fminmax_res[i];
                            tmp_fflags[i] = 0;
                            tmp_fflags[i].NV = a_type_s0[i].is_signaling | b_type_s0[i].is_signaling;
                        end
                        //5,6,7: MOVE
                        default: begin
                            tmp_result[i] = dataa[i];
                            tmp_fflags[i] = 'x;
                        end
                    endcase
                end    
            endcase
        end
    end

    wire has_fflags_s0 = ((op_type_s0 == `FPU_MISC) 
                       && (frm_s0 == 3             // MIN
                        || frm_s0 == 4))           // MAX 
                      || (op_type_s0 == `FPU_CMP); // CMP

    assign stall = ~ready_out && valid_out;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + (LANES * 32) + 1 + (LANES * `FFG_BITS)),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_in_s0, tag_in_s0, tmp_result, has_fflags_s0, tmp_fflags}),
        .data_out ({valid_out,   tag_out,   result,     has_fflags,    fflags})
    );

    assign ready_in = ~stall;

endmodule