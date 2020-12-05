`include "VX_define.vh"

module VX_fp_noncomp #( 
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

    reg valid_in_r;
    reg [TAGW-1:0] tag_in_r;
    reg [`FPU_BITS-1:0] op_type_r;
    reg [`FRM_BITS-1:0] frm_r;

    reg [LANES-1:0][31:0]  dataa_r;
    reg [LANES-1:0][31:0]  datab_r;

    reg [LANES-1:0]       a_sign, b_sign;
    reg [LANES-1:0][7:0]  a_exponent;
    reg [LANES-1:0][22:0] a_mantissa;
    fp_type_t [LANES-1:0] a_type, b_type;
    reg [LANES-1:0] a_smaller, ab_equal;

    reg [LANES-1:0][31:0] fclass_mask;  // generate a 10-bit mask for integer reg
    reg [LANES-1:0][31:0] fminmax_res;  // result of fmin/fmax
    reg [LANES-1:0][31:0] fsgnj_res;    // result of sign injection
    reg [LANES-1:0][31:0] fcmp_res;     // result of comparison
    reg [LANES-1:0][ 4:0] fcmp_excp;    // exception of comparison

    wire stall = ~ready_out && valid_out;

    // Setup
    for (genvar i = 0; i < LANES; i++) begin
        wire            tmp_a_sign = dataa[i][31]; 
        wire [7:0]  tmp_a_exponent = dataa[i][30:23];
        wire [22:0] tmp_a_mantissa = dataa[i][22:0];

        wire            tmp_b_sign = datab[i][31]; 
        wire [7:0]  tmp_b_exponent = datab[i][30:23];
        wire [22:0] tmp_b_mantissa = datab[i][22:0];

        fp_type_t tmp_a_type, tmp_b_type;

        VX_fp_type fp_type_a (
            .exponent(tmp_a_exponent),
            .mantissa(tmp_a_mantissa),
            .o_type(tmp_a_type)
        );

        VX_fp_type fp_type_b (
            .exponent(tmp_b_exponent),
            .mantissa(tmp_b_mantissa),
            .o_type(tmp_b_type)
        );

        wire tmp_a_smaller = $signed(dataa[i]) < $signed(datab[i]);
        wire tmp_ab_equal  = (dataa[i] == datab[i]) | (tmp_a_type[4] & tmp_b_type[4]);

        VX_generic_register #(
            .N(1 + 1 + 8 + 23 + $bits(fp_type_t) + $bits(fp_type_t) + 1 + 1),
            .R(0)
        ) pipe_reg0 (
            .clk      (clk),
            .reset    (reset),
            .stall    (stall),
            .flush    (1'b0),
            .data_in  ({tmp_a_sign, tmp_b_sign, tmp_a_exponent, tmp_a_mantissa, tmp_a_type, tmp_b_type, tmp_a_smaller, tmp_ab_equal}),
            .data_out ({a_sign[i],  b_sign[i],  a_exponent[i],  a_mantissa[i],  a_type[i],  b_type[i],  a_smaller[i],  ab_equal[i]})
        );
    end  

    VX_generic_register #(
        .N(1 + TAGW + `FPU_BITS + `FRM_BITS + (2 * `NUM_THREADS * 32)),
        .R(1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .stall    (stall),
        .flush    (1'b0),
        .data_in  ({valid_in,   tag_in,   op_type,   frm,   dataa,   datab}),
        .data_out ({valid_in_r, tag_in_r, op_type_r, frm_r, dataa_r, datab_r})
    ); 

    // FCLASS
    for (genvar i = 0; i < LANES; i++) begin
        always @(*) begin 
            if (a_type[i].is_normal) begin
                fclass_mask[i] = a_sign[i] ? NEG_NORM : POS_NORM;
            end 
            else if (a_type[i].is_inf) begin
                fclass_mask[i] = a_sign[i] ? NEG_INF : POS_INF;
            end 
            else if (a_type[i].is_zero) begin
                fclass_mask[i] = a_sign[i] ? NEG_ZERO : POS_ZERO;
            end 
            else if (a_type[i].is_subnormal) begin
                fclass_mask[i] = a_sign[i] ? NEG_SUBNORM : POS_SUBNORM;
            end 
            else if (a_type[i].is_nan) begin
                fclass_mask[i] = {22'h0, a_type[i].is_quiet, a_type[i].is_signaling, 8'h0};
            end 
            else begin                     
                fclass_mask[i] = QUT_NAN;
            end
        end
    end

    // Min/Max
    for (genvar i = 0; i < LANES; i++) begin
        always @(*) begin
            if (a_type[i].is_nan && b_type[i].is_nan)
                fminmax_res[i] = {1'b0, 8'hff, 1'b1, 22'd0}; // canonical qNaN
            else if (a_type[i].is_nan) 
                fminmax_res[i] = datab_r[i];
            else if (b_type[i].is_nan) 
                fminmax_res[i] = dataa_r[i];
            else begin 
                case (frm_r) // use LSB to distinguish MIN and MAX
                    3: fminmax_res[i] = a_smaller[i] ? dataa_r[i] : datab_r[i];
                    4: fminmax_res[i] = a_smaller[i] ? datab_r[i] : dataa_r[i];
              default: fminmax_res[i] = 'x;  // don't care value
                endcase
            end
        end
    end

    // Sign Injection
    for (genvar i = 0; i < LANES; i++) begin
        always @(*) begin
            case (frm_r)
                0: fsgnj_res[i] = { b_sign[i], a_exponent[i], a_mantissa[i]};
                1: fsgnj_res[i] = {~b_sign[i], a_exponent[i], a_mantissa[i]};
                2: fsgnj_res[i] = { a_sign[i] ^ b_sign[i], a_exponent[i], a_mantissa[i]};
          default: fsgnj_res[i] = 'x;  // don't care value
            endcase
        end
    end

    // Comparison    
    for (genvar i = 0; i < LANES; i++) begin
        always @(*) begin
            case (frm_r)
                `FRM_RNE: begin
                    if (a_type[i].is_nan || b_type[i].is_nan) begin
                        fcmp_res[i]  = 32'h0;        // result is 0 when either operand is NaN
                        fcmp_excp[i] = {1'b1, 4'h0}; // raise NV flag when either operand is NaN
                    end
                    else begin
                        fcmp_res[i] = {31'h0, (a_smaller[i] | ab_equal[i])};
                        fcmp_excp[i] = 5'h0;
                    end
                end
                `FRM_RTZ: begin 
                    if (a_type[i].is_nan || b_type[i].is_nan) begin
                        fcmp_res[i]  = 32'h0;        // result is 0 when either operand is NaN
                        fcmp_excp[i] = {1'b1, 4'h0}; // raise NV flag when either operand is NaN
                    end
                    else begin
                        fcmp_res[i] = {31'h0, (a_smaller[i] & ~ab_equal[i])};
                        fcmp_excp[i] = 5'h0;
                    end                    
                end
                `FRM_RDN: begin
                    if (a_type[i].is_nan || b_type[i].is_nan) begin
                        fcmp_res[i]  = 32'h0;        // result is 0 when either operand is NaN
                        // FEQS only raise NV flag when either operand is signaling NaN
                        fcmp_excp[i] = {(a_type[i].is_signaling | b_type[i].is_signaling), 4'h0}; 
                    end
                    else begin
                        fcmp_res[i] = {31'h0, ab_equal[i]};
                        fcmp_excp[i] = 5'h0;
                    end
                end
                default: begin
                    fcmp_res[i]  = 'x;  // don't care value
                    fcmp_excp[i] = 5'h0;                        
                end
            endcase
        end
    end

    // outputs

    fflags_t [LANES-1:0] tmp_fflags;
    reg [LANES-1:0][31:0] tmp_result;

    for (genvar i = 0; i < LANES; i++) begin
        always @(*) begin
            case (op_type_r)
                `FPU_CLASS: begin
                    tmp_result[i] = fclass_mask[i];
                    {tmp_fflags[i].NV, tmp_fflags[i].DZ, tmp_fflags[i].OF, tmp_fflags[i].UF, tmp_fflags[i].NX} = 5'h0;
                end   
                `FPU_CMP: begin 
                    tmp_result[i] = fcmp_res[i];
                    {tmp_fflags[i].NV, tmp_fflags[i].DZ, tmp_fflags[i].OF, tmp_fflags[i].UF, tmp_fflags[i].NX} = fcmp_excp[i];
                end      
                //`FPU_MISC:
                default: begin
                    case (frm_r)
                        0,1,2:  begin
                            tmp_result[i] = fsgnj_res[i];
                            {tmp_fflags[i].NV, tmp_fflags[i].DZ, tmp_fflags[i].OF, tmp_fflags[i].UF, tmp_fflags[i].NX} = 5'h0;
                        end
                        3,4: begin
                            tmp_result[i] = fminmax_res[i];
                            {tmp_fflags[i].NV, tmp_fflags[i].DZ, tmp_fflags[i].OF, tmp_fflags[i].UF, tmp_fflags[i].NX} = {a_type[i][0] | b_type[i][0], 4'h0};    
                        end
                        //5,6,7: 
                        default: begin
                            tmp_result[i] = dataa[i];
                            {tmp_fflags[i].NV, tmp_fflags[i].DZ, tmp_fflags[i].OF, tmp_fflags[i].UF, tmp_fflags[i].NX} = 5'h0;    
                        end
                    endcase
                end    
            endcase
        end
    end

    wire tmp_has_fflags = ((op_type_r == `FPU_MISC) && (frm == 3 || frm == 4)) // MIN/MAX 
                       || (op_type_r == `FPU_CMP); // CMP

    VX_generic_register #(
        .N(1 + TAGW + (LANES * 32) + 1 + (LANES * `FFG_BITS)),
        .R(1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .stall    (stall),
        .flush    (1'b0),
        .data_in  ({valid_in_r, tag_in_r, tmp_result, tmp_has_fflags, tmp_fflags}),
        .data_out ({valid_out,  tag_out,  result,     has_fflags,     fflags})
    );

    assign ready_in = ~stall;

endmodule