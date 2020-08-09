`include "VX_define.vh"

module VX_fp_noncomp (
	input wire clk,
	input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [`ISTAG_BITS-1:0] tag_in,
	
    input wire [`FPU_BITS-1:0] op,
    input wire [`FRM_BITS-1:0] frm,

    input wire [`NUM_THREADS-1:0][31:0]  dataa,
    input wire [`NUM_THREADS-1:0][31:0]  datab,
    output wire [`NUM_THREADS-1:0][31:0] result, 

    output wire has_fflags,
    output fflags_t [`NUM_THREADS-1:0] fflags,

    output wire [`ISTAG_BITS-1:0] tag_out,

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

    wire [`NUM_THREADS-1:0]       a_sign, b_sign;
    wire [`NUM_THREADS-1:0][7:0]  a_exponent, b_exponent;
    wire [`NUM_THREADS-1:0][22:0] a_mantissa, b_mantissa;
    fp_type_t [`NUM_THREADS-1:0]  a_type, b_type;

    wire [`NUM_THREADS-1:0] a_smaller, ab_equal;

    reg [`NUM_THREADS-1:0][31:0] fclass_mask;  // generate a 10-bit mask for integer reg
    reg [`NUM_THREADS-1:0][31:0] fminmax_res;  // result of fmin/fmax
    reg [`NUM_THREADS-1:0][31:0] fsgnj_res;    // result of sign injection
    reg [`NUM_THREADS-1:0][31:0] fcmp_res;     // result of comparison
    reg [`NUM_THREADS-1:0][ 4:0] fcmp_excp;    // exception of comparison

    genvar i;
    
    // Setup
    for (i = 0; i < `NUM_THREADS; i++) begin
        assign a_sign[i]     = dataa[i][31]; 
        assign a_exponent[i] = dataa[i][30:23];
        assign a_mantissa[i] = dataa[i][22:0];

        assign b_sign[i]     = datab[i][31]; 
        assign b_exponent[i] = datab[i][30:23];
        assign b_mantissa[i] = datab[i][22:0];

        assign a_smaller[i]  = (dataa[i] < datab[i]) ^ (a_sign[i] || b_sign[i]);
        assign ab_equal[i]   = (dataa[i] == datab[i]) | (a_type[i][4] & b_type[i][4]);

        VX_fp_type fp_type_a (
            .exponent(a_exponent[i]),
            .mantissa(a_mantissa[i]),
            .o_type(a_type[i])
        );

        VX_fp_type fp_type_b (
            .exponent(b_exponent[i]),
            .mantissa(b_mantissa[i]),
            .o_type(b_type[i])
        );
    end   

    // FCLASS
    for (i = 0; i < `NUM_THREADS; i++) begin
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
    for (i = 0; i < `NUM_THREADS; i++) begin
        always @(*) begin
            if (a_type[i].is_nan && b_type[i].is_nan)
                fminmax_res[i] = {1'b0, 8'hff, 1'b1, 22'd0}; // canonical qNaN
            else if (a_type[i].is_nan) 
                fminmax_res[i] = datab[i];
            else if (b_type[i].is_nan) 
                fminmax_res[i] = dataa[i];
            else begin 
                case (op) // use LSB to distinguish MIN and MAX
                    `FPU_MIN: fminmax_res[i] = a_smaller[i] ? dataa[i] : datab[i];
                    `FPU_MAX: fminmax_res[i] = a_smaller[i] ? datab[i] : dataa[i];
                    default:  fminmax_res[i] = 32'hdeadbeaf;  // don't care value
                endcase
            end
        end
    end

    // Sign Injection
    for (i = 0; i < `NUM_THREADS; i++) begin
        always @(*) begin
            case (op)
                `FPU_SGNJ:  fsgnj_res[i] = { b_sign[i], a_exponent[i], a_mantissa[i]};
                `FPU_SGNJN: fsgnj_res[i] = {~b_sign[i], a_exponent[i], a_mantissa[i]};
                `FPU_SGNJX: fsgnj_res[i] = { a_sign[i] ^ b_sign[i], a_exponent[i], a_mantissa[i]};
                default: fsgnj_res[i] = 32'hdeadbeaf;  // don't care value
            endcase
        end
    end

    // Comparison    
    for (i = 0; i < `NUM_THREADS; i++) begin
        always @(*) begin
            case (frm)
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
                        // ** FEQS only raise NV flag when either operand is signaling NaN
                        fcmp_excp[i] = {(a_type[i].is_signaling | b_type[i].is_signaling), 4'h0};
                    end
                    else begin
                        fcmp_res[i] = {31'h0, ab_equal[i]};
                        fcmp_excp[i] = 5'h0;
                    end
                end
                default: begin
                    fcmp_res[i] = 32'hdeadbeaf;  // don't care value
                    fcmp_excp[i] = 5'h0;                        
                end
            endcase
        end
    end

    // outputs

    reg tmp_valid;
    reg tmp_has_fflags;
    fflags_t [`NUM_THREADS-1:0] tmp_fflags;
    reg [`NUM_THREADS-1:0][31:0] tmp_result;

    always @(*) begin        
        case (op)
            `FPU_SGNJ:  tmp_has_fflags = 0;
            `FPU_SGNJN: tmp_has_fflags = 0;
            `FPU_SGNJX: tmp_has_fflags = 0;
            `FPU_MVXW:  tmp_has_fflags = 0;
            `FPU_MVWX:  tmp_has_fflags = 0;
            `FPU_CLASS: tmp_has_fflags = 0;
            default:    tmp_has_fflags = 1;
        endcase
    end   

    for (i = 0; i < `NUM_THREADS; i++) begin
        always @(*) begin
            tmp_valid = 1'b1;
            case (op)
                `FPU_CLASS: begin
                    tmp_result[i] = fclass_mask[i];
                    {tmp_fflags[i].NV, tmp_fflags[i].DZ, tmp_fflags[i].OF, tmp_fflags[i].UF, tmp_fflags[i].NX} = 5'h0;
                end                
                `FPU_MVXW,`FPU_MVWX: begin                    
                    tmp_result[i] = dataa[i];
                    {tmp_fflags[i].NV, tmp_fflags[i].DZ, tmp_fflags[i].OF, tmp_fflags[i].UF, tmp_fflags[i].NX} = 5'h0;
                end
                `FPU_MIN,`FPU_MAX: begin                     
                    tmp_result[i] = fminmax_res[i];
                    {tmp_fflags[i].NV, tmp_fflags[i].DZ, tmp_fflags[i].OF, tmp_fflags[i].UF, tmp_fflags[i].NX} = {a_type[i][0] | b_type[i][0], 4'h0};
                end
                `FPU_SGNJ,`FPU_SGNJN,`FPU_SGNJX: begin
                    tmp_result[i] = fsgnj_res[i];
                    {tmp_fflags[i].NV, tmp_fflags[i].DZ, tmp_fflags[i].OF, tmp_fflags[i].UF, tmp_fflags[i].NX} = 5'h0;
                end
                `FPU_CMP: begin 
                    tmp_result[i] = fcmp_res[i];
                    {tmp_fflags[i].NV, tmp_fflags[i].DZ, tmp_fflags[i].OF, tmp_fflags[i].UF, tmp_fflags[i].NX} = fcmp_excp[i];
                end                
                default: begin                      
                    tmp_result[i] = 32'hdeadbeaf;
                    {tmp_fflags[i].NV, tmp_fflags[i].DZ, tmp_fflags[i].OF, tmp_fflags[i].UF, tmp_fflags[i].NX} = 5'h0;
                    tmp_valid = 1'b0;
                end
            endcase
        end
    end

    wire stall = ~ready_out && valid_out;
    assign ready_in = ~stall;

    VX_generic_register #(
        .N(1 + `ISTAG_BITS + (`NUM_THREADS * 32) + 1 + (`NUM_THREADS * `FFG_BITS))
    ) nc_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (1'b0),
        .in    ({tmp_valid, tag_in,  tmp_result, tmp_has_fflags, tmp_fflags}),
        .out   ({valid_out, tag_out, result,     has_fflags,     fflags})
    );

endmodule