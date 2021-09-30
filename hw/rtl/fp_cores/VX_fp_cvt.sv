`include "VX_fpu_define.vh"

/// Modified port of cast module from fpnew Libray 
/// reference: https://github.com/pulp-platform/fpnew

module VX_fp_cvt #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire [`INST_FRM_BITS-1:0] frm,

    input wire is_itof,
    input wire is_signed,

    input wire [LANES-1:0][31:0]  dataa,
    output wire [LANES-1:0][31:0] result, 

    output wire has_fflags,
    output fflags_t [LANES-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);   
    // Constants
 
    localparam MAN_BITS = 23;
    localparam EXP_BITS = 8;
    localparam EXP_BIAS = 2**(EXP_BITS-1)-1;
    
    // Use 32-bit integer
    localparam MAX_INT_WIDTH = 32;

    // The internal mantissa includes normal bit or an entire integer
    localparam INT_MAN_WIDTH = `MAX(MAN_BITS + 1, MAX_INT_WIDTH);

    // The lower 2p+3 bits of the internal FMA result will be needed for leading-zero detection
    localparam LZC_RESULT_WIDTH = $clog2(INT_MAN_WIDTH);

    // The internal exponent must be able to represent the smallest denormal input value as signed
    // or the number of bits in an integer
    localparam INT_EXP_WIDTH = `MAX($clog2(MAX_INT_WIDTH), `MAX(EXP_BITS, $clog2(EXP_BIAS + MAN_BITS))) + 1;

    // shift amount for denormalization
    localparam SHAMT_BITS = $clog2(INT_MAN_WIDTH+1);

    localparam FMT_SHIFT_COMPENSATION = INT_MAN_WIDTH - 1 - MAN_BITS;
    localparam NUM_FP_STICKY  = 2 * INT_MAN_WIDTH - MAN_BITS - 1;   // removed mantissa, 1. and R
    localparam NUM_INT_STICKY = 2 * INT_MAN_WIDTH - MAX_INT_WIDTH;  // removed int and R
    
    // Input processing
    
    fp_class_t [LANES-1:0] fp_clss;
      
    for (genvar i = 0; i < LANES; ++i) begin
        VX_fp_class #( 
            .EXP_BITS (EXP_BITS),
            .MAN_BITS (MAN_BITS)
        ) fp_class (
            .exp_i  (dataa[i][30:23]),
            .man_i  (dataa[i][22:0]),
            .clss_o (fp_clss[i])
        );
    end

    wire [LANES-1:0][INT_MAN_WIDTH-1:0] encoded_mant; // input mantissa with implicit bit    
    wire [LANES-1:0][INT_EXP_WIDTH-1:0] fmt_exponent;    
    wire [LANES-1:0]                    input_sign;
    
    for (genvar i = 0; i < LANES; ++i) begin
    `IGNORE_WARNINGS_BEGIN
        wire [INT_MAN_WIDTH-1:0] int_mantissa;
        wire [INT_MAN_WIDTH-1:0] fmt_mantissa;
        wire fmt_sign       = dataa[i][31];
        wire int_sign       = dataa[i][31] & is_signed;
        assign int_mantissa = int_sign ? (-dataa[i]) : dataa[i];
        assign fmt_mantissa = INT_MAN_WIDTH'({fp_clss[i].is_normal, dataa[i][MAN_BITS-1:0]});
        assign fmt_exponent[i] = {1'b0, dataa[i][MAN_BITS +: EXP_BITS]} +
                                 {1'b0, fp_clss[i].is_subnormal};
        assign encoded_mant[i] = is_itof ? int_mantissa : fmt_mantissa;
        assign input_sign[i]   = is_itof ? int_sign : fmt_sign;
    `IGNORE_WARNINGS_END
    end

    // Pipeline stage0
    
    wire                    valid_in_s0;
    wire [TAGW-1:0]         tag_in_s0;
    wire                    is_itof_s0;
    wire                    unsigned_s0;
    wire [2:0]              rnd_mode_s0;
    fp_class_t [LANES-1:0]  fp_clss_s0;
    wire [LANES-1:0]        input_sign_s0;
    wire [LANES-1:0][INT_EXP_WIDTH-1:0] fmt_exponent_s0;
    wire [LANES-1:0][INT_MAN_WIDTH-1:0] encoded_mant_s0;

    wire stall;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + 1 + `INST_FRM_BITS + 1 + LANES * ($bits(fp_class_t) + 1 + INT_EXP_WIDTH + INT_MAN_WIDTH)),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in,    tag_in,    is_itof,    !is_signed,  frm,         fp_clss,    input_sign,    fmt_exponent,    encoded_mant}),
        .data_out ({valid_in_s0, tag_in_s0, is_itof_s0, unsigned_s0, rnd_mode_s0, fp_clss_s0, input_sign_s0, fmt_exponent_s0, encoded_mant_s0})
    );
    
    // Normalization

    wire [LANES-1:0][LZC_RESULT_WIDTH-1:0] renorm_shamt_s0; // renormalization shift amount
    wire [LANES-1:0] mant_is_zero_s0;                       // for integer zeroes

    for (genvar i = 0; i < LANES; ++i) begin
        wire mant_is_nonzero;
        VX_lzc #(
            .N    (INT_MAN_WIDTH),
            .MODE (1)
        ) lzc (
            .in_i    (encoded_mant_s0[i]),
            .cnt_o   (renorm_shamt_s0[i]),
            .valid_o (mant_is_nonzero)
        );
        assign mant_is_zero_s0[i] = ~mant_is_nonzero;  
    end

    wire [LANES-1:0][INT_MAN_WIDTH-1:0] input_mant_s0;      // normalized input mantissa    
    wire [LANES-1:0][INT_EXP_WIDTH-1:0] input_exp_s0;       // unbiased true exponent
    
    for (genvar i = 0; i < LANES; ++i) begin
    `IGNORE_WARNINGS_BEGIN
       // Realign input mantissa, append zeroes if destination is wider
        assign input_mant_s0[i] = encoded_mant_s0[i] << renorm_shamt_s0[i];

        // Unbias exponent and compensate for shift
        wire [INT_EXP_WIDTH-1:0] fp_input_exp = fmt_exponent_s0[i] + (FMT_SHIFT_COMPENSATION - EXP_BIAS) - {1'b0, renorm_shamt_s0[i]};                                 
        wire [INT_EXP_WIDTH-1:0] int_input_exp = (INT_MAN_WIDTH-1) - {1'b0, renorm_shamt_s0[i]};

        assign input_exp_s0[i] = is_itof_s0 ? int_input_exp : fp_input_exp;
    `IGNORE_WARNINGS_END
    end

    // Pipeline stage1

    wire                    valid_in_s1;
    wire [TAGW-1:0]         tag_in_s1;
    wire                    is_itof_s1;
    wire                    unsigned_s1;
    wire [2:0]              rnd_mode_s1;
    fp_class_t [LANES-1:0]  fp_clss_s1;
    wire [LANES-1:0]        input_sign_s1;
    wire [LANES-1:0]        mant_is_zero_s1;
    wire [LANES-1:0][INT_MAN_WIDTH-1:0] input_mant_s1;
    wire [LANES-1:0][INT_EXP_WIDTH-1:0] input_exp_s1;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + 1 + `INST_FRM_BITS + 1 + LANES * ($bits(fp_class_t) + 1 + 1 + INT_MAN_WIDTH + INT_EXP_WIDTH)),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in_s0, tag_in_s0, is_itof_s0, unsigned_s0, rnd_mode_s0, fp_clss_s0, input_sign_s0, mant_is_zero_s0, input_mant_s0, input_exp_s0}),
        .data_out ({valid_in_s1, tag_in_s1, is_itof_s1, unsigned_s1, rnd_mode_s1, fp_clss_s1, input_sign_s1, mant_is_zero_s1, input_mant_s1, input_exp_s1})
    );

    // Perform adjustments to mantissa and exponent

    wire [LANES-1:0][2*INT_MAN_WIDTH:0] destination_mant_s1;
    wire [LANES-1:0][INT_EXP_WIDTH-1:0] final_exp_s1;
    wire [LANES-1:0]                    of_before_round_s1;

    for (genvar i = 0; i < LANES; ++i) begin
        reg [2*INT_MAN_WIDTH:0] preshift_mant;      // mantissa before final shift                
        reg [SHAMT_BITS-1:0]    denorm_shamt;       // shift amount for denormalization
        reg [INT_EXP_WIDTH-1:0] final_exp;          // after eventual adjustments
        reg                     of_before_round;

        always @(*) begin           
        `IGNORE_WARNINGS_BEGIN     
            // Default assignment
            final_exp       = input_exp_s1[i] + EXP_BIAS; // take exponent as is, only look at lower bits
            preshift_mant   = {input_mant_s1[i], 33'b0};  // Place mantissa to the left of the shifter
            denorm_shamt    = 0;      // right of mantissa
            of_before_round = 1'b0;

            // Handle INT casts
            if (is_itof_s1) begin                   
                if ($signed(input_exp_s1[i]) >= $signed(2**EXP_BITS-1-EXP_BIAS)) begin
                    // Overflow or infinities (for proper rounding)
                    final_exp     = (2**EXP_BITS-2); // largest normal value
                    preshift_mant = ~0;  // largest normal value and RS bits set
                    of_before_round = 1'b1;
                end else if ($signed(input_exp_s1[i]) < $signed(-MAN_BITS-EXP_BIAS)) begin
                    // Limit the shift to retain sticky bits
                    final_exp     = 0; // denormal result
                    denorm_shamt  = (2 + MAN_BITS); // to sticky                
                end else if ($signed(input_exp_s1[i]) < $signed(1-EXP_BIAS)) begin
                    // Denormalize underflowing values
                    final_exp     = 0; // denormal result
                    denorm_shamt  = (1-EXP_BIAS) - input_exp_s1[i]; // adjust right shifting               
                end
            end else begin                                
                if ($signed(input_exp_s1[i]) >= $signed((MAX_INT_WIDTH-1) + unsigned_s1)) begin
                    // overflow: when converting to unsigned the range is larger by one
                    denorm_shamt = SHAMT_BITS'(0); // prevent shifting
                    of_before_round = 1'b1;                
                end else if ($signed(input_exp_s1[i]) < $signed(-1)) begin
                    // underflow
                    denorm_shamt = MAX_INT_WIDTH+1; // all bits go to the sticky
                end else begin
                    // By default right shift mantissa to be an integer
                    denorm_shamt = (MAX_INT_WIDTH-1) - input_exp_s1[i];
                end              
            end     
        `IGNORE_WARNINGS_END  
        end

        assign destination_mant_s1[i] = preshift_mant >> denorm_shamt;
        assign final_exp_s1[i]        = final_exp;
        assign of_before_round_s1[i]  = of_before_round;
    end

    // Pipeline stage2
    
    wire                    valid_in_s2;
    wire [TAGW-1:0]         tag_in_s2;
    wire                    is_itof_s2;
    wire                    unsigned_s2;
    wire [2:0]              rnd_mode_s2;
    fp_class_t [LANES-1:0]  fp_clss_s2;   
    wire [LANES-1:0]        mant_is_zero_s2;
    wire [LANES-1:0]        input_sign_s2;
    wire [LANES-1:0][2*INT_MAN_WIDTH:0] destination_mant_s2;
    wire [LANES-1:0][INT_EXP_WIDTH-1:0] final_exp_s2;
    wire [LANES-1:0]        of_before_round_s2;
    
    VX_pipe_register #(
        .DATAW  (1 + TAGW + 1 + 1 + `INST_FRM_BITS + LANES * ($bits(fp_class_t) + 1 + 1 + (2*INT_MAN_WIDTH+1) + INT_EXP_WIDTH + 1)),
        .RESETW (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in_s1, tag_in_s1, is_itof_s1, unsigned_s1, rnd_mode_s1, fp_clss_s1, mant_is_zero_s1, input_sign_s1, destination_mant_s1, final_exp_s1, of_before_round_s1}),
        .data_out ({valid_in_s2, tag_in_s2, is_itof_s2, unsigned_s2, rnd_mode_s2, fp_clss_s2, mant_is_zero_s2, input_sign_s2, destination_mant_s2, final_exp_s2, of_before_round_s2})
    );

    wire [LANES-1:0]       rounded_sign;
    wire [LANES-1:0][31:0] rounded_abs;     // absolute value of result after rounding
    wire [LANES-1:0][1:0]  fp_round_sticky_bits, int_round_sticky_bits;
    
    // Rouding and classification
   
    for (genvar i = 0; i < LANES; ++i) begin
        wire [MAN_BITS-1:0]      final_mant;        // mantissa after adjustments
        wire [MAX_INT_WIDTH-1:0] final_int;         // integer shifted in position
        wire [1:0]               round_sticky_bits;
        wire [31:0]              fmt_pre_round_abs;
        wire [31:0]              pre_round_abs;

        // Extract final mantissa and round bit, discard the normal bit (for FP)
        assign {final_mant, fp_round_sticky_bits[i][1]} = destination_mant_s2[i][2*INT_MAN_WIDTH-1 : 2*INT_MAN_WIDTH-1 - (MAN_BITS+1) + 1];
        assign {final_int, int_round_sticky_bits[i][1]} = destination_mant_s2[i][2*INT_MAN_WIDTH   : 2*INT_MAN_WIDTH   - (MAX_INT_WIDTH+1) + 1];

        // Collapse sticky bits
        assign fp_round_sticky_bits[i][0]  = (| destination_mant_s2[i][NUM_FP_STICKY-1:0]);
        assign int_round_sticky_bits[i][0] = (| destination_mant_s2[i][NUM_INT_STICKY-1:0]);

        // select RS bits for destination operation
        assign round_sticky_bits = is_itof_s2 ? fp_round_sticky_bits[i] : int_round_sticky_bits[i];

        // Pack exponent and mantissa into proper rounding form
        assign fmt_pre_round_abs = {1'b0, final_exp_s2[i][EXP_BITS-1:0], final_mant[MAN_BITS-1:0]};

        // Select output with destination format and operation
        assign pre_round_abs = is_itof_s2 ? fmt_pre_round_abs : final_int;

        // Perform the rounding
        VX_fp_rounding #(
            .DAT_WIDTH (32)
        ) fp_rounding (
            .abs_value_i    (pre_round_abs),
            .sign_i         (input_sign_s2[i]),
            .round_sticky_bits_i(round_sticky_bits),
            .rnd_mode_i     (rnd_mode_s2),
            .effective_subtraction_i(1'b0),
            .abs_rounded_o  (rounded_abs[i]),
            .sign_o         (rounded_sign[i]),
            `UNUSED_PIN (exact_zero_o)
        );
    end

    // Pipeline stage3

    wire                    valid_in_s3;
    wire [TAGW-1:0]         tag_in_s3;
    wire                    is_itof_s3;
    wire                    unsigned_s3;
    fp_class_t [LANES-1:0]  fp_clss_s3;   
    wire [LANES-1:0]        mant_is_zero_s3;
    wire [LANES-1:0]        input_sign_s3;
    wire [LANES-1:0]        rounded_sign_s3;
    wire [LANES-1:0][31:0]  rounded_abs_s3;
    wire [LANES-1:0]        of_before_round_s3;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + 1 + 1 + LANES * ($bits(fp_class_t) + 1 + 1 + 32 + 1 + 1)),
        .RESETW (1)
    ) pipe_reg3 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in_s2, tag_in_s2, is_itof_s2, unsigned_s2, fp_clss_s2, mant_is_zero_s2, input_sign_s2, rounded_abs,    rounded_sign,    of_before_round_s2}),
        .data_out ({valid_in_s3, tag_in_s3, is_itof_s3, unsigned_s3, fp_clss_s3, mant_is_zero_s3, input_sign_s3, rounded_abs_s3, rounded_sign_s3, of_before_round_s3})
    );
     
    wire [LANES-1:0] of_after_round;
    wire [LANES-1:0] uf_after_round;
    wire [LANES-1:0][31:0] fmt_result;
    wire [LANES-1:0][31:0] rounded_int_res; // after possible inversion
    wire [LANES-1:0] rounded_int_res_zero;  // after rounding

    for (genvar i = 0; i < LANES; ++i) begin
        // Assemble regular result, nan box short ones. Int zeroes need to be detected
        assign fmt_result[i] = (is_itof_s3 & mant_is_zero_s3[i]) ? 0 : {rounded_sign_s3[i], rounded_abs_s3[i][EXP_BITS+MAN_BITS-1:0]};

        // Classification after rounding select by destination format
        assign uf_after_round[i] = (rounded_abs_s3[i][EXP_BITS+MAN_BITS-1:MAN_BITS] == 0); // denormal
        assign of_after_round[i] = (rounded_abs_s3[i][EXP_BITS+MAN_BITS-1:MAN_BITS] == ~0); // inf exp.

        // Negative integer result needs to be brought into two's complement
        assign rounded_int_res[i] = rounded_sign_s3[i] ? (-rounded_abs_s3[i]) : rounded_abs_s3[i];
        assign rounded_int_res_zero[i] = (rounded_int_res[i] == 0);
    end

    // FP Special case handling

    wire [LANES-1:0][31:0]  fp_special_result;
    fflags_t [LANES-1:0]    fp_special_status;
    wire [LANES-1:0]        fp_result_is_special;

    localparam logic [EXP_BITS-1:0] QNAN_EXPONENT = 2**EXP_BITS-1;
    localparam logic [MAN_BITS-1:0] QNAN_MANTISSA = 2**(MAN_BITS-1);

    for (genvar i = 0; i < LANES; ++i) begin
        // Detect special case from source format, I2F casts don't produce a special result
        assign fp_result_is_special[i] = ~is_itof_s3 & (fp_clss_s3[i].is_zero | fp_clss_s3[i].is_nan);

        // Signalling input NaNs raise invalid flag, otherwise no flags set
        assign fp_special_status[i] = fp_clss_s3[i].is_signaling ? {1'b1, 4'h0} : 5'h0;   // invalid operation

        // Assemble result according to destination format
        assign fp_special_result[i] = fp_clss_s3[i].is_zero ? (32'(input_sign_s3) << 31) // signed zero
                                                            : {1'b0, QNAN_EXPONENT, QNAN_MANTISSA}; // qNaN
    end

    // INT Special case handling

    reg [LANES-1:0][31:0]   int_special_result;
    fflags_t [LANES-1:0]    int_special_status;
    wire [LANES-1:0]        int_result_is_special;

    for (genvar i = 0; i < LANES; ++i) begin
         // Assemble result according to destination format
        always @(*) begin
            if (input_sign_s3[i] && !fp_clss_s3[i].is_nan) begin
                int_special_result[i][30:0] = 0;               // alone yields 2**(31)-1
                int_special_result[i][31]   = ~unsigned_s3;    // for unsigned casts yields 2**31
            end else begin
                int_special_result[i][30:0] = 2**(31) - 1;     // alone yields 2**(31)-1
                int_special_result[i][31]   = unsigned_s3;     // for unsigned casts yields 2**31
            end
        end            

        // Detect special case from source format (inf, nan, overflow, nan-boxing or negative unsigned)
        assign int_result_is_special[i] = fp_clss_s3[i].is_nan 
                                        | fp_clss_s3[i].is_inf 
                                        | of_before_round_s3[i] 
                                        | (input_sign_s3[i] & unsigned_s3 & ~rounded_int_res_zero[i]);
                                        
        // All integer special cases are invalid
        assign int_special_status[i] = {1'b1, 4'h0};
    end

    // Result selection and Output handshake

    fflags_t [LANES-1:0] tmp_fflags;    
    wire [LANES-1:0][31:0] tmp_result;

    for (genvar i = 0; i < LANES; ++i) begin
        fflags_t    fp_regular_status, int_regular_status;
        fflags_t    fp_status, int_status;    
        wire [31:0] fp_result, int_result;

        wire inexact = is_itof_s3 ? (| fp_round_sticky_bits[i]) // overflow is invalid in i2f;        
                                  : (| fp_round_sticky_bits[i]) | (~fp_clss_s3[i].is_inf & (of_before_round_s3[i] | of_after_round[i]));
                                  
        assign fp_regular_status.NV = is_itof_s3 & (of_before_round_s3[i] | of_after_round[i]); // overflow is invalid for I2F casts
        assign fp_regular_status.DZ = 1'b0; // no divisions
        assign fp_regular_status.OF = ~is_itof_s3 & (~fp_clss_s3[i].is_inf & (of_before_round_s3[i] | of_after_round[i])); // inf casts no OF
        assign fp_regular_status.UF = uf_after_round[i] & inexact;
        assign fp_regular_status.NX = inexact;

        assign int_regular_status = (| int_round_sticky_bits[i]) ? {4'h0, 1'b1} : 5'h0;

        assign fp_result  = fp_result_is_special[i]  ? fp_special_result[i]  : fmt_result[i];        
        assign int_result = int_result_is_special[i] ? int_special_result[i] : rounded_int_res[i];

        assign fp_status  = fp_result_is_special[i]  ? fp_special_status[i]  : fp_regular_status;
        assign int_status = int_result_is_special[i] ? int_special_status[i] : int_regular_status;

        // Select output depending on special case detection
        assign tmp_result[i] = is_itof_s3 ? fp_result : int_result;
        assign tmp_fflags[i] = is_itof_s3 ? fp_status : int_status;
    end

    assign stall = ~ready_out && valid_out;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + (LANES * 32) + (LANES * `FFLAGS_BITS)),
        .RESETW (1)
    ) pipe_reg4 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_in_s3, tag_in_s3, tmp_result, tmp_fflags}),
        .data_out ({valid_out,   tag_out,   result,     fflags})
    );

    assign ready_in = ~stall;

    assign has_fflags = 1'b1;

endmodule