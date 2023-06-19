`include "VX_fpu_define.vh"

/// Modified port of cast module from fpnew Libray 
/// reference: https://github.com/pulp-platform/fpnew

module VX_fpu_cvt #(
    parameter NUM_LANES = 1,
    parameter TAGW = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire [`INST_FRM_BITS-1:0] frm,

    input wire is_itof,
    input wire is_signed,

    input wire [NUM_LANES-1:0][31:0]  dataa,
    output wire [NUM_LANES-1:0][31:0] result, 

    output wire has_fflags,
    output wire [NUM_LANES-1:0][`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);   
    // Constants
 
    localparam MAN_BITS = 23;
    localparam EXP_BITS = 8;
    localparam EXP_BIAS = 2**(EXP_BITS-1)-1;    

    localparam logic [EXP_BITS-1:0] QNAN_EXPONENT = 2**EXP_BITS-1;
    localparam logic [MAN_BITS-1:0] QNAN_MANTISSA = 2**(MAN_BITS-1);
    
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
    
    fclass_t [NUM_LANES-1:0] fclass;
      
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        VX_fpu_class #( 
            .EXP_BITS (EXP_BITS),
            .MAN_BITS (MAN_BITS)
        ) fp_class (
            .exp_i  (dataa[i][30:23]),
            .man_i  (dataa[i][22:0]),
            .clss_o (fclass[i])
        );
    end

    wire [NUM_LANES-1:0][INT_MAN_WIDTH-1:0] input_mant;
    wire [NUM_LANES-1:0][INT_EXP_WIDTH-1:0] input_exp;    
    wire [NUM_LANES-1:0]                    input_sign;
    
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [INT_MAN_WIDTH-1:0] int_mantissa;
        wire [INT_MAN_WIDTH-1:0] fmt_mantissa;
        wire fmt_sign        = dataa[i][31];
        wire int_sign        = dataa[i][31] && is_signed;
        assign int_mantissa  = int_sign ? (-dataa[i]) : dataa[i];
        assign fmt_mantissa  = INT_MAN_WIDTH'({fclass[i].is_normal, dataa[i][MAN_BITS-1:0]});
        assign input_exp[i]  = {1'b0, dataa[i][MAN_BITS +: EXP_BITS]} + INT_EXP_WIDTH'({1'b0, fclass[i].is_subnormal});
        assign input_mant[i] = is_itof ? int_mantissa : fmt_mantissa;
        assign input_sign[i] = is_itof ? int_sign : fmt_sign;
    end

    // Pipeline stage0
    
    wire                    valid_in_s0;
    wire [TAGW-1:0]         tag_in_s0;
    wire                    is_itof_s0;
    wire                    unsigned_s0;
    wire [2:0]              rnd_mode_s0;
    fclass_t [NUM_LANES-1:0] fclass_s0;
    wire [NUM_LANES-1:0]    input_sign_s0;
    wire [NUM_LANES-1:0][INT_EXP_WIDTH-1:0] fmt_exponent_s0;
    wire [NUM_LANES-1:0][INT_MAN_WIDTH-1:0] encoded_mant_s0;

    wire stall;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + 1 + `INST_FRM_BITS + 1 + NUM_LANES * ($bits(fclass_t) + 1 + INT_EXP_WIDTH + INT_MAN_WIDTH)),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in,    tag_in,    is_itof,    !is_signed,  frm,         fclass,    input_sign,    input_exp,    input_mant}),
        .data_out ({valid_in_s0, tag_in_s0, is_itof_s0, unsigned_s0, rnd_mode_s0, fclass_s0, input_sign_s0, fmt_exponent_s0, encoded_mant_s0})
    );
    
    // Normalization

    wire [NUM_LANES-1:0][LZC_RESULT_WIDTH-1:0] renorm_shamt_s0; // renormalization shift amount
    wire [NUM_LANES-1:0] mant_is_zero_s0;                       // for integer zeroes

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire mant_is_nonzero_s0;
        VX_lzc #(
            .N (INT_MAN_WIDTH)
        ) lzc (
            .data_in   (encoded_mant_s0[i]),
            .data_out  (renorm_shamt_s0[i]),
            .valid_out (mant_is_nonzero_s0)
        );
        assign mant_is_zero_s0[i] = ~mant_is_nonzero_s0;  
    end

    wire [NUM_LANES-1:0][INT_MAN_WIDTH-1:0] input_mant_n_s0;    // normalized input mantissa    
    wire [NUM_LANES-1:0][INT_EXP_WIDTH-1:0] input_exp_n_s0;     // unbiased true exponent
    
    for (genvar i = 0; i < NUM_LANES; ++i) begin
       // Realign input mantissa, append zeroes if destination is wider
        assign input_mant_n_s0[i] = encoded_mant_s0[i] << renorm_shamt_s0[i];

        // Unbias exponent and compensate for shift
        wire [INT_EXP_WIDTH-1:0] fp_input_exp_s0 = fmt_exponent_s0[i] + INT_EXP_WIDTH'(FMT_SHIFT_COMPENSATION - EXP_BIAS) - INT_EXP_WIDTH'({1'b0, renorm_shamt_s0[i]});
        wire [INT_EXP_WIDTH-1:0] int_input_exp_s0 = INT_EXP_WIDTH'(INT_MAN_WIDTH-1) - INT_EXP_WIDTH'({1'b0, renorm_shamt_s0[i]});
        assign input_exp_n_s0[i] = is_itof_s0 ? int_input_exp_s0 : fp_input_exp_s0;
    end

    // Pipeline stage1

    wire                    valid_in_s1;
    wire [TAGW-1:0]         tag_in_s1;
    wire                    is_itof_s1;
    wire                    unsigned_s1;
    wire [2:0]              rnd_mode_s1;
    fclass_t [NUM_LANES-1:0] fclass_s1;
    wire [NUM_LANES-1:0]    input_sign_s1;
    wire [NUM_LANES-1:0]    mant_is_zero_s1;
    wire [NUM_LANES-1:0][INT_MAN_WIDTH-1:0] input_mant_s1;
    wire [NUM_LANES-1:0][INT_EXP_WIDTH-1:0] input_exp_s1;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + 1 + `INST_FRM_BITS + 1 + NUM_LANES * ($bits(fclass_t) + 1 + 1 + INT_MAN_WIDTH + INT_EXP_WIDTH)),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in_s0, tag_in_s0, is_itof_s0, unsigned_s0, rnd_mode_s0, fclass_s0, input_sign_s0, mant_is_zero_s0, input_mant_n_s0, input_exp_n_s0}),
        .data_out ({valid_in_s1, tag_in_s1, is_itof_s1, unsigned_s1, rnd_mode_s1, fclass_s1, input_sign_s1, mant_is_zero_s1, input_mant_s1, input_exp_s1})
    );

    // Perform adjustments to mantissa and exponent

    wire [NUM_LANES-1:0][2*INT_MAN_WIDTH:0] destination_mant_s1;
    wire [NUM_LANES-1:0][INT_EXP_WIDTH-1:0] final_exp_s1;
    wire [NUM_LANES-1:0]                    of_before_round_s1;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        reg [2*INT_MAN_WIDTH:0] preshift_mant_s1;   // mantissa before final shift                
        reg [SHAMT_BITS-1:0]    denorm_shamt_s1;    // shift amount for denormalization
        reg [INT_EXP_WIDTH-1:0] final_exp_tmp_s1;   // after eventual adjustments
        reg                     of_before_round_tmp_s1;

        always @(*) begin
            final_exp_tmp_s1 = input_exp_s1[i] + INT_EXP_WIDTH'(EXP_BIAS); // take exponent as is, only look at lower bits
            preshift_mant_s1 = {input_mant_s1[i], 33'b0};
            denorm_shamt_s1  = '0;
            of_before_round_tmp_s1 = 1'b0;

            if (is_itof_s1) begin                   
                if ($signed(input_exp_s1[i]) >= INT_EXP_WIDTH'($signed(2**EXP_BITS-1-EXP_BIAS))) begin
                    // Overflow or infinities (for proper rounding)
                    final_exp_tmp_s1 = (2**EXP_BITS-2); // largest normal value
                    preshift_mant_s1 = ~0;  // largest normal value and RS bits set
                    of_before_round_tmp_s1 = 1'b1;
                end else if ($signed(input_exp_s1[i]) < INT_EXP_WIDTH'($signed(-MAN_BITS-EXP_BIAS))) begin
                    // Limit the shift to retain sticky bits
                    final_exp_tmp_s1 = '0; // denormal result
                    denorm_shamt_s1  = (2 + MAN_BITS); // to sticky                
                end else if ($signed(input_exp_s1[i]) < INT_EXP_WIDTH'($signed(1-EXP_BIAS))) begin
                    // Denormalize underflowing values
                    final_exp_tmp_s1 = '0; // denormal result
                    denorm_shamt_s1  = SHAMT_BITS'(1-EXP_BIAS) - SHAMT_BITS'(input_exp_s1[i]); // adjust right shifting               
                end
            end else begin
                if ($signed(input_exp_s1[i]) >= $signed(INT_EXP_WIDTH'(MAX_INT_WIDTH-1) + INT_EXP_WIDTH'(unsigned_s1))) begin
                    // overflow: when converting to unsigned the range is larger by one
                    of_before_round_tmp_s1 = 1'b1;                
                end else if ($signed(input_exp_s1[i]) < INT_EXP_WIDTH'($signed(-1))) begin
                    // underflow
                    denorm_shamt_s1 = MAX_INT_WIDTH+1; // all bits go to the sticky
                end else begin
                    // By default right shift mantissa to be an integer
                    denorm_shamt_s1 = SHAMT_BITS'(MAX_INT_WIDTH-1) - SHAMT_BITS'(input_exp_s1[i]);
                end              
            end
        end

        assign destination_mant_s1[i] = preshift_mant_s1 >> denorm_shamt_s1;
        assign final_exp_s1[i]        = final_exp_tmp_s1;
        assign of_before_round_s1[i]  = of_before_round_tmp_s1;
    end

    // Pipeline stage2
    
    wire                    valid_in_s2;
    wire [TAGW-1:0]         tag_in_s2;
    wire                    is_itof_s2;
    wire                    unsigned_s2;
    wire [2:0]              rnd_mode_s2;
    fclass_t [NUM_LANES-1:0] fclass_s2;   
    wire [NUM_LANES-1:0]    mant_is_zero_s2;
    wire [NUM_LANES-1:0]    input_sign_s2;
    wire [NUM_LANES-1:0][2*INT_MAN_WIDTH:0] destination_mant_s2;
    wire [NUM_LANES-1:0][INT_EXP_WIDTH-1:0] final_exp_s2;
    wire [NUM_LANES-1:0]    of_before_round_s2;
    
    VX_pipe_register #(
        .DATAW  (1 + TAGW + 1 + 1 + `INST_FRM_BITS + NUM_LANES * ($bits(fclass_t) + 1 + 1 + (2*INT_MAN_WIDTH+1) + INT_EXP_WIDTH + 1)),
        .RESETW (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in_s1, tag_in_s1, is_itof_s1, unsigned_s1, rnd_mode_s1, fclass_s1, mant_is_zero_s1, input_sign_s1, destination_mant_s1, final_exp_s1, of_before_round_s1}),
        .data_out ({valid_in_s2, tag_in_s2, is_itof_s2, unsigned_s2, rnd_mode_s2, fclass_s2, mant_is_zero_s2, input_sign_s2, destination_mant_s2, final_exp_s2, of_before_round_s2})
    );

    wire [NUM_LANES-1:0]       rounded_sign_s2;
    wire [NUM_LANES-1:0][31:0] rounded_abs_s2;      // absolute value of result after rounding
    wire [NUM_LANES-1:0]       int_round_has_sticky_s2;
    wire [NUM_LANES-1:0]       fp_round_has_sticky_s2;
    
    // Rouding and classification
   
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [MAN_BITS-1:0]      final_mant_s2;        // mantissa after adjustments
        wire [MAX_INT_WIDTH-1:0] final_int_s2;         // integer shifted in position
        wire [1:0]               round_sticky_bits_s2;
        wire [31:0]              fmt_pre_round_abs_s2;
        wire [31:0]              pre_round_abs_s2;
        wire [1:0]               int_round_sticky_bits_s2, fp_round_sticky_bits_s2;

        // Extract final mantissa and round bit, discard the normal bit (for FP)
        assign {final_mant_s2, fp_round_sticky_bits_s2[1]} = destination_mant_s2[i][2*INT_MAN_WIDTH-1 : 2*INT_MAN_WIDTH-1 - (MAN_BITS+1) + 1];
        assign {final_int_s2, int_round_sticky_bits_s2[1]} = destination_mant_s2[i][2*INT_MAN_WIDTH   : 2*INT_MAN_WIDTH   - (MAX_INT_WIDTH+1) + 1];

        // Collapse sticky bits
        assign fp_round_sticky_bits_s2[0]  = (| destination_mant_s2[i][NUM_FP_STICKY-1:0]);
        assign int_round_sticky_bits_s2[0] = (| destination_mant_s2[i][NUM_INT_STICKY-1:0]);
        assign fp_round_has_sticky_s2[i]   = (| fp_round_sticky_bits_s2);
        assign int_round_has_sticky_s2[i]  = (| int_round_sticky_bits_s2);

        // select RS bits for destination operation
        assign round_sticky_bits_s2 = is_itof_s2 ? fp_round_sticky_bits_s2 : int_round_sticky_bits_s2;

        // Pack exponent and mantissa into proper rounding form
        assign fmt_pre_round_abs_s2 = {1'b0, final_exp_s2[i][EXP_BITS-1:0], final_mant_s2[MAN_BITS-1:0]};

        // Select output with destination format and operation
        assign pre_round_abs_s2 = is_itof_s2 ? fmt_pre_round_abs_s2 : final_int_s2;

        // Perform the rounding
        VX_fpu_rounding #(
            .DAT_WIDTH (32)
        ) fp_rounding (
            .abs_value_i (pre_round_abs_s2),
            .sign_i      (input_sign_s2[i]),
            .round_sticky_bits_i (round_sticky_bits_s2),
            .rnd_mode_i  (rnd_mode_s2),
            .effective_subtraction_i (1'b0),
            .abs_rounded_o (rounded_abs_s2[i]),
            .sign_o      (rounded_sign_s2[i]),
            `UNUSED_PIN  (exact_zero_o)
        );
    end

    // Pipeline stage3

    wire                 valid_in_s3;
    wire [TAGW-1:0]      tag_in_s3;
    wire                 is_itof_s3;
    wire                 unsigned_s3;
    fclass_t [NUM_LANES-1:0] fclass_s3;   
    wire [NUM_LANES-1:0] mant_is_zero_s3;
    wire [NUM_LANES-1:0] input_sign_s3;
    wire [NUM_LANES-1:0] rounded_sign_s3;
    wire [NUM_LANES-1:0][31:0] rounded_abs_s3;
    wire [NUM_LANES-1:0] of_before_round_s3;   
    wire [NUM_LANES-1:0] int_round_has_sticky_s3;
    wire [NUM_LANES-1:0] fp_round_has_sticky_s3; 

    VX_pipe_register #(
        .DATAW  (1 + TAGW + 1 + 1 + NUM_LANES * ($bits(fclass_t) + 1 + 1 + 32 + 1 + 1 + 1 + 1)),
        .RESETW (1)
    ) pipe_reg3 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in_s2, tag_in_s2, is_itof_s2, unsigned_s2, fclass_s2, mant_is_zero_s2, input_sign_s2, rounded_abs_s2, rounded_sign_s2, of_before_round_s2, int_round_has_sticky_s2, fp_round_has_sticky_s2}),
        .data_out ({valid_in_s3, tag_in_s3, is_itof_s3, unsigned_s3, fclass_s3, mant_is_zero_s3, input_sign_s3, rounded_abs_s3, rounded_sign_s3, of_before_round_s3, int_round_has_sticky_s3, fp_round_has_sticky_s3})
    );
     
    wire [NUM_LANES-1:0] of_after_round_s3;
    wire [NUM_LANES-1:0] uf_after_round_s3;
    wire [NUM_LANES-1:0][31:0] fmt_result_s3;
    wire [NUM_LANES-1:0][31:0] rounded_int_res_s3; // after possible inversion
    wire [NUM_LANES-1:0] rounded_int_res_zero_s3;  // after rounding

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        // Assemble regular result, nan box short ones. Int zeroes need to be detected
        assign fmt_result_s3[i] = (is_itof_s3 & mant_is_zero_s3[i]) ? 0 : {rounded_sign_s3[i], rounded_abs_s3[i][EXP_BITS+MAN_BITS-1:0]};

        // Classification after rounding select by destination format
        assign uf_after_round_s3[i] = (rounded_abs_s3[i][EXP_BITS+MAN_BITS-1:MAN_BITS] == 0);  // denormal
        assign of_after_round_s3[i] = (rounded_abs_s3[i][EXP_BITS+MAN_BITS-1:MAN_BITS] == ~0); // inf exp.

        // Negative integer result needs to be brought into two's complement
        assign rounded_int_res_s3[i] = rounded_sign_s3[i] ? (-rounded_abs_s3[i]) : rounded_abs_s3[i];
        assign rounded_int_res_zero_s3[i] = (rounded_int_res_s3[i] == 0);
    end

    // FP Special case handling

    wire [NUM_LANES-1:0][31:0] fp_special_result_s3;
    fflags_t [NUM_LANES-1:0]   fp_special_status_s3;
    wire [NUM_LANES-1:0]       fp_result_is_special_s3;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        // Detect special case from source format, I2F casts don't produce a special result
        assign fp_result_is_special_s3[i] = ~is_itof_s3 & (fclass_s3[i].is_zero | fclass_s3[i].is_nan);

        // Signalling input NaNs raise invalid flag, otherwise no flags set
        assign fp_special_status_s3[i] = fclass_s3[i].is_signaling ? {1'b1, 4'h0} : 5'h0; // invalid operation

        // Assemble result according to destination format
        assign fp_special_result_s3[i] = fclass_s3[i].is_zero ? (32'(input_sign_s3) << 31) // signed zero
                                                              : {1'b0, QNAN_EXPONENT, QNAN_MANTISSA}; // qNaN
    end

    // INT Special case handling

    reg [NUM_LANES-1:0][31:0] int_special_result_s3;
    fflags_t [NUM_LANES-1:0]  int_special_status_s3;
    wire [NUM_LANES-1:0]      int_result_is_special_s3;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
         // Assemble result according to destination format
        always @(*) begin
            if (input_sign_s3[i] && !fclass_s3[i].is_nan) begin
                int_special_result_s3[i][30:0] = '0;            // alone yields 2**(31)-1
                int_special_result_s3[i][31]   = ~unsigned_s3;  // for unsigned casts yields 2**31
            end else begin
                int_special_result_s3[i][30:0] = 2**(31) - 1;   // alone yields 2**(31)-1
                int_special_result_s3[i][31]   = unsigned_s3;   // for unsigned casts yields 2**31
            end
        end            

        // Detect special case from source format (inf, nan, overflow, nan-boxing or negative unsigned)
        assign int_result_is_special_s3[i] = fclass_s3[i].is_nan 
                                           | fclass_s3[i].is_inf
                                           | of_before_round_s3[i]
                                           | (input_sign_s3[i] & unsigned_s3 & ~rounded_int_res_zero_s3[i]);
                                        
        // All integer special cases are invalid
        assign int_special_status_s3[i] = {1'b1, 4'h0};
    end

    // Result selection and Output handshake

    fflags_t [NUM_LANES-1:0] tmp_fflags_s3;    
    wire [NUM_LANES-1:0][31:0] tmp_result_s3;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        fflags_t    fp_regular_status_s3, int_regular_status_s3;
        fflags_t    fp_status_s3, int_status_s3;    
        wire [31:0] fp_result_s3, int_result_s3;

        wire inexact_s3 = is_itof_s3 ? fp_round_has_sticky_s3[i] // overflow is invalid in i2f;        
                                     : (fp_round_has_sticky_s3[i] || (~fclass_s3[i].is_inf && (of_before_round_s3[i] || of_after_round_s3[i])));
                                  
        assign fp_regular_status_s3.NV = is_itof_s3 & (of_before_round_s3[i] | of_after_round_s3[i]); // overflow is invalid for I2F casts
        assign fp_regular_status_s3.DZ = 1'b0; // no divisions
        assign fp_regular_status_s3.OF = ~is_itof_s3 & (~fclass_s3[i].is_inf & (of_before_round_s3[i] | of_after_round_s3[i])); // inf casts no OF
        assign fp_regular_status_s3.UF = uf_after_round_s3[i] & inexact_s3;
        assign fp_regular_status_s3.NX = inexact_s3;

        assign int_regular_status_s3 = int_round_has_sticky_s3[i] ? {4'h0, 1'b1} : 5'h0;

        assign fp_result_s3  = fp_result_is_special_s3[i]  ? fp_special_result_s3[i]  : fmt_result_s3[i];        
        assign int_result_s3 = int_result_is_special_s3[i] ? int_special_result_s3[i] : rounded_int_res_s3[i];

        assign fp_status_s3  = fp_result_is_special_s3[i]  ? fp_special_status_s3[i]  : fp_regular_status_s3;
        assign int_status_s3 = int_result_is_special_s3[i] ? int_special_status_s3[i] : int_regular_status_s3;

        // Select output depending on special case detection
        assign tmp_result_s3[i] = is_itof_s3 ? fp_result_s3 : int_result_s3;
        assign tmp_fflags_s3[i] = is_itof_s3 ? fp_status_s3 : int_status_s3;
    end

    assign stall = ~ready_out && valid_out;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + (NUM_LANES * 32) + (NUM_LANES * `FP_FLAGS_BITS)),
        .RESETW (1)
    ) pipe_reg4 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_in_s3, tag_in_s3, tmp_result_s3, tmp_fflags_s3}),
        .data_out ({valid_out,   tag_out,   result,        fflags})
    );

    assign ready_in = ~stall;

    assign has_fflags = 1'b1;

endmodule
