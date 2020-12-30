`include "VX_define.vh"

`ifndef SYNTHESIS
`include "float_dpi.vh"
`endif

module VX_fp_cvt #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire [`FRM_BITS-1:0] frm,

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
    
    fp_type_t [LANES-1:0] in_a_type;
      
    for (genvar i = 0; i < LANES; ++i) begin
        VX_fp_type fp_type (
            .exp_i  (dataa[i][30:23]),
            .man_i  (dataa[i][22:0]),
            .type_o (in_a_type[i])
        );
    end

    wire [LANES-1:0][INT_MAN_WIDTH-1:0]        encoded_mant; // input mantissa with implicit bit    
    wire signed [LANES-1:0][INT_EXP_WIDTH-1:0] fmt_exponent;    
    wire [LANES-1:0]                           input_sign;
    
    for (genvar i = 0; i < LANES; ++i) begin
        wire [INT_MAN_WIDTH-1:0] int_mantissa;
        wire [INT_MAN_WIDTH-1:0] fmt_mantissa;
        wire fmt_sign       = dataa[i][31];
        wire int_sign       = dataa[i][31] & is_signed;
        assign int_mantissa = int_sign ? $unsigned(-dataa[i]) : dataa[i];
        assign fmt_mantissa = INT_MAN_WIDTH'({in_a_type[i].is_normal, dataa[i][MAN_BITS-1:0]});            

        assign fmt_exponent[i] = $signed({1'b0, dataa[i][MAN_BITS+EXP_BITS-1:MAN_BITS]});
        assign encoded_mant[i] = is_itof ? int_mantissa : fmt_mantissa;
        assign input_sign[i]   = is_itof ? int_sign : fmt_sign;
    end

    wire [LANES-1:0][LZC_RESULT_WIDTH-1:0] renorm_shamt; // renormalization shift amount
    wire [LANES-1:0] mant_is_zero;                       // for integer zeroes

    for (genvar i = 0; i < LANES; ++i) begin
        // Leading zero counter for cancellations
        wire mant_is_nonzero;
        VX_lzc #(
            .DATAW (INT_MAN_WIDTH)
        ) lzc (
            .data_in   (encoded_mant[i]),
            .data_out  (renorm_shamt[i]),
            .valid_out (mant_is_nonzero)
        );
        assign mant_is_zero[i] = ~mant_is_nonzero;
    end

    // Pipeline stage0
    
    wire                    valid_in_s0;
    wire [TAGW-1:0]         tag_in_s0;
    wire                    is_itof_s0;
    wire                    unsigned_s0;
    wire [2:0]              rnd_mode_s0;
    fp_type_t [LANES-1:0]   in_a_type_s0;
    wire [LANES-1:0]        input_sign_s0;
    wire signed [LANES-1:0][INT_EXP_WIDTH-1:0] fmt_exponent_s0;
    wire [LANES-1:0][INT_MAN_WIDTH-1:0] encoded_mant_s0;    
    wire [LANES-1:0][LZC_RESULT_WIDTH-1:0] renorm_shamt_s0;
    wire [LANES-1:0]        mant_is_zero_s0;

    wire stall;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + 1 + `FRM_BITS + 1 + LANES * ($bits(fp_type_t) + 1 +INT_EXP_WIDTH + INT_MAN_WIDTH + LZC_RESULT_WIDTH + 1)),
        .RESETW (1),
        .DEPTH  (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in,    tag_in,    is_itof,    !is_signed,  frm,         in_a_type,    input_sign,    fmt_exponent,    encoded_mant,    renorm_shamt,    mant_is_zero}),
        .data_out ({valid_in_s0, tag_in_s0, is_itof_s0, unsigned_s0, rnd_mode_s0, in_a_type_s0, input_sign_s0, fmt_exponent_s0, encoded_mant_s0, renorm_shamt_s0, mant_is_zero_s0})
    );
    
    // Normalization

    wire        [LANES-1:0][INT_MAN_WIDTH-1:0] input_mant;      // normalized input mantissa    
    wire signed [LANES-1:0][INT_EXP_WIDTH-1:0] input_exp;       // unbiased true exponent
    wire signed [LANES-1:0][INT_EXP_WIDTH-1:0] destination_exp; // re-biased exponent for destination

    for (genvar i = 0; i < LANES; ++i) begin
    `IGNORE_WARNINGS_BEGIN            
        // Input mantissa needs to be normalized
        wire signed [INT_EXP_WIDTH-1:0] fp_input_exp;
        wire signed [INT_EXP_WIDTH-1:0] int_input_exp;
        wire [LZC_RESULT_WIDTH:0] renorm_shamt_sgn; 
        
        // signed form for calculations
        assign renorm_shamt_sgn = $signed({1'b0, renorm_shamt_s0[i]});

        // Realign input mantissa, append zeroes if destination is wider
        assign input_mant[i] = encoded_mant_s0[i] << renorm_shamt_s0[i];

        // Unbias exponent and compensate for shift
        assign fp_input_exp = $signed(fmt_exponent_s0[i] + 
                                      $signed({1'b0, in_a_type_s0[i].is_subnormal}) - 
                                      $signed(EXP_BIAS) -
                                      renorm_shamt_sgn + 
                                      $signed(FMT_SHIFT_COMPENSATION));
                                 
        assign int_input_exp = $signed(INT_MAN_WIDTH - 1 - renorm_shamt_sgn);

        assign input_exp[i]  = is_itof_s0 ? int_input_exp : fp_input_exp;

        // Rebias the exponent
        assign destination_exp[i] = input_exp[i] + $signed(EXP_BIAS);
    `IGNORE_WARNINGS_END
    end

    // Pipeline stage1
    
    wire                    valid_in_s1;
    wire [TAGW-1:0]         tag_in_s1;
    wire                    is_itof_s1;
    wire                    unsigned_s1;
    wire [2:0]              rnd_mode_s1;
    fp_type_t [LANES-1:0]   in_a_type_s1;   
    wire [LANES-1:0]        mant_is_zero_s1;
    wire        [LANES-1:0]                    input_sign_s1;
    wire signed [LANES-1:0][INT_EXP_WIDTH-1:0] input_exp_s1;
    wire signed [LANES-1:0][INT_EXP_WIDTH-1:0] destination_exp_s1;
    wire        [LANES-1:0][INT_MAN_WIDTH-1:0] input_mant_s1;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + 1 + `FRM_BITS + 1 + LANES * ($bits(fp_type_t) + 1 + 1 + INT_MAN_WIDTH + 2*INT_EXP_WIDTH)),
        .RESETW (1),
        .DEPTH  (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in_s0, tag_in_s0, is_itof_s0, unsigned_s0, rnd_mode_s0, in_a_type_s0, mant_is_zero_s0, input_sign_s0, input_mant,    input_exp,    destination_exp}),
        .data_out ({valid_in_s1, tag_in_s1, is_itof_s1, unsigned_s1, rnd_mode_s1, in_a_type_s1, mant_is_zero_s1, input_sign_s1, input_mant_s1, input_exp_s1, destination_exp_s1})
    );

    // Casting
    reg  [LANES-1:0][INT_EXP_WIDTH-1:0] final_exp;          // after eventual adjustments

    reg  [LANES-1:0][2*INT_MAN_WIDTH:0]  preshift_mant;     // mantissa before final shift
    wire [LANES-1:0][2*INT_MAN_WIDTH:0]  destination_mant;  // mantissa from shifter, with rnd bit
    wire [LANES-1:0][MAN_BITS-1:0]       final_mant;        // mantissa after adjustments
    wire [LANES-1:0][MAX_INT_WIDTH-1:0]  final_int;         // integer shifted in position

    reg  [LANES-1:0][SHAMT_BITS-1:0] denorm_shamt; // shift amount for denormalization

    wire [LANES-1:0][1:0] fp_round_sticky_bits, int_round_sticky_bits, round_sticky_bits;
    reg  [LANES-1:0]      of_before_round;

    // Perform adjustments to mantissa and exponent
    for (genvar i = 0; i < LANES; ++i) begin
    `IGNORE_WARNINGS_BEGIN
        always @(*) begin
            // Default assignment
            final_exp[i]       = $unsigned(destination_exp_s1[i]); // take exponent as is, only look at lower bits
            preshift_mant[i]   = 65'b0;  // initialize mantissa container with zeroes
            denorm_shamt[i]    = MAN_BITS - MAN_BITS; // right of mantissa
            of_before_round[i] = 1'b0;

            // Place mantissa to the left of the shifter
            preshift_mant[i] = {input_mant_s1[i], 33'b0};

            // Handle INT casts
            if (is_itof_s1) begin                
                // Overflow or infinities (for proper rounding)
                if ((destination_exp_s1[i] >= 2**EXP_BITS-1) 
                 || (~is_itof_s1 && in_a_type_s1[i].is_inf)) begin
                    final_exp[i]       = $unsigned(2**EXP_BITS-2); // largest normal value
                    preshift_mant[i]   = ~0;  // largest normal value and RS bits set
                    of_before_round[i] = 1'b1;
                // Denormalize underflowing values
                end else if ((destination_exp_s1[i] < 1) 
                          && (destination_exp_s1[i] >= -$signed(MAN_BITS))) begin
                    final_exp[i]       = 0; // denormal result
                    denorm_shamt[i]    = $unsigned(denorm_shamt[i] + 1 - destination_exp_s1[i]); // adjust right shifting
                // Limit the shift to retain sticky bits
                end else if (destination_exp_s1[i] < -$signed(MAN_BITS)) begin
                    final_exp[i]       = 0; // denormal result
                    denorm_shamt[i]    = $unsigned(denorm_shamt[i] + 2 + MAN_BITS); // to sticky
                end
            end else begin
                // By default right shift mantissa to be an integer
                denorm_shamt[i] = $unsigned(MAX_INT_WIDTH - 1 - input_exp_s1[i]);
                // overflow: when converting to unsigned the range is larger by one
                if (input_exp_s1[i] >=  $signed(MAX_INT_WIDTH -1 + unsigned_s1)) begin
                    denorm_shamt[i]    = 1'b0; // prevent shifting
                    of_before_round[i] = 1'b1;
                // underflow
                end else if (input_exp_s1[i] < -1) begin
                    denorm_shamt[i]    = MAX_INT_WIDTH + 1; // all bits go to the sticky
                end               
            end
        end

        // Mantissa adjustment shift
        assign destination_mant[i] = preshift_mant[i] >> denorm_shamt[i];
        
        // Extract final mantissa and round bit, discard the normal bit (for FP)
        assign {final_mant[i], fp_round_sticky_bits[i][1]} = destination_mant[i][2*INT_MAN_WIDTH-1 : 2*INT_MAN_WIDTH-1 - (MAN_BITS+1) + 1];
        assign {final_int[i], int_round_sticky_bits[i][1]} = destination_mant[i][2*INT_MAN_WIDTH   : 2*INT_MAN_WIDTH   - (MAX_INT_WIDTH+1) + 1];

        // Collapse sticky bits
        assign fp_round_sticky_bits[i][0]  = (| destination_mant[i][NUM_FP_STICKY-1:0]);
        assign int_round_sticky_bits[i][0] = (| destination_mant[i][NUM_INT_STICKY-1:0]);

        // select RS bits for destination operation
        assign round_sticky_bits[i] = is_itof_s1 ? fp_round_sticky_bits[i] : int_round_sticky_bits[i];
    `IGNORE_WARNINGS_END
    end

    // Rouding and classification

    wire [LANES-1:0]        rounded_sign;
    wire [LANES-1:0][31:0]  rounded_abs;     // absolute value of result after rounding    

    for (genvar i = 0; i < LANES; ++i) begin
        // Pack exponent and mantissa into proper rounding form
        wire [31:0] fmt_pre_round_abs = {1'b0, final_exp[i][EXP_BITS-1:0], final_mant[i][MAN_BITS-1:0]};

        // Sign-extend integer result
        wire [31:0] ifmt_pre_round_abs = final_int[i];

        // Select output with destination format and operation
        wire [31:0] pre_round_abs = is_itof_s1 ? fmt_pre_round_abs : ifmt_pre_round_abs;

        // Perform the rounding
        VX_fp_rounding #(
            .DAT_WIDTH (32)
        ) fp_rounding (
            .abs_value_i (pre_round_abs),
            .sign_i (input_sign_s1[i]),
            .round_sticky_bits_i (round_sticky_bits[i]),
            .rnd_mode_i (rnd_mode_s1),
            .effective_subtraction_i (1'b0),
            .abs_rounded_o (rounded_abs[i]),
            .sign_o (rounded_sign[i]),
            `UNUSED_PIN (exact_zero_o)
        );
    end

    // Pipeline stage2

    wire                    valid_in_s2;
    wire [TAGW-1:0]         tag_in_s2;
    wire                    is_itof_s2;
    wire                    unsigned_s2;
    fp_type_t [LANES-1:0]   in_a_type_s2;   
    wire [LANES-1:0]        mant_is_zero_s2;
    wire [LANES-1:0]        input_sign_s2;
    wire [LANES-1:0]        rounded_sign_s2;
    wire [LANES-1:0][31:0]  rounded_abs_s2;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + 1 + 1 + LANES * ($bits(fp_type_t) + 1 + 1 + 32 + 1)),
        .RESETW (1),
        .DEPTH  (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in_s1, tag_in_s1, is_itof_s1, unsigned_s1, in_a_type_s1, mant_is_zero_s1, input_sign_s1, rounded_abs,    rounded_sign}),
        .data_out ({valid_in_s2, tag_in_s2, is_itof_s2, unsigned_s2, in_a_type_s2, mant_is_zero_s2, input_sign_s2, rounded_abs_s2, rounded_sign_s2})
    );
     
    wire [LANES-1:0] of_after_round;
    wire [LANES-1:0] uf_after_round;

    wire [LANES-1:0][31:0] fmt_result;

    wire [LANES-1:0][31:0] rounded_int_res; // after possible inversion
    wire [LANES-1:0] rounded_int_res_zero;  // after rounding

    for (genvar i = 0; i < LANES; ++i) begin
        // Assemble regular result, nan box short ones. Int zeroes need to be detected
        assign fmt_result[i] = (is_itof_s2 & mant_is_zero_s2[i]) ? 0 : {rounded_sign_s2[i], rounded_abs_s2[i][EXP_BITS+MAN_BITS-1:0]};

        // Classification after rounding select by destination format
        assign uf_after_round[i] = (rounded_abs_s2[i][EXP_BITS+MAN_BITS-1:MAN_BITS] == 0); // denormal
        assign of_after_round[i] = (rounded_abs_s2[i][EXP_BITS+MAN_BITS-1:MAN_BITS] == ~0); // inf exp.

        // Negative integer result needs to be brought into two's complement
        assign rounded_int_res[i] = rounded_sign_s2[i] ? $unsigned(-rounded_abs_s2[i]) : rounded_abs_s2[i];
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
        assign fp_result_is_special[i] = ~is_itof_s2 & (in_a_type_s2[i].is_zero | in_a_type_s2[i].is_nan);

        // Signalling input NaNs raise invalid flag, otherwise no flags set
        assign fp_special_status[i] = in_a_type_s2[i].is_signaling ? {1'b1, 4'h0} : 5'h0;   // invalid operation

        // Assemble result according to destination format
        assign fp_special_result[i] = in_a_type_s2[i].is_zero ? (32'(input_sign_s2) << 31) // signed zero
                                                              : {1'b0, QNAN_EXPONENT, QNAN_MANTISSA}; // qNaN
    end

    // INT Special case handling

    reg [LANES-1:0][31:0]   int_special_result;
    fflags_t [LANES-1:0]    int_special_status;
    wire [LANES-1:0]        int_result_is_special;

    for (genvar i = 0; i < LANES; ++i) begin
         // Assemble result according to destination format
        always @(*) begin
            if (input_sign_s2[i] && !in_a_type_s2[i].is_nan) begin
                int_special_result[i][30:0] = 0;               // alone yields 2**(31)-1
                int_special_result[i][31]   = ~unsigned_s2;    // for unsigned casts yields 2**31
            end else begin
                int_special_result[i][30:0] = 2**(31) -1;      // alone yields 2**(31)-1
                int_special_result[i][31]   = unsigned_s2;     // for unsigned casts yields 2**31
            end
        end            

        // Detect special case from source format (inf, nan, overflow, nan-boxing or negative unsigned)
        assign int_result_is_special[i] = in_a_type_s2[i].is_nan 
                                        | in_a_type_s2[i].is_inf 
                                        | of_before_round[i] 
                                        | (input_sign_s2[i] & unsigned_s2 & ~rounded_int_res_zero[i]);
                                        
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

        wire inexact = is_itof_s2 ? (| fp_round_sticky_bits[i]) // overflow is invalid in i2f;        
                                  : (| fp_round_sticky_bits[i]) | (~in_a_type_s2[i].is_inf & (of_before_round[i] | of_after_round[i]));
                                  
        assign fp_regular_status.NV = is_itof_s2 & (of_before_round[i] | of_after_round[i]); // overflow is invalid for I2F casts
        assign fp_regular_status.DZ = 1'b0; // no divisions
        assign fp_regular_status.OF = ~is_itof_s2 & (~in_a_type_s2[i].is_inf & (of_before_round[i] | of_after_round[i])); // inf casts no OF
        assign fp_regular_status.UF = uf_after_round[i] & inexact;
        assign fp_regular_status.NX = inexact;

        assign int_regular_status = (| int_round_sticky_bits[i]) ? {4'h0, 1'b1} : 5'h0;

        assign fp_result  = fp_result_is_special[i]  ? fp_special_result[i]  : fmt_result[i];        
        assign int_result = int_result_is_special[i] ? int_special_result[i] : rounded_int_res[i];

        assign fp_status  = fp_result_is_special[i]  ? fp_special_status[i]  : fp_regular_status;
        assign int_status = int_result_is_special[i] ? int_special_status[i] : int_regular_status;

        // Select output depending on special case detection
        assign tmp_result[i] = is_itof_s2 ? fp_result : int_result;
        assign tmp_fflags[i] = is_itof_s2 ? fp_status : int_status;
    end

    assign stall = ~ready_out && valid_out;

    VX_pipe_register #(
        .DATAW  (1 + TAGW + (LANES * 32) + (LANES * `FFG_BITS)),
        .RESETW (1)
    ) pipe_reg3 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_in_s2, tag_in_s2, tmp_result, tmp_fflags}),
        .data_out ({valid_out,   tag_out,   result,     fflags})
    );

    assign ready_in = ~stall;

    assign has_fflags = 1'b1;

endmodule
