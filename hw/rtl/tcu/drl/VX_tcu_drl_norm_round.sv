`include "VX_define.vh"

module VX_tcu_drl_norm_round import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N = 5,
    parameter W = 25,
    parameter WA = 30,
    parameter EXP_W = 10,
    parameter C_HI_W = 8
) (
    input wire          clk,
    input wire          valid_in,
    input wire [31:0]   req_id,
    input wire [EXP_W-1:0] max_exp,
    input wire [WA-1:0] acc_sig,
    input wire [C_HI_W-1:0] cval_hi,
    input wire [N-2:0]  sig_signs,
    input wire          is_int,
    input wire          sticky_in,
    input fedp_excep_t  exceptions,
    output wire [31:0]  result
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in})

    // ----------------------------------------------------------------------
    // 1. Signed Magnitude Extraction
    // ----------------------------------------------------------------------
    // The accumulator is in 2's complement.
    wire             sum_sign = acc_sig[WA-1];
    wire [WA-1:0]    abs_sum  = sum_sign ? -acc_sig : acc_sig;
    wire             zero_sum = ~|abs_sum;

    // ----------------------------------------------------------------------
    // 2. Leading Zero Count (LZC)
    // ----------------------------------------------------------------------
    wire [$clog2(WA)-1:0] lz_count;
    VX_lzc #(
        .N (WA)
    ) lzc_inst (
        .data_in   (abs_sum),
        .data_out  (lz_count),
        `UNUSED_PIN (valid_out)
    );

    // ----------------------------------------------------------------------
    // 3. Normalization Shifter
    // ----------------------------------------------------------------------
    localparam HR = WA - W;

    // We want to shift left by lz_count to normalize
    wire [WA-1:0] shifted_sum;
    assign shifted_sum = abs_sum << lz_count;

    // ----------------------------------------------------------------------
    // 4. Exponent Adjustment & Rounding Bits
    // ----------------------------------------------------------------------
    // Extract Normalized Mantissa (1 hidden + 23 fraction)
    // Top bit of shifted_sum is at [WA-1]
    wire [23:0] norm_man = shifted_sum[WA-1 -: 24];

    // Bits below the mantissa
    wire        guard_bit = shifted_sum[WA-25];
    wire        round_bit = shifted_sum[WA-26];
    wire        sticky_rem = |shifted_sum[WA-27 : 0];
    wire        sticky_bit = sticky_rem | sticky_in;

    // Calculate Exponent
    // shift_adj = HR - lzc.
    wire signed [9:0] shift_adj = 10'(HR) - 10'(lz_count);
    wire signed [9:0] norm_exp_s = $signed(max_exp) + shift_adj + 10'(W - 1);

    // ----------------------------------------------------------------------
    // 5. Rounding (RNE - Round to Nearest Even)
    // ----------------------------------------------------------------------
    wire lsb_bit  = norm_man[0];
    wire round_up = guard_bit && (round_bit || sticky_bit || lsb_bit);

    // Add round bit to mantissa
    wire [24:0] rounded_sig_full = {1'b0, norm_man} + 25'(round_up);

    // Check for carry out after rounding (e.g. 1.11...1 + 1 = 10.00...0)
    wire carry_out = rounded_sig_full[24];

    // Final Mantissa (23 bits)
    // If carry_out, we shift right by 1 (exp increments).
    wire [22:0] final_man = carry_out ? rounded_sig_full[23:1] : rounded_sig_full[22:0];

    // Final Exponent
    wire signed [9:0] final_exp_s = norm_exp_s + 10'(carry_out);

    // ----------------------------------------------------------------------
    // 6. Exception & Result Packing
    // ----------------------------------------------------------------------
    logic [7:0] packed_exp;
    logic       exp_overflow, exp_underflow;

    always_comb begin
        if (final_exp_s >= 255) begin
            packed_exp = 8'hFF;
            exp_overflow = 1'b1;
            exp_underflow = 1'b0;
        end else if (final_exp_s <= 0) begin
            packed_exp = 8'h00;
            exp_overflow = 1'b0;
            exp_underflow = 1'b1;
        end else begin
            packed_exp = final_exp_s[7:0];
            exp_overflow = 1'b0;
            exp_underflow = 1'b0;
        end
    end

    logic [31:0] fp_result;
    always_comb begin
        if (exceptions.is_nan) begin
            // qNaN
            fp_result = {1'b0, 8'hFF, 1'b1, 22'd0};
        end else if (exceptions.is_inf) begin
            // Inf
            fp_result = {exceptions.sign, 8'hFF, 23'd0};
        end else begin
            // Normal / Zero / Overflow / Underflow
            if (zero_sum || exp_underflow) begin
                 fp_result = {sum_sign, 8'd0, 23'd0};
            end else if (exp_overflow) begin
                 fp_result = {sum_sign, 8'hFF, 23'd0};
            end else begin
                 fp_result = {sum_sign, packed_exp, final_man};
            end
        end
    end

    // ----------------------------------------------------------------------
    // 7. Integer Handling
    // ----------------------------------------------------------------------

    `UNUSED_VAR (sig_signs)

    // Extract high part of accumulator
    wire [6:0] ext_acc_int = 7'($signed(acc_sig[WA-1:W]));
    
    wire [6:0] int_hi;
    VX_ks_adder #(
        .N (7)
    ) int_adder (
        .dataa (ext_acc_int),
        .datab (cval_hi),
        .sum   (int_hi),
        `UNUSED_PIN (cout)
    );

    // Concatenate high integer part with lower accumulator bits
    wire [31:0] int_result = {int_hi, acc_sig[24:0]};

    assign result = is_int ? int_result : fp_result;

`ifdef DBG_TRACE_TCU
    always_ff @(posedge clk) begin
        if (valid_in) begin
            `TRACE(4, ("%t: %s FEDP-NORM(%0d): is_int=%b, acc_sig=0x%0h, sign=%b, lzc=%0d, norm_exp=%0d, shifted=0x%0h, R=%b, S=%b, Rup=%b, carry=%b, final_exp=%0d, result=0x%0h\n",
                $time, INSTANCE_ID, req_id, is_int, acc_sig, sum_sign, lz_count, norm_exp_s, shifted_sum, round_bit, sticky_bit, round_up, carry_out, final_exp_s, result));
        end
    end
`endif

endmodule
