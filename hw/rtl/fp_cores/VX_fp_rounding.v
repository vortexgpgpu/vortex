`include "VX_fpu_define.vh"

/// Modified port of rouding module from fpnew Libray
/// reference: https://github.com/pulp-platform/fpnew

module VX_fp_rounding #(
    parameter DAT_WIDTH = 2 // Width of the abolute value, without sign bit
) (
    // inputs
    input wire [DAT_WIDTH-1:0]  abs_value_i,             // absolute value without sign
    input wire                  sign_i,
    // rounding information
    input wire [1:0]            round_sticky_bits_i,     // round and sticky bits {RS}
    input wire [2:0]            rnd_mode_i,
    input wire                  effective_subtraction_i, // sign of inputs affects rounding of zeroes
    // outputs
    output wire [DAT_WIDTH-1:0] abs_rounded_o,           // absolute value without sign
    output wire                 sign_o,
    output wire                 exact_zero_o             // output is an exact zero
);

    reg round_up; // Rounding decision

    // Take the rounding decision according to RISC-V spec
    // RoundMode | Mnemonic | Meaning
    // :--------:|:--------:|:-------
    //    000    |   RNE    | Round to Nearest, ties to Even
    //    001    |   RTZ    | Round towards Zero
    //    010    |   RDN    | Round Down (towards -\infty)
    //    011    |   RUP    | Round Up (towards \infty)
    //    100    |   RMM    | Round to Nearest, ties to Max Magnitude
    //  others   |          | *invalid*

    always @(*) begin
        case (rnd_mode_i)
            `INST_FRM_RNE: // Decide accoring to round/sticky bits
                case (round_sticky_bits_i)
                      2'b00, 
                      2'b01: round_up = 1'b0;            // < ulp/2 away, round down
                      2'b10: round_up = abs_value_i[0];  // = ulp/2 away, round towards even result
                      2'b11: round_up = 1'b1;            // > ulp/2 away, round up
                    default: round_up = 1'bx;
                endcase
            `INST_FRM_RTZ: round_up = 1'b0; // always round down
            `INST_FRM_RDN: round_up = (| round_sticky_bits_i) & sign_i;  // to 0 if +, away if -
            `INST_FRM_RUP: round_up = (| round_sticky_bits_i) & ~sign_i; // to 0 if -, away if +
            `INST_FRM_RMM: round_up = round_sticky_bits_i[1]; // round down if < ulp/2 away, else up
            default:  round_up = 1'bx; // propagate x
        endcase
    end

    // Perform the rounding, exponent change and overflow to inf happens automagically
    assign abs_rounded_o = abs_value_i + DAT_WIDTH'(round_up);

    // True zero result is a zero result without dirty round/sticky bits
    assign exact_zero_o = (abs_value_i == 0) && (round_sticky_bits_i == 0);

    // In case of effective subtraction (thus signs of addition operands must have differed) and a
    // true zero result, the result sign is '-' in case of RDN and '+' for other modes.
    assign sign_o = (exact_zero_o && effective_subtraction_i) ? (rnd_mode_i == `INST_FRM_RDN)
                                                              : sign_i;

endmodule