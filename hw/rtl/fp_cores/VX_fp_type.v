
`include "VX_define.vh"

module VX_fp_type (
    // inputs
    input  [7:0]  exponent,
    input  [22:0] mantissa,
    // outputs
    output fp_type_t o_type
);
    wire is_normal    = (exponent != 8'd0) && (exponent != 8'hff);
    wire is_zero      = (exponent == 8'd0) && (mantissa == 23'd0);
    wire is_subnormal = (exponent == 8'd0) && !is_zero;
    wire is_inf       = (exponent == 8'hff) && (mantissa == 23'd0); 
    wire is_nan       = (exponent == 8'hff) && (mantissa != 23'd0);
    wire is_signaling = is_nan && (mantissa[22] == 1'b0);
    wire is_quiet     = is_nan && !is_signaling;

    assign o_type.is_normal    = is_normal;
    assign o_type.is_zero      = is_zero;
    assign o_type.is_subnormal = is_subnormal;
    assign o_type.is_inf       = is_inf;
    assign o_type.is_nan       = is_nan;
    assign o_type.is_signaling = is_signaling;
    assign o_type.is_quiet     = is_quiet;

endmodule