
`include "VX_define.vh"

module VX_fp_type (
	// inputs 
    input  [7:0]  exponent,
    input  [22:0] mantissa,
    // outputs
    output fp_type_t o_type
);
    assign o_type.is_normal    = (exponent != 8'd0) && (exponent != 8'hff);
    assign o_type.is_zero      = (exponent == 8'd0) && (mantissa == 23'd0);
    assign o_type.is_subnormal = (exponent == 8'd0) && !o_type.is_zero;
    assign o_type.is_inf       = (exponent == 8'hff) && (mantissa == 23'd0);
    assign o_type.is_nan       = (exponent == 8'hff) && (mantissa != 23'd0);
    assign o_type.is_signaling = o_type.is_nan && (mantissa[22] == 1'b0);
    assign o_type.is_quiet     = o_type.is_nan && !o_type.is_signaling;

endmodule