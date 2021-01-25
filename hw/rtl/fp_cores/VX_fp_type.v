
`include "VX_define.vh"

module VX_fp_type (
    // inputs
    input  [7:0]  exp_i,
    input  [22:0] man_i,
    // outputs
    output fp_type_t type_o
);
    wire is_normal    = (exp_i != 8'd0) && (exp_i != 8'hff);
    wire is_zero      = (exp_i == 8'd0) && (man_i == 23'd0);
    wire is_subnormal = (exp_i == 8'd0) && !is_zero;
    wire is_inf       = (exp_i == 8'hff) && (man_i == 23'd0); 
    wire is_nan       = (exp_i == 8'hff) && (man_i != 23'd0);
    wire is_signaling = is_nan && (man_i[22] == 1'b0);
    wire is_quiet     = is_nan && !is_signaling;

    assign type_o.is_normal    = is_normal;
    assign type_o.is_zero      = is_zero;
    assign type_o.is_subnormal = is_subnormal;
    assign type_o.is_inf       = is_inf;
    assign type_o.is_nan       = is_nan;
    assign type_o.is_quiet     = is_quiet;
    assign type_o.is_signaling = is_signaling;

endmodule