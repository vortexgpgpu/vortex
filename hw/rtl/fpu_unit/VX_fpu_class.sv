
`include "VX_fpu_define.vh"

module VX_fpu_class # (    
    parameter MAN_BITS = 23,
    parameter EXP_BITS = 8
) (
    input  [EXP_BITS-1:0] exp_i,
    input  [MAN_BITS-1:0] man_i,
    output fclass_t       fclass_o
);
    wire is_normal    = (exp_i != '0) && (exp_i != '1);
    wire is_zero      = (exp_i == '0) && (man_i == '0);
    wire is_subnormal = (exp_i == '0) && (man_i != '0);
    wire is_inf       = (exp_i == '1) && (man_i == '0); 
    wire is_nan       = (exp_i == '1) && (man_i != '0);
    wire is_signaling = is_nan && ~man_i[MAN_BITS-1];
    wire is_quiet     = is_nan && ~is_signaling;

    assign fclass_o.is_normal    = is_normal;
    assign fclass_o.is_zero      = is_zero;
    assign fclass_o.is_subnormal = is_subnormal;
    assign fclass_o.is_inf       = is_inf;
    assign fclass_o.is_nan       = is_nan;
    assign fclass_o.is_quiet     = is_quiet;
    assign fclass_o.is_signaling = is_signaling;

endmodule
