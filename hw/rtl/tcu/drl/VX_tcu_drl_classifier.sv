`include "VX_define.vh"

module VX_tcu_drl_classifier import VX_tcu_pkg::*; #(
    parameter N     = 1,
    parameter WIDTH = 32,
    parameter FMT   = 0
) (
    input wire [N-1:0][WIDTH-1:0] val,
    output fedp_class_t [N-1:0]   cls
);
    localparam EXP_BITS = VX_tcu_pkg::exp_bits(FMT);
    localparam MAN_BITS = VX_tcu_pkg::sig_bits(FMT);
    localparam SIGN_POS = VX_tcu_pkg::sign_pos(FMT);

    localparam EXP_START = SIGN_POS - 1;
    localparam EXP_END   = EXP_START - EXP_BITS + 1;

    localparam MAN_START = EXP_END - 1;
    localparam MAN_END   = MAN_START - MAN_BITS + 1;

    localparam logic [EXP_BITS-1:0] EXP_ZERO = '0;
    localparam logic [EXP_BITS-1:0] EXP_ONES = '1;
    localparam logic [MAN_BITS-1:0] MAN_ZERO = '0;

    for (genvar i = 0; i < N; ++i) begin : g_cls
        wire sign = val[i][SIGN_POS];
        wire [EXP_BITS-1:0] exp  = val[i][EXP_START : EXP_END];
        wire [MAN_BITS-1:0] man  = val[i][MAN_START : MAN_END];

        assign cls[i].sign    = sign;
        assign cls[i].is_zero = (exp == EXP_ZERO) && (man == MAN_ZERO);
        assign cls[i].is_sub  = (exp == EXP_ZERO) && (man != MAN_ZERO);
        assign cls[i].is_inf  = (exp == EXP_ONES) && (man == MAN_ZERO);
        assign cls[i].is_nan  = (exp == EXP_ONES) && (man != MAN_ZERO);
    end

endmodule
