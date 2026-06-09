// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0

`include "VX_define.vh"

module VX_tcu_tfr_exc_reduce import VX_tcu_pkg::*; #(
    parameter TCK = 4
) (
    input  fedp_excep_t [TCK:0] exc_in,
    output fedp_excep_t         exc_out
);
    // Unpack lane exceptions
    logic [TCK-1:0] lane_nan;
    logic [TCK-1:0] lane_inf;
    logic [TCK-1:0] lane_sign;

    for (genvar i = 0; i < TCK; ++i) begin : g_unpack
        assign lane_nan[i]  = exc_in[i].is_nan;
        assign lane_inf[i]  = exc_in[i].is_inf;
        assign lane_sign[i] = exc_in[i].sign;
    end

    // C-term exception (index TCK)
    wire c_is_nan = exc_in[TCK].is_nan;
    wire c_is_inf = exc_in[TCK].is_inf;
    wire c_sign   = exc_in[TCK].sign;

    // Infinity sign analysis
    wire [TCK-1:0] p_pos_inf = lane_inf & ~lane_sign;
    wire [TCK-1:0] p_neg_inf = lane_inf &  lane_sign;

    wire has_pos = (|p_pos_inf) | (c_is_inf & ~c_sign);
    wire has_neg = (|p_neg_inf) | (c_is_inf &  c_sign);

    // NaN: any input NaN, or +Inf + -Inf
    wire any_input_nan = (|lane_nan) | c_is_nan;
    wire res_nan = any_input_nan | (has_pos & has_neg);

    // Inf: any infinity, but not NaN
    wire res_inf = (has_pos | has_neg) & ~res_nan;

    // Sign
    wire res_sign = has_neg & ~has_pos;

    assign exc_out.is_nan = res_nan;
    assign exc_out.is_inf = res_inf;
    assign exc_out.sign   = res_sign;

endmodule
