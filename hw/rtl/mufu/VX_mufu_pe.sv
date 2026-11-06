`include "VX_define.vh"

module VX_mufu_pe import VX_gpu_pkg::*; (
    input wire          clk,
    input wire          reset,
    input wire          enable,
    input wire [2:0]    fn,
    input wire [31:0]   dataa,
    output wire [31:0]  result
);
    `STATIC_ASSERT (`VX_MUFU_LATENCY == 7, ("VX_mufu_pe: the datapath is 7 stages, VX_MUFU_LATENCY=%0d", `VX_MUFU_LATENCY))

    localparam TBL_RCP   = 3'd0;
    localparam TBL_RSQRT = 3'd1;
    localparam TBL_EX2   = 3'd2;
    localparam TBL_LG2   = 3'd3;
    localparam TBL_SIN   = 3'd4;
    localparam TBL_TANH  = 3'd5;

    localparam F32_QNAN    = 32'h7fc00000;
    localparam F32_POS_INF = 32'h7f800000;
    localparam F32_NEG_INF = 32'hff800000;
    localparam F32_ONE     = 32'h3f800000;

    wire        x_s = dataa[31];
    wire [7:0]  x_e = dataa[30:23];
    wire [22:0] x_m = dataa[22:0];
    wire [23:0] x_sig = {1'b1, x_m};

    wire x_zero = (x_e == 8'd0);
    wire x_inf  = (x_e == 8'hff) && (x_m == 23'd0);
    wire x_nan  = (x_e == 8'hff) && (x_m != 23'd0);

    wire signed [9:0] x_k = 10'($signed({2'b00, x_e})) - 10'sd127;

    wire [3:0] ex2_lsh = 4'(x_e - 8'd120);
    wire [7:0] ex2_rsh = 8'd120 - x_e;
    wire [37:0] ex2_mag = (x_e >= 8'd120) ? (38'(x_sig) << ex2_lsh)
                        : ((ex2_rsh < 8'd24) ? 38'(x_sig >> ex2_rsh[4:0]) : 38'd0);
    wire [38:0] ex2_x = x_s ? -{1'b0, ex2_mag} : {1'b0, ex2_mag};

    wire [7:0] trig_lsh = x_e - 8'd119;
    wire [7:0] trig_rsh = 8'd119 - x_e;
    wire is_sin = (fn == 3'(INST_MUFU_SINPI));
    wire [31:0] trig_u0 = (x_e >= 8'd119) ? ((trig_lsh < 8'd32) ? (32'(x_sig) << trig_lsh[4:0]) : 32'd0)
                        : ((trig_rsh < 8'd24) ? 32'(x_sig >> trig_rsh[4:0]) : 32'd0);
    wire [31:0] trig_u = is_sin ? trig_u0 : (trig_u0 + 32'h40000000);
    wire        trig_odd = trig_u[30];
    wire [29:0] trig_t = trig_odd ? ~trig_u[29:0] : trig_u[29:0];
    wire        trig_neg = trig_u[31] ^ (is_sin && x_s);
    wire        trig_one = trig_odd && (trig_u[29:0] == 30'd0);
    `UNUSED_VAR ({ex2_x[5:0], trig_t[5:0]})

    reg         r_special;
    reg [31:0]  r_sval;
    reg [2:0]   r_tbl;
    reg [6:0]   r_idx;
    reg [17:0]  r_xl;
    reg signed [9:0] r_exp;
    reg         r_neg;
    reg         r_zsign;
    reg         r_lg2;

    always @(*) begin
        r_special = 1'b0;
        r_sval    = 32'd0;
        r_tbl     = TBL_RCP;
        r_idx     = x_m[22:16];
        r_xl      = {x_m[15:0], 2'b00};
        r_exp     = 10'sd0;
        r_neg     = 1'b0;
        r_zsign   = 1'b0;
        r_lg2     = 1'b0;
        case (fn)
        3'(INST_MUFU_RCP): begin
            r_neg = x_s;
            r_exp = 10'sd127 - 10'($signed({2'b00, x_e}));
            if (x_inf) begin
                r_special = 1'b1;
                r_sval = {x_s, 31'd0};
            end else if (x_zero) begin
                r_special = 1'b1;
                r_sval = x_s ? F32_NEG_INF : F32_POS_INF;
            end
        end
        3'(INST_MUFU_RSQRT): begin
            r_tbl = TBL_RSQRT;
            r_idx = {x_k[0], x_m[22:17]};
            r_xl  = {x_m[16:0], 1'b0};
            r_exp = -(x_k >>> 1);
            if (x_zero) begin
                r_special = 1'b1;
                r_sval = x_s ? F32_NEG_INF : F32_POS_INF;
            end else if (x_s) begin
                r_special = 1'b1;
                r_sval = F32_QNAN;
            end else if (x_inf) begin
                r_special = 1'b1;
                r_sval = 32'd0;
            end
        end
        3'(INST_MUFU_EX2): begin
            r_tbl = TBL_EX2;
            r_idx = {1'b0, ex2_x[29:24]};
            r_xl  = ex2_x[23:6];
            r_exp = 10'($signed(ex2_x[38:30]));
            if (x_inf) begin
                r_special = 1'b1;
                r_sval = x_s ? 32'd0 : F32_POS_INF;
            end else if (x_zero) begin
                r_special = 1'b1;
                r_sval = F32_ONE;
            end else if (x_e >= 8'd134) begin
                r_special = 1'b1;
                r_sval = x_s ? 32'd0 : F32_POS_INF;
            end
        end
        3'(INST_MUFU_LG2): begin
            r_tbl = TBL_LG2;
            r_idx = {1'b0, x_m[22:17]};
            r_xl  = {x_m[16:0], 1'b0};
            r_lg2 = 1'b1;
            if (x_zero) begin
                r_special = 1'b1;
                r_sval = F32_NEG_INF;
            end else if (x_s) begin
                r_special = 1'b1;
                r_sval = F32_QNAN;
            end else if (x_inf) begin
                r_special = 1'b1;
                r_sval = F32_POS_INF;
            end
        end
        3'(INST_MUFU_SINPI),
        3'(INST_MUFU_COSPI): begin
            r_tbl   = TBL_SIN;
            r_idx   = {1'b0, trig_t[29:24]};
            r_xl    = trig_t[23:6];
            r_neg   = trig_neg;
            r_zsign = is_sin && x_s;
            if (x_inf) begin
                r_special = 1'b1;
                r_sval = F32_QNAN;
            end else if (x_zero) begin
                r_special = 1'b1;
                r_sval = is_sin ? {x_s, 31'd0} : F32_ONE;
            end else if (trig_one) begin
                r_special = 1'b1;
                r_sval = {trig_neg, F32_ONE[30:0]};
            end
        end
        3'(INST_MUFU_TANH): begin
            r_tbl = TBL_TANH;
            r_idx = {1'b0, 3'(x_e - 8'd122), x_m[22:20]};
            r_xl  = x_m[19:2];
            r_exp = x_k + 10'sd4;
            r_neg = x_s;
            if (x_e >= 8'd130) begin
                r_special = 1'b1;
                r_sval = {x_s, F32_ONE[30:0]};
            end else if (x_zero) begin
                r_special = 1'b1;
                r_sval = {x_s, 31'd0};
            end else if (x_e < 8'd122) begin
                r_special = 1'b1;
                r_sval = dataa;
            end
        end
        default: begin
            r_special = 1'b1;
            r_sval = 32'd0;
        end
        endcase
        if (x_nan) begin
            r_special = 1'b1;
            r_sval = F32_QNAN;
        end
    end

    localparam MISC_W = 1 + 32 + 10 + 1 + 1 + 1 + 10;

    wire [MISC_W-1:0] s1_misc, s2_misc, s3_misc, s4_misc;
    wire [2:0]  s1_tbl;
    wire [6:0]  s1_idx;
    wire [17:0] s1_xl, s2_xl, s3_xl;

    VX_pipe_register #(
        .DATAW (MISC_W + 3 + 7 + 18)
    ) s1_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({r_special, r_sval, r_exp, r_neg, r_zsign, r_lg2, x_k, r_tbl,  r_idx,  r_xl}),
        .data_out ({s1_misc,                                          s1_tbl, s1_idx, s1_xl})
    );

    wire [31:0]        rom_a0;
    wire signed [26:0] rom_a1;
    wire signed [19:0] rom_a2;

    VX_mufu_rom rom (
        .clk    (clk),
        .enable (enable),
        .tbl    (s1_tbl),
        .idx    (s1_idx),
        .a0     (rom_a0),
        .a1     (rom_a1),
        .a2     (rom_a2)
    );

    VX_pipe_register #(
        .DATAW (MISC_W + 18)
    ) s2_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({s1_misc, s1_xl}),
        .data_out ({s2_misc, s2_xl})
    );

    wire [35:0] xl_sq = 36'(s2_xl) * 36'(s2_xl);
    `UNUSED_VAR (xl_sq[17:0])

    wire [17:0] s3_sq;

    VX_pipe_register #(
        .DATAW (MISC_W + 18 + 18)
    ) s3_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({s2_misc, s2_xl, xl_sq[35:18]}),
        .data_out ({s3_misc, s3_xl, s3_sq})
    );

    wire signed [45:0] s4_p1;
    wire signed [38:0] s4_p2;

    VX_multiplier #(
        .A_WIDTH (27),
        .B_WIDTH (19),
        .R_WIDTH (46),
        .SIGNED  (1),
        .LATENCY (1)
    ) mul1 (
        .clk    (clk),
        .enable (enable),
        .dataa  (rom_a1),
        .datab  ({1'b0, s3_xl}),
        .result (s4_p1)
    );

    VX_multiplier #(
        .A_WIDTH (20),
        .B_WIDTH (19),
        .R_WIDTH (39),
        .SIGNED  (1),
        .LATENCY (1)
    ) mul2 (
        .clk    (clk),
        .enable (enable),
        .dataa  (rom_a2),
        .datab  ({1'b0, s3_sq}),
        .result (s4_p2)
    );

    wire [31:0] s4_a0;

    VX_pipe_register #(
        .DATAW (MISC_W + 32)
    ) s4_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({s3_misc, rom_a0}),
        .data_out ({s4_misc, s4_a0})
    );

    wire        s4_special, s4_neg, s4_zsign, s4_lg2;
    wire [31:0] s4_sval;
    wire signed [9:0] s4_exp, s4_k;
    assign {s4_special, s4_sval, s4_exp, s4_neg, s4_zsign, s4_lg2, s4_k} = s4_misc;

    wire signed [27:0] t1 = 28'(s4_p1 >>> 18);
    wire signed [20:0] t2 = 21'(s4_p2 >>> 18);
    wire signed [34:0] p_sum = $signed({3'b000, s4_a0}) + 35'(t1) + 35'(t2);
    wire [33:0] p_pos = p_sum[34] ? 34'd0 : p_sum[33:0];
    wire signed [40:0] lg2_sum = 41'($signed({s4_k, 30'd0})) + $signed({7'd0, p_pos});
    wire lg2_neg = lg2_sum[40];
    wire [39:0] lg2_mag = 40'(lg2_neg ? -lg2_sum : lg2_sum);

    wire [39:0] mag = s4_lg2 ? lg2_mag : 40'(p_pos);
    wire        neg = s4_lg2 ? lg2_neg : s4_neg;
    wire signed [9:0] exp_adj = s4_lg2 ? 10'sd0 : s4_exp;
    wire zsign = s4_lg2 ? 1'b0 : s4_zsign;

    wire        s5_special, s5_neg, s5_zsign;
    wire [31:0] s5_sval;
    wire signed [9:0] s5_exp;
    wire [39:0] s5_mag;

    VX_pipe_register #(
        .DATAW (1 + 32 + 10 + 1 + 1 + 40)
    ) s5_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({s4_special, s4_sval, exp_adj, neg,    zsign,    mag}),
        .data_out ({s5_special, s5_sval, s5_exp,  s5_neg, s5_zsign, s5_mag})
    );

    wire [5:0] lz;
    wire mag_nz;

    VX_lzc #(
        .N (40)
    ) lzc (
        .data_in   (s5_mag),
        .data_out  (lz),
        .valid_out (mag_nz)
    );

    wire [39:0] norm = s5_mag << lz;
    wire signed [11:0] bexp_n = 12'sd136 - 12'($signed({1'b0, lz})) + 12'(s5_exp);
    `UNUSED_VAR (norm[14:0])

    wire        s6_special, s6_neg, s6_zsign, s6_nz;
    wire [31:0] s6_sval;
    wire signed [11:0] s6_bexp;
    wire [24:0] s6_sig;

    VX_pipe_register #(
        .DATAW (1 + 32 + 1 + 1 + 1 + 12 + 25)
    ) s6_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({s5_special, s5_sval, s5_neg, s5_zsign, mag_nz, bexp_n,  norm[39:15]}),
        .data_out ({s6_special, s6_sval, s6_neg, s6_zsign, s6_nz,  s6_bexp, s6_sig})
    );

    wire [24:0] sig_r = {1'b0, s6_sig[24:1]} + 25'(s6_sig[0]);
    wire signed [11:0] bexp = s6_bexp + 12'(sig_r[24]);
    wire [22:0] mant = sig_r[24] ? sig_r[23:1] : sig_r[22:0];

    reg [31:0] packed_res;
    always @(*) begin
        if (s6_special) begin
            packed_res = s6_sval;
        end else if (~s6_nz) begin
            packed_res = {s6_zsign, 31'd0};
        end else if (bexp >= 12'sd255) begin
            packed_res = {s6_neg, F32_POS_INF[30:0]};
        end else if (bexp <= 12'sd0) begin
            packed_res = {s6_neg, 31'd0};
        end else begin
            packed_res = {s6_neg, bexp[7:0], mant};
        end
    end

    VX_pipe_register #(
        .DATAW (32)
    ) s7_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (packed_res),
        .data_out (result)
    );

endmodule
