// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// VX_rtu_tri_pe — pipelined watertight ray-triangle intersector (Woop, Benthin,
// Wald, "Watertight Ray/Triangle Intersection", JCGT 2013), all F32. Streams one
// triangle per cycle and emits {hit, t, u, v, back_facing} after a fixed
// latency. Mirrors SimX rtu::ray_triangle op for op:
//
//   kz = argmax|dir|, kx/ky follow (swapped when dir[kz] < 0)
//   sz = 1/dir[kz], sx = dir[kx]*sz, sy = dir[ky]*sz
//   r = vertex - origin, px = fma(-sx, rz, rx), py = fma(-sy, rz, ry), pz = sz*rz
//   w0 = px2*py1 - py2*px1, w1 = px0*py2 - py0*px2, w2 = px1*py0 - py1*px0
//   det = (w0 + w1) + w2, T = fma(w2, pz2, fma(w1, pz1, w0*pz0))
//   rcp = 1/det, t = T*rcp, (u, v) = (w1*rcp, w2*rcp)
//   hit = !(any w < 0 && any w > 0) && det != 0 && t_min < t < t_max
//   back_facing = det < 0
//
// Each edge function is two rounded products and a rounded difference of the
// sheared vertices alone, so an edge shared by two triangles evaluates to
// exactly negated weights in both: no ray slips between them. t_min < t < t_max
// is the Vulkan ray interval for triangles (open at both ends). The FP units
// flush subnormals.

`include "VX_define.vh"

module VX_rtu_tri_pe import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter LATENCY_FMA  = RTU_LATENCY_FMA,
    parameter LATENCY_FDIV = RTU_FDIV_LAT,
    parameter TAG_WIDTH    = 1
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             enable,
    input  wire             valid_in,
    input  wire [TAG_WIDTH-1:0] tag_in,    // caller side-band (e.g. context id)

    input  wire [2:0][31:0] origin,
    input  wire [2:0][31:0] dir,
    input  wire [2:0][31:0] v0,
    input  wire [2:0][31:0] v1,
    input  wire [2:0][31:0] v2,
    input  wire [31:0]      t_min,
    input  wire [31:0]      t_max,

    output wire             valid_out,
    output wire [TAG_WIDTH-1:0] tag_out,
    output wire             hit,
    output wire [31:0]      t,
    output wire [31:0]      u,
    output wire [31:0]      v,
    output wire             back_facing
);
    localparam F = LATENCY_FMA;
    localparam V = LATENCY_FDIV;

    // stage start times (cycles after valid_in)
    localparam T_B = 1;                 // axis select registered
    localparam T_C = T_B + V;           // sz ready
    localparam T_D = T_C + F;           // sx, sy ready
    localparam T_E = T_D + F;           // px, py, pz ready
    localparam T_F = T_E + 2 * F;       // w ready
    localparam T_G = T_F + 2 * F + V;   // 1/det ready (T at T_F + 3F)
    localparam T_H = T_G + F;           // t, u, v ready
    localparam LATENCY = T_H + 1;       // verdict registered

    `STATIC_ASSERT(V >= F, ("tri PE: FDIV latency must cover the r subtract and T"))

    localparam [INST_FMT_BITS-1:0] FMT_ADD = 2'b00;
    localparam [INST_FMT_BITS-1:0] FMT_SUB = 2'b10;
    localparam [31:0] F32_ONE = 32'h3F800000;

    // IEEE ordering on F32 (+0 == -0); NaN compares false
    function automatic logic f32_lt(input logic [31:0] a, input logic [31:0] b);
        logic [31:0] ka, kb;
        ka = (a[30:0] == 31'd0) ? 32'h80000000 : (a[31] ? ~a : {1'b1, a[30:0]});
        kb = (b[30:0] == 31'd0) ? 32'h80000000 : (b[31] ? ~b : {1'b1, b[30:0]});
        f32_lt = !((a[30:23] == 8'hff) && (a[22:0] != 23'd0))
              && !((b[30:23] == 8'hff) && (b[22:0] != 23'd0))
              && (ka < kb);
    endfunction

    // ── stage A (@0 -> @T_B): axis select ─────────────────────────────
    wire [30:0] ad0 = dir[0][30:0];
    wire [30:0] ad1 = dir[1][30:0];
    wire [30:0] ad2 = dir[2][30:0];
    wire [1:0] kz_w = (ad0 >= ad1) ? ((ad0 >= ad2) ? 2'd0 : 2'd2)
                                   : ((ad1 >= ad2) ? 2'd1 : 2'd2);
    wire [1:0] kx0 = (kz_w == 2'd2) ? 2'd0 : (kz_w + 2'd1);
    wire [1:0] ky0 = (kx0  == 2'd2) ? 2'd0 : (kx0  + 2'd1);
    wire dz_neg = dir[kz_w][31] && (dir[kz_w][30:0] != 31'd0);
    wire [1:0] kx_w = dz_neg ? ky0 : kx0;
    wire [1:0] ky_w = dz_neg ? kx0 : ky0;

    // per vertex: (x, y, z) components in the sheared frame's axes
    wire [2:0][2:0][31:0] q_w;     // [vertex][axis x/y/z]
    wire [2:0][2:0][31:0] cvs = {v2, v1, v0};
    for (genvar i = 0; i < 3; ++i) begin : g_q
        assign q_w[i][0] = cvs[i][kx_w];
        assign q_w[i][1] = cvs[i][ky_w];
        assign q_w[i][2] = cvs[i][kz_w];
    end
    wire [2:0][31:0] o_w = {origin[kz_w], origin[ky_w], origin[kx_w]};
    wire [2:0][31:0] d_w = {dir[kz_w], dir[ky_w], dir[kx_w]};

    reg [2:0][2:0][31:0] q_a;
    reg [2:0][31:0]      o_a, d_a;
    reg [31:0]           tmin_a, tmax_a;
    always_ff @(posedge clk) begin
        if (enable) begin
            q_a    <= q_w;
            o_a    <= o_w;
            d_a    <= d_w;
            tmin_a <= t_min;
            tmax_a <= t_max;
        end
    end

    // ── stage B (@T_B): sz = 1/dir[kz]; r = vertex - origin ───────────
    wire [31:0] sz_c;
    VX_fdiv_unit #(
        .LATENCY        (V),
        .FLEN           (32),
        .USE_DSP        (`VX_CFG_RTU_USE_DSP),
        .SUBNORM_ENABLE (0),
        .EXCEPT_ENABLE  (1)
    ) fdiv_sz (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .mask    (1'b1),
        .fmt     ('0),
        .frm     (INST_FRM_RNE),
        .dataa   (F32_ONE),
        .datab   (d_a[2]),
        .result  (sz_c),
        `UNUSED_PIN (fflags)
    );

    wire [2:0][2:0][31:0] r_f;
    for (genvar i = 0; i < 3; ++i) begin : g_r
        for (genvar a = 0; a < 3; ++a) begin : g_ax
            VX_fma_unit #(
                .LATENCY        (F),
                .USE_DSP        (`VX_CFG_RTU_USE_DSP),
                .SUBNORM_ENABLE (0),
                .EXCEPT_ENABLE  (1)
            ) fsub_r (
                .clk     (clk),
                .reset   (reset),
                .enable  (enable),
                .mask    (1'b1),
                .op_type (INST_FPU_ADD),
                .fmt     (FMT_SUB),
                .frm     (INST_FRM_RNE),
                .dataa   (q_a[i][a]),
                .datab   (o_a[a]),
                .datac   ('0),
                .result  (r_f[i][a]),
                `UNUSED_PIN (fflags)
            );
        end
    end

    // r from @T_B+F to @T_D (consumed by the shear stage)
    wire [2:0][2:0][31:0] r_d;
    VX_shift_register #(
        .DATAW (9 * 32),
        .DEPTH (T_D - (T_B + F))
    ) sr_r (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (r_f),
        .data_out (r_d)
    );

    wire [1:0][31:0] dxy_c;
    VX_shift_register #(
        .DATAW (64),
        .DEPTH (T_C - T_B)
    ) sr_dxy (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({d_a[1], d_a[0]}),
        .data_out (dxy_c)
    );

    // ── stage C (@T_C): sx = dir[kx]*sz, sy = dir[ky]*sz ───────────────
    wire [1:0][31:0] sxy_d;
    for (genvar a = 0; a < 2; ++a) begin : g_sxy
        VX_fma_unit #(
            .LATENCY        (F),
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (1)
        ) fmul_s (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (1'b1),
            .op_type (INST_FPU_MUL),
            .fmt     (FMT_ADD),
            .frm     (INST_FRM_RNE),
            .dataa   (dxy_c[a]),
            .datab   (sz_c),
            .datac   ('0),
            .result  (sxy_d[a]),
            `UNUSED_PIN (fflags)
        );
    end

    wire [31:0] sz_d;
    VX_shift_register #(
        .DATAW (32),
        .DEPTH (T_D - T_C)
    ) sr_sz (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (sz_c),
        .data_out (sz_d)
    );

    // ── stage D (@T_D): px/py = fma(-s, rz, r), pz = sz*rz ────────────
    wire [2:0][1:0][31:0] p_e;   // [vertex][x/y]
    wire [2:0][31:0]      pz_e;
    for (genvar i = 0; i < 3; ++i) begin : g_shear
        for (genvar a = 0; a < 2; ++a) begin : g_ax
            VX_fma_unit #(
                .LATENCY        (F),
                .USE_DSP        (`VX_CFG_RTU_USE_DSP),
                .SUBNORM_ENABLE (0),
                .EXCEPT_ENABLE  (1)
            ) fma_p (
                .clk     (clk),
                .reset   (reset),
                .enable  (enable),
                .mask    (1'b1),
                .op_type (INST_FPU_MADD),
                .fmt     (FMT_ADD),
                .frm     (INST_FRM_RNE),
                .dataa   ({~sxy_d[a][31], sxy_d[a][30:0]}),
                .datab   (r_d[i][2]),
                .datac   (r_d[i][a]),
                .result  (p_e[i][a]),
                `UNUSED_PIN (fflags)
            );
        end
        VX_fma_unit #(
            .LATENCY        (F),
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (1)
        ) fmul_pz (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (1'b1),
            .op_type (INST_FPU_MUL),
            .fmt     (FMT_ADD),
            .frm     (INST_FRM_RNE),
            .dataa   (sz_d),
            .datab   (r_d[i][2]),
            .datac   ('0),
            .result  (pz_e[i]),
            `UNUSED_PIN (fflags)
        );
    end

    // ── stage E (@T_E): w_i = px_a*py_b - py_a*px_b ───────────────────
    // (a, b) per weight: w0 <- (2, 1), w1 <- (0, 2), w2 <- (1, 0)
    wire [2:0][1:0][31:0] cp_e;   // [weight][px_a*py_b, py_a*px_b]
    wire [2:0][31:0]      w_f;
    for (genvar i = 0; i < 3; ++i) begin : g_w
        localparam IA = (i == 0) ? 2 : ((i == 1) ? 0 : 1);
        localparam IB = (i == 0) ? 1 : ((i == 1) ? 2 : 0);
        for (genvar k = 0; k < 2; ++k) begin : g_prod
            VX_fma_unit #(
                .LATENCY        (F),
                .USE_DSP        (`VX_CFG_RTU_USE_DSP),
                .SUBNORM_ENABLE (0),
                .EXCEPT_ENABLE  (1)
            ) fmul_c (
                .clk     (clk),
                .reset   (reset),
                .enable  (enable),
                .mask    (1'b1),
                .op_type (INST_FPU_MUL),
                .fmt     (FMT_ADD),
                .frm     (INST_FRM_RNE),
                .dataa   (p_e[IA][k]),
                .datab   (p_e[IB][1-k]),
                .datac   ('0),
                .result  (cp_e[i][k]),
                `UNUSED_PIN (fflags)
            );
        end
        VX_fma_unit #(
            .LATENCY        (F),
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (1)
        ) fsub_w (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (1'b1),
            .op_type (INST_FPU_ADD),
            .fmt     (FMT_SUB),
            .frm     (INST_FRM_RNE),
            .dataa   (cp_e[i][0]),
            .datab   (cp_e[i][1]),
            .datac   ('0),
            .result  (w_f[i]),
            `UNUSED_PIN (fflags)
        );
    end

    // pz from @T_E to @T_F
    wire [2:0][31:0] pz_f;
    VX_shift_register #(
        .DATAW (3 * 32),
        .DEPTH (T_F - T_E)
    ) sr_pz (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (pz_e),
        .data_out (pz_f)
    );

    // ── stage F (@T_F): det = (w0 + w1) + w2; T = fma chain over w*pz ──
    // w, pz delayed one and two FMA stages for the later chain links
    wire [2:0][31:0] w_f1, pz_f1;
    wire [31:0]      w1_f2, w2_f2, pz2_f2;
    VX_shift_register #(
        .DATAW (6 * 32),
        .DEPTH (F)
    ) sr_wpz1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({w_f, pz_f}),
        .data_out ({w_f1, pz_f1})
    );
    VX_shift_register #(
        .DATAW (3 * 32),
        .DEPTH (F)
    ) sr_wpz2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({w_f1[2], pz_f1[2], w_f1[1]}),
        .data_out ({w2_f2, pz2_f2, w1_f2})
    );

    wire [31:0] det01, det_g, tp0, tp01, t_num;
    VX_fma_unit #(.LATENCY (F), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fadd_det01 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_ADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (w_f[0]), .datab (w_f[1]), .datac ('0),
        .result (det01), `UNUSED_PIN (fflags)
    );
    VX_fma_unit #(.LATENCY (F), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fadd_det (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_ADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (det01), .datab (w_f1[2]), .datac ('0),
        .result (det_g), `UNUSED_PIN (fflags)
    );
    VX_fma_unit #(.LATENCY (F), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fmul_tp0 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MUL), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (w_f[0]), .datab (pz_f[0]), .datac ('0),
        .result (tp0), `UNUSED_PIN (fflags)
    );
    VX_fma_unit #(.LATENCY (F), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fma_tp01 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (w_f1[1]), .datab (pz_f1[1]), .datac (tp0),
        .result (tp01), `UNUSED_PIN (fflags)
    );
    VX_fma_unit #(.LATENCY (F), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fma_t (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (w2_f2), .datab (pz2_f2), .datac (tp01),
        .result (t_num), `UNUSED_PIN (fflags)
    );

    // ── 1/det (@T_F+2F -> @T_G) ───────────────────────────────────────
    wire [31:0] rcp_g;
    VX_fdiv_unit #(
        .LATENCY        (V),
        .FLEN           (32),
        .USE_DSP        (`VX_CFG_RTU_USE_DSP),
        .SUBNORM_ENABLE (0),
        .EXCEPT_ENABLE  (1)
    ) fdiv_rcp (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .mask    (1'b1),
        .fmt     ('0),
        .frm     (INST_FRM_RNE),
        .dataa   (F32_ONE),
        .datab   (det_g),
        .result  (rcp_g),
        `UNUSED_PIN (fflags)
    );

    // T from @T_F+3F, w1/w2 from @T_F+2F, to @T_G
    wire [31:0] t_num_g, w1_g, w2_g;
    VX_shift_register #(
        .DATAW (32),
        .DEPTH (T_G - (T_F + 3 * F))
    ) sr_tnum (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (t_num),
        .data_out (t_num_g)
    );
    VX_shift_register #(
        .DATAW (2 * 32),
        .DEPTH (V)
    ) sr_wuv (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({w2_f2, w1_f2}),
        .data_out ({w2_g, w1_g})
    );
    `UNUSED_VAR ({w_f1[0], pz_f1[0]})

    // ── stage G (@T_G): t = T*rcp, u = w1*rcp, v = w2*rcp ─────────────
    wire [2:0][31:0] tuv_h;
    wire [2:0][31:0] tuv_num = {w2_g, w1_g, t_num_g};
    for (genvar k = 0; k < 3; ++k) begin : g_scale
        VX_fma_unit #(
            .LATENCY        (F),
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (1)
        ) fmul_tuv (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (1'b1),
            .op_type (INST_FPU_MUL),
            .fmt     (FMT_ADD),
            .frm     (INST_FRM_RNE),
            .dataa   (tuv_num[k]),
            .datab   (rcp_g),
            .datac   ('0),
            .result  (tuv_h[k]),
            `UNUSED_PIN (fflags)
        );
    end

    // ── verdict flags: edge signs (@T_F), det tests (@T_F+2F) ─────────
    reg edge_ok_f;
    always @(*) begin
        logic any_neg, any_pos;
        any_neg = 1'b0;
        any_pos = 1'b0;
        for (integer i = 0; i < 3; ++i) begin
            if (w_f[i][30:0] != 31'd0
             && !((w_f[i][30:23] == 8'hff) && (w_f[i][22:0] != 23'd0))) begin
                any_neg = any_neg |  w_f[i][31];
                any_pos = any_pos | ~w_f[i][31];
            end
        end
        edge_ok_f = !(any_neg && any_pos);
    end

    wire edge_ok_d;
    VX_shift_register #(
        .DATAW (1),
        .DEPTH (2 * F)
    ) sr_edge (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (edge_ok_f),
        .data_out (edge_ok_d)
    );

    wire det_nan  = (det_g[30:23] == 8'hff) && (det_g[22:0] != 23'd0);
    wire det_ok_d = (det_g[30:0] != 31'd0) && !det_nan;
    wire back_d   = det_g[31];

    wire [2:0] flags_h;
    VX_shift_register #(
        .DATAW (3),
        .DEPTH (T_H - (T_F + 2 * F))
    ) sr_flags (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({edge_ok_d, det_ok_d, back_d}),
        .data_out (flags_h)
    );

    wire [63:0] tmm_h;
    VX_shift_register #(
        .DATAW (64),
        .DEPTH (T_H - T_B)
    ) sr_tmm (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({tmin_a, tmax_a}),
        .data_out (tmm_h)
    );

    // ── stage H (@T_H): t_min < t < t_max ─────────────────────────────
    wire range_ok = f32_lt(tmm_h[63:32], tuv_h[0]) && f32_lt(tuv_h[0], tmm_h[31:0]);

    reg        hit_r, bf_r;
    reg [31:0] u_r, v_r, t_r;
    always_ff @(posedge clk) begin
        if (enable) begin
            hit_r <= flags_h[2] && flags_h[1] && range_ok;
            bf_r  <= flags_h[0];
            t_r   <= tuv_h[0];
            u_r   <= tuv_h[1];
            v_r   <= tuv_h[2];
        end
    end

    reg [LATENCY-1:0] valid_pipe_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            valid_pipe_r <= '0;
        end else if (enable) begin
            valid_pipe_r <= {valid_pipe_r[LATENCY-2:0], valid_in};
        end
    end

    wire [TAG_WIDTH-1:0] tag_out_w;
    VX_shift_register #(
        .DATAW (TAG_WIDTH),
        .DEPTH (LATENCY)
    ) sr_tag (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (tag_in),
        .data_out (tag_out_w)
    );

    assign valid_out   = valid_pipe_r[LATENCY-1];
    assign tag_out     = tag_out_w;
    assign hit         = hit_r;
    assign t           = t_r;
    assign u           = u_r;
    assign v           = v_r;
    assign back_facing = bf_r;

endmodule
