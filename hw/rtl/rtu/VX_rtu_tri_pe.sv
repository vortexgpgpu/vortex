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
// Wald, JCGT 2013). Streams one triangle per cycle and emits {hit, t, u, v,
// back_facing} after a fixed latency. Mirrors SimX rtu::ray_triangle op for op:
//
//   kz = argmax|dir|, kx/ky follow (swapped when dir[kz] < 0)
//   F32: sz = 1/dir[kz], sx = dir[kx]*sz, sy = dir[ky]*sz
//        r = vertex - origin, px = rx - sx*rz, py = ry - sy*rz
//   F64: pz = sz*rz (exact), w_i = px_a*py_b - py_a*px_b (one rounding)
//        det = w0 + (w1 + w2), T = (w0*pz0 + w1*pz1) + w2*pz2
//        t = f32(T / det), (u, v) = f32(w1, w2) / f32(det)
//   hit = !(any w < 0 && any w > 0) && det != 0 && tmin <= t <= tmax
//   back_facing = det < 0
//
// A shared edge evaluates to exactly negated weights in its two triangles, so
// the test is watertight; the F64 edge functions and t keep t within half an
// ulp of the exact intersection, in the op order the Vulkan reference
// (lavapipe) uses, so coincident triangles resolve the same way.

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
    localparam F   = LATENCY_FMA;
    localparam V   = LATENCY_FDIV;
    localparam D   = RTU_LATENCY_FMA64;
    localparam V64 = RTU_FDIV64_LAT;

    // stage start times (cycles after valid_in)
    localparam T_B = 1;                 // canonical/axis select registered
    localparam T_C = T_B + V;           // sz ready
    localparam T_D = T_C + F;           // sx, sy ready
    localparam T_E = T_D + F;           // sx*rz, sy*rz ready
    localparam T_F = T_E + F;           // px, py ready
    localparam T_G = T_F + 2 * D;       // w ready
    localparam T_H = T_G + 3 * D;       // T ready (det at T_G + 2D)
    localparam T_I = T_H + V64 + 1;     // t narrowed and registered
    localparam LATENCY = T_I + 1;       // verdict registered

    `STATIC_ASSERT(V >= F, ("tri PE: FDIV latency must cover the r subtract"))
    `STATIC_ASSERT(T_G + 2 * D + 1 + V <= T_I, ("tri PE: bary divide must land before t"))

    localparam [INST_FMT_BITS-1:0] FMT_ADD = 2'b00;
    localparam [INST_FMT_BITS-1:0] FMT_SUB = 2'b10;
    localparam [31:0] F32_ONE = 32'h3F800000;

    // ── helpers ───────────────────────────────────────────────────────
    // exact F32 -> F64 widening (subnormals normalized)
    function automatic [63:0] f32_to_f64(input [31:0] a);
        reg [7:0]  e;
        reg [22:0] m;
        reg [4:0]  lz;
        reg [22:0] mn;
        begin
            e = a[30:23];
            m = a[22:0];
            if (e == 8'hff) begin
                f32_to_f64 = {a[31], 11'h7ff, m, 29'd0};
            end else if (e == 8'd0) begin
                if (m == 23'd0) begin
                    f32_to_f64 = {a[31], 63'd0};
                end else begin
                    lz = 5'd0;
                    for (integer i = 22; i >= 0; --i) begin
                        if (m[i]) begin
                            lz = 5'(22 - i);
                            break;
                        end
                    end
                    mn = m << (lz + 5'd1);
                    f32_to_f64 = {a[31], 11'(11'd896 - 11'(lz)), mn, 29'd0};
                end
            end else begin
                f32_to_f64 = {a[31], 11'(e) + 11'd896, m, 29'd0};
            end
        end
    endfunction

    // F64 -> F32, round to nearest even
    function automatic [31:0] f64_to_f32(input [63:0] a);
        reg        s;
        reg [10:0] e;
        reg [51:0] m;
        reg signed [12:0] ue;
        reg [52:0] sig;
        reg [6:0]  sh;
        reg [22:0] keep;
        reg        guard, sticky;
        reg [31:0] base;
        begin
            s  = a[63];
            e  = a[62:52];
            m  = a[51:0];
            ue = 13'(e) - 13'sd896;
            if (e == 11'h7ff) begin
                f64_to_f32 = {s, 8'hff, (m != 52'd0) ? {1'b1, m[50:29]} : 23'd0};
            end else if (e == 11'd0) begin
                f64_to_f32 = {s, 31'd0};
            end else if (ue >= 13'sd255) begin
                f64_to_f32 = {s, 8'hff, 23'd0};
            end else if (ue >= 13'sd1) begin
                guard  = m[28];
                sticky = (m[27:0] != 28'd0);
                base   = {s, ue[7:0], m[51:29]};
                f64_to_f32 = base + 32'((guard && (sticky || m[29])) ? 1 : 0);
            end else begin
                // subnormal: mantissa = sig >> (30 - ue), ue <= 0
                sig = {1'b1, m};
                sh  = (ue < -13'sd30) ? 7'd61 : 7'(13'sd30 - ue);
                if (sh > 7'd54) begin
                    keep   = 23'd0;
                    guard  = 1'b0;
                    sticky = 1'b1;
                end else begin
                    keep   = 23'(sig >> sh);
                    guard  = sig[6'(sh - 7'd1)];
                    sticky = (64'(sig) & ((64'd1 << (sh - 7'd1)) - 64'd1)) != 64'd0;
                end
                base = {s, 8'd0, keep};
                f64_to_f32 = base + 32'((guard && (sticky || keep[0])) ? 1 : 0);
            end
        end
    endfunction

    // IEEE ordering on F32 (+0 == -0); NaN compares false
    function automatic f32_le(input [31:0] a, input [31:0] b);
        reg a_nan, b_nan;
        reg [31:0] ka, kb;
        begin
            a_nan = (a[30:23] == 8'hff) && (a[22:0] != 23'd0);
            b_nan = (b[30:23] == 8'hff) && (b[22:0] != 23'd0);
            ka = (a[30:0] == 31'd0) ? 32'h80000000 : (a[31] ? ~a : {1'b1, a[30:0]});
            kb = (b[30:0] == 31'd0) ? 32'h80000000 : (b[31] ? ~b : {1'b1, b[30:0]});
            f32_le = !a_nan && !b_nan && (ka <= kb);
        end
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

    // per canonical vertex: (x, y, z) components in the sheared frame's axes
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
        .EXCEPT_ENABLE  (0)
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
                .LATENCY (F),
                .USE_DSP        (`VX_CFG_RTU_USE_DSP),
                .SUBNORM_ENABLE (0),
                .EXCEPT_ENABLE  (0)
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

    // r from @T_B+F to @T_D (consumed by the sx*rz stage)
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
            .LATENCY (F),
                .USE_DSP        (`VX_CFG_RTU_USE_DSP),
                .SUBNORM_ENABLE (0),
                .EXCEPT_ENABLE  (0)
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

    // ── stage D (@T_D): sx*rz, sy*rz (F32); pz = sz*rz (F64, exact) ────
    wire [2:0][1:0][31:0] m_e;
    wire [2:0][63:0]      pz_x;   // @T_D + D
    for (genvar i = 0; i < 3; ++i) begin : g_shear
        for (genvar a = 0; a < 2; ++a) begin : g_ax
            VX_fma_unit #(
                .LATENCY (F),
                .USE_DSP        (`VX_CFG_RTU_USE_DSP),
                .SUBNORM_ENABLE (0),
                .EXCEPT_ENABLE  (0)
            ) fmul_m (
                .clk     (clk),
                .reset   (reset),
                .enable  (enable),
                .mask    (1'b1),
                .op_type (INST_FPU_MUL),
                .fmt     (FMT_ADD),
                .frm     (INST_FRM_RNE),
                .dataa   (sxy_d[a]),
                .datab   (r_d[i][2]),
                .datac   ('0),
                .result  (m_e[i][a]),
                `UNUSED_PIN (fflags)
            );
        end
        VX_fma_unit #(
            .LATENCY        (D),
            .MAN_BITS       (52),
            .EXP_BITS       (11),
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (0)
        ) fmul_pz (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (1'b1),
            .op_type (INST_FPU_MUL),
            .fmt     (FMT_ADD),
            .frm     (INST_FRM_RNE),
            .dataa   (f32_to_f64(sz_d)),
            .datab   (f32_to_f64(r_d[i][2])),
            .datac   ('0),
            .result  (pz_x[i]),
            `UNUSED_PIN (fflags)
        );
    end

    // rx, ry from @T_D to @T_E
    wire [2:0][1:0][31:0] rxy_e;
    VX_shift_register #(
        .DATAW (6 * 32),
        .DEPTH (T_E - T_D)
    ) sr_rxy (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({r_d[2][1], r_d[2][0], r_d[1][1], r_d[1][0], r_d[0][1], r_d[0][0]}),
        .data_out (rxy_e)
    );

    // ── stage E (@T_E): px = rx - sx*rz, py = ry - sy*rz ──────────────
    wire [2:0][1:0][31:0] p_f;   // [vertex][x/y]
    for (genvar i = 0; i < 3; ++i) begin : g_p
        for (genvar a = 0; a < 2; ++a) begin : g_ax
            VX_fma_unit #(
                .LATENCY (F),
                .USE_DSP        (`VX_CFG_RTU_USE_DSP),
                .SUBNORM_ENABLE (0),
                .EXCEPT_ENABLE  (0)
            ) fsub_p (
                .clk     (clk),
                .reset   (reset),
                .enable  (enable),
                .mask    (1'b1),
                .op_type (INST_FPU_ADD),
                .fmt     (FMT_SUB),
                .frm     (INST_FRM_RNE),
                .dataa   (rxy_e[i][a]),
                .datab   (m_e[i][a]),
                .datac   ('0),
                .result  (p_f[i][a]),
                `UNUSED_PIN (fflags)
            );
        end
    end

    // ── stage F (@T_F): w_i = px_a*py_b - py_a*px_b in F64 ────────────
    // (a, b) per weight: w0 <- (2, 1), w1 <- (0, 2), w2 <- (1, 0)
    wire [2:0][1:0][63:0] p64_f;
    for (genvar i = 0; i < 3; ++i) begin : g_p64
        assign p64_f[i][0] = f32_to_f64(p_f[i][0]);
        assign p64_f[i][1] = f32_to_f64(p_f[i][1]);
    end

    wire [2:0][1:0][63:0] p64_g1;   // operands delayed D for the fused stage
    VX_shift_register #(
        .DATAW (6 * 64),
        .DEPTH (D)
    ) sr_p64 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (p64_f),
        .data_out (p64_g1)
    );

    wire [2:0][63:0] w_g;
    for (genvar i = 0; i < 3; ++i) begin : g_w
        localparam IA = (i == 0) ? 2 : ((i == 1) ? 0 : 1);
        localparam IB = (i == 0) ? 1 : ((i == 1) ? 2 : 0);
        wire [63:0] cross_q;   // py_a * px_b, exact
        VX_fma_unit #(
            .LATENCY        (D),
            .MAN_BITS       (52),
            .EXP_BITS       (11),
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (0)
        ) fmul_c (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (1'b1),
            .op_type (INST_FPU_MUL),
            .fmt     (FMT_ADD),
            .frm     (INST_FRM_RNE),
            .dataa   (p64_f[IA][1]),
            .datab   (p64_f[IB][0]),
            .datac   ('0),
            .result  (cross_q),
            `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(
            .LATENCY        (D),
            .MAN_BITS       (52),
            .EXP_BITS       (11),
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (0)
        ) fmsub_w (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (1'b1),
            .op_type (INST_FPU_MADD),
            .fmt     (FMT_SUB),
            .frm     (INST_FRM_RNE),
            .dataa   (p64_g1[IA][0]),
            .datab   (p64_g1[IB][1]),
            .datac   (cross_q),
            .result  (w_g[i]),
            `UNUSED_PIN (fflags)
        );
    end

    // pz from @T_D+D to @T_G
    wire [2:0][63:0] pz_g;
    VX_shift_register #(
        .DATAW (3 * 64),
        .DEPTH (T_G - (T_D + D))
    ) sr_pz (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (pz_x),
        .data_out (pz_g)
    );

    // ── stage G (@T_G): det = w0 + (w1 + w2); T = (w0 pz0 + w1 pz1) + w2 pz2
    wire [63:0] det12, det_g2, tp0, tp1, tp2, t01, t_num;
    wire [63:0] w0_g1, tp2_g2;
    VX_shift_register #(
        .DATAW (64),
        .DEPTH (D)
    ) sr_w0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (w_g[0]),
        .data_out (w0_g1)
    );
    VX_fma_unit #(.LATENCY (D), .MAN_BITS (52), .EXP_BITS (11), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (0)) fadd_det12 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_ADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (w_g[1]), .datab (w_g[2]), .datac ('0),
        .result (det12), `UNUSED_PIN (fflags)
    );
    VX_fma_unit #(.LATENCY (D), .MAN_BITS (52), .EXP_BITS (11), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (0)) fadd_det (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_ADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (w0_g1), .datab (det12), .datac ('0),
        .result (det_g2), `UNUSED_PIN (fflags)
    );
    VX_fma_unit #(.LATENCY (D), .MAN_BITS (52), .EXP_BITS (11), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (0)) fmul_tp0 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MUL), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (w_g[0]), .datab (pz_g[0]), .datac ('0),
        .result (tp0), `UNUSED_PIN (fflags)
    );
    VX_fma_unit #(.LATENCY (D), .MAN_BITS (52), .EXP_BITS (11), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (0)) fmul_tp1 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MUL), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (w_g[1]), .datab (pz_g[1]), .datac ('0),
        .result (tp1), `UNUSED_PIN (fflags)
    );
    VX_fma_unit #(.LATENCY (D), .MAN_BITS (52), .EXP_BITS (11), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (0)) fmul_tp2 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MUL), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (w_g[2]), .datab (pz_g[2]), .datac ('0),
        .result (tp2), `UNUSED_PIN (fflags)
    );
    VX_fma_unit #(.LATENCY (D), .MAN_BITS (52), .EXP_BITS (11), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (0)) fadd_t01 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_ADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (tp0), .datab (tp1), .datac ('0),
        .result (t01), `UNUSED_PIN (fflags)
    );
    VX_shift_register #(
        .DATAW (64),
        .DEPTH (D)
    ) sr_tp2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (tp2),
        .data_out (tp2_g2)
    );
    VX_fma_unit #(.LATENCY (D), .MAN_BITS (52), .EXP_BITS (11), .USE_DSP (`VX_CFG_RTU_USE_DSP), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (0)) fadd_t (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_ADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (t01), .datab (tp2_g2), .datac ('0),
        .result (t_num), `UNUSED_PIN (fflags)
    );

    // det from @T_G+2D to @T_H
    wire [63:0] det_h;
    VX_shift_register #(
        .DATAW (64),
        .DEPTH (T_H - (T_G + 2 * D))
    ) sr_det (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (det_g2),
        .data_out (det_h)
    );

    // ── stage H (@T_H): t = f32(T / det) ──────────────────────────────
    wire [63:0] t64;
    VX_fdiv_unit #(
        .LATENCY        (V64),
        .FLEN           (64),
        .SUBNORM_ENABLE (0),
        .EXCEPT_ENABLE  (0)
    ) fdiv_t (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .mask    (1'b1),
        .fmt     (2'b01),
        .frm     (INST_FRM_RNE),
        .dataa   (t_num),
        .datab   (det_h),
        .result  (t64),
        `UNUSED_PIN (fflags)
    );

    reg [31:0] t_i;
    always_ff @(posedge clk) begin
        if (enable) begin
            t_i <= f64_to_f32(t64);
        end
    end

    // ── barycentrics (@T_G+2D): f32(w1) / f32(det), f32(w2) / f32(det)
    wire [1:0][63:0] w_b;   // w1, w2
    VX_shift_register #(
        .DATAW (2 * 64),
        .DEPTH (2 * D)
    ) sr_wb (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({w_g[2], w_g[1]}),
        .data_out (w_b)
    );
    reg [31:0] wu_r, wv_r, det32_r;
    always_ff @(posedge clk) begin
        if (enable) begin
            wu_r    <= f64_to_f32(w_b[0]);
            wv_r    <= f64_to_f32(w_b[1]);
            det32_r <= f64_to_f32(det_g2);
        end
    end

    wire [31:0] u_q, v_q;
    VX_fdiv_unit #(
        .LATENCY        (V),
        .FLEN           (32),
        .USE_DSP        (`VX_CFG_RTU_USE_DSP),
        .SUBNORM_ENABLE (0),
        .EXCEPT_ENABLE  (0)
    ) fdiv_u (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .mask    (1'b1),
        .fmt     ('0),
        .frm     (INST_FRM_RNE),
        .dataa   (wu_r),
        .datab   (det32_r),
        .result  (u_q),
        `UNUSED_PIN (fflags)
    );
    VX_fdiv_unit #(
        .LATENCY        (V),
        .FLEN           (32),
        .USE_DSP        (`VX_CFG_RTU_USE_DSP),
        .SUBNORM_ENABLE (0),
        .EXCEPT_ENABLE  (0)
    ) fdiv_v (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .mask    (1'b1),
        .fmt     ('0),
        .frm     (INST_FRM_RNE),
        .dataa   (wv_r),
        .datab   (det32_r),
        .result  (v_q),
        `UNUSED_PIN (fflags)
    );

    wire [31:0] u_i, v_i;
    VX_shift_register #(
        .DATAW (64),
        .DEPTH (T_I - (T_G + 2 * D + 1 + V))
    ) sr_uv (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({u_q, v_q}),
        .data_out ({u_i, v_i})
    );

    // ── verdict flags: edge signs (@T_G), det tests (@T_G+2D) ─────────
    reg edge_ok_g;
    always @(*) begin
        reg any_neg, any_pos;
        any_neg = 1'b0;
        any_pos = 1'b0;
        for (integer i = 0; i < 3; ++i) begin
            if (w_g[i][62:0] != 63'd0
             && !((w_g[i][62:52] == 11'h7ff) && (w_g[i][51:0] != 52'd0))) begin
                any_neg = any_neg |  w_g[i][63];
                any_pos = any_pos | ~w_g[i][63];
            end
        end
        edge_ok_g = !(any_neg && any_pos);
    end

    wire edge_ok_d;
    VX_shift_register #(
        .DATAW (1),
        .DEPTH (2 * D)
    ) sr_edge (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (edge_ok_g),
        .data_out (edge_ok_d)
    );

    wire det_nan  = (det_g2[62:52] == 11'h7ff) && (det_g2[51:0] != 52'd0);
    wire det_ok_d = (det_g2[62:0] != 63'd0) && !det_nan;
    wire back_d   = det_g2[63];

    wire [2:0] flags_i;
    VX_shift_register #(
        .DATAW (3),
        .DEPTH (T_I - (T_G + 2 * D))
    ) sr_flags (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({edge_ok_d, det_ok_d, back_d}),
        .data_out (flags_i)
    );

    wire [63:0] tmm_i;
    VX_shift_register #(
        .DATAW (64),
        .DEPTH (T_I - T_B)
    ) sr_tmm (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({tmin_a, tmax_a}),
        .data_out (tmm_i)
    );

    // ── stage I (@T_I): range test and commit ─────────────────────────
    wire range_ok = f32_le(tmm_i[63:32], t_i) && f32_le(t_i, tmm_i[31:0]);

    reg        hit_r, bf_r;
    reg [31:0] u_r, v_r, t_r;
    always_ff @(posedge clk) begin
        if (enable) begin
            hit_r <= flags_i[2] && flags_i[1] && range_ok;
            bf_r  <= flags_i[0];
            u_r   <= u_i;
            v_r   <= v_i;
            t_r   <= t_i;
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
