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

// VX_rtu_box_pe — pipelined ray-vs-AABB slab intersector for one child box.
// Streams one box per cycle; emits {hit, t_near} after a fixed latency. Mirrors
// SimX reconstruct_child_aabb + rtu::ray_aabb_intersect op for op, so every
// accept decision and t_near match it:
//
//   dequant   mn[a] = origin[a] + q[a]*2^exp[a]        (product exact, one add)
//   slab      t0[a] = (mn[a] - ro[a]) * inv_d[a]       (t1 from mx)
//   reduce    lo = max(-inf, min(t0,t1)[*])   hi = min(+inf, max(t0,t1)[*])
//             (fmin/fmax drop a NaN operand, so lo and hi are never NaN)
//   hit       = hi >= max(0, lo) && lo <= t_max
//   t_near    = max(t_min, lo)                         (descent order only)
//
// The box is culled against [0, t_max], not [t_min, t_max]: t_min belongs to
// the primitive test alone, whose t carries rounding the slab distances do not.
// inv_d is the ray-setup reciprocal (VX_rtu_recip), FLT_MAX for a zero
// direction component. The FP units flush subnormals.

`include "VX_define.vh"

module VX_rtu_box_pe import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter LATENCY_FMA  = RTU_LATENCY_FMA,
    parameter TAG_WIDTH    = 1
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,
    input  wire        valid_in,
    input  wire [TAG_WIDTH-1:0] tag_in,    // caller side-band (the context id)

    // node common terms (broadcast across all children)
    input  wire [2:0][31:0] origin,
    input  wire [2:0][7:0]  exp,
    // this child's quantized AABB corners
    input  wire [2:0][7:0]  qmin,
    input  wire [2:0][7:0]  qmax,
    // raw (unquantized) AABB path — procedural-leaf boxes carry float min/max
    // directly instead of node-relative quantized corners.
    input  wire             raw,
    input  wire [2:0][31:0] raw_min,
    input  wire [2:0][31:0] raw_max,
    // ray terms (precomputed per ray)
    input  wire [2:0][31:0] ro,
    input  wire [2:0][31:0] inv_d,
    input  wire [31:0]      t_min,
    input  wire [31:0]      t_max,

    output wire        valid_out,
    output wire [TAG_WIDTH-1:0] tag_out,
    // the same tag one cycle ahead of the result, so a consumer can pre-decode
    // it instead of doing so on the cycle the result lands
    output wire [TAG_WIDTH-1:0] tag_out_pre,
    output wire        hit,
    output wire [31:0] t_near
);
    localparam F       = LATENCY_FMA;
    localparam LAT_FMA = 3 * F;                 // mn, mn - ro, * inv_d
    localparam LATENCY = LAT_FMA + 4;           // + per-axis, 2 reduce, verdict

    localparam [INST_FMT_BITS-1:0] FMT_ADD = 2'b00;   // F32, a*b + c
    localparam [INST_FMT_BITS-1:0] FMT_SUB = 2'b10;   // F32, a*b - c
    localparam [31:0] F32_ONE  = 32'h3F800000;
    localparam [31:0] F32_NEG0 = 32'h80000000;        // x + -0 == x, signs kept
    localparam [31:0] F32_PINF = 32'h7F800000;
    localparam [31:0] F32_NINF = 32'hFF800000;

    // ── helpers ───────────────────────────────────────────────────────
    // q * 2^e for an 8-bit integer q and an int8 e, as the F32 product rounds:
    // exact, +inf past the range, 0 below it (subnormals flush).
    function automatic logic [31:0] q_scale(input logic [7:0] q, input logic [7:0] e);
        logic [2:0]  msb;
        logic [6:0]  frac;
        logic signed [9:0] be;
        if (q == 8'd0) begin
            q_scale = 32'd0;
        end else begin
            msb = 3'd0;
            for (integer b = 0; b < 8; ++b) begin
                if (q[b]) begin
                    msb = b[2:0];
                end
            end
            frac = 7'(q << (3'd7 - msb));
            be = 10'sd127 + 10'(msb) + 10'($signed(e));
            if (be >= 10'sd255) begin
                q_scale = F32_PINF;
            end else if (be <= 10'sd0) begin
                q_scale = 32'd0;
            end else begin
                q_scale = {1'b0, be[7:0], frac, 16'd0};
            end
        end
    endfunction

    function automatic logic f32_is_nan(input logic [30:0] a);
        f32_is_nan = (a[30:23] == 8'hff) && (a[22:0] != 23'd0);
    endfunction

    // monotone integer key of a non-NaN F32 (+0 and -0 share one key)
    function automatic logic [31:0] f32_key(input logic [31:0] a);
        f32_key = (a[30:0] == 31'd0) ? 32'h80000000 : (a[31] ? ~a : {1'b1, a[30:0]});
    endfunction

    // IEEE a <= b; false on NaN
    function automatic logic f32_le(input logic [31:0] a, input logic [31:0] b);
        f32_le = !f32_is_nan(a[30:0]) && !f32_is_nan(b[30:0]) && (f32_key(a) <= f32_key(b));
    endfunction

    // fmin / fmax: a NaN operand yields the other one; two NaNs yield `none`
    function automatic logic [31:0] f32_min(input logic [31:0] a, input logic [31:0] b,
                                            input logic [31:0] none);
        if (f32_is_nan(a[30:0])) begin
            f32_min = f32_is_nan(b[30:0]) ? none : b;
        end else if (f32_is_nan(b[30:0])) begin
            f32_min = a;
        end else begin
            f32_min = (f32_key(b) < f32_key(a)) ? b : a;
        end
    endfunction

    function automatic logic [31:0] f32_max(input logic [31:0] a, input logic [31:0] b,
                                            input logic [31:0] none);
        if (f32_is_nan(a[30:0])) begin
            f32_max = f32_is_nan(b[30:0]) ? none : b;
        end else if (f32_is_nan(b[30:0])) begin
            f32_max = a;
        end else begin
            f32_max = (f32_key(b) > f32_key(a)) ? b : a;
        end
    endfunction

    // ── stage 1: box corners mn = origin + q*2^exp (raw: the corner itself) ──
    wire [2:0][31:0] mn_a, mx_a, mnx_c;
    for (genvar a = 0; a < 3; ++a) begin : g_prep
        assign mn_a[a]  = raw ? raw_min[a] : q_scale(qmin[a], exp[a]);
        assign mx_a[a]  = raw ? raw_max[a] : q_scale(qmax[a], exp[a]);
        assign mnx_c[a] = raw ? F32_NEG0   : origin[a];
    end

    wire [2:0][31:0] mn, mx;
    for (genvar a = 0; a < 3; ++a) begin : g_corner
        VX_fma_unit #(
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .LATENCY        (F),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (1)
        ) fma_mn (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (valid_in),
            .op_type (INST_FPU_MADD),
            .fmt     (FMT_ADD),
            .frm     (INST_FRM_RNE),
            .dataa   (mn_a[a]),
            .datab   (F32_ONE),
            .datac   (mnx_c[a]),
            .result  (mn[a]),
            `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .LATENCY        (F),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (1)
        ) fma_mx (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (valid_in),
            .op_type (INST_FPU_MADD),
            .fmt     (FMT_ADD),
            .frm     (INST_FRM_RNE),
            .dataa   (mx_a[a]),
            .datab   (F32_ONE),
            .datac   (mnx_c[a]),
            .result  (mx[a]),
            `UNUSED_PIN (fflags)
        );
    end

    wire [2:0][31:0] ro_d;
    VX_shift_register #(
        .DATAW (3*32),
        .DEPTH (F)
    ) sr_ro (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (ro),
        .data_out (ro_d)
    );

    // ── stage 2: corners relative to the ray origin, mn - ro / mx - ro ──
    wire [2:0][31:0] dmn, dmx;
    for (genvar a = 0; a < 3; ++a) begin : g_rel
        VX_fma_unit #(
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .LATENCY        (F),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (1)
        ) fma_dmn (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (1'b1),
            .op_type (INST_FPU_MADD),
            .fmt     (FMT_SUB),
            .frm     (INST_FRM_RNE),
            .dataa   (mn[a]),
            .datab   (F32_ONE),
            .datac   (ro_d[a]),
            .result  (dmn[a]),
            `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .LATENCY        (F),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (1)
        ) fma_dmx (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (1'b1),
            .op_type (INST_FPU_MADD),
            .fmt     (FMT_SUB),
            .frm     (INST_FRM_RNE),
            .dataa   (mx[a]),
            .datab   (F32_ONE),
            .datac   (ro_d[a]),
            .result  (dmx[a]),
            `UNUSED_PIN (fflags)
        );
    end

    wire [2:0][31:0] inv_d_q;
    VX_shift_register #(
        .DATAW (3*32),
        .DEPTH (2 * F)
    ) sr_invd (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (inv_d),
        .data_out (inv_d_q)
    );

    // ── stage 3: slab entry/exit per axis = (corner - ro) * inv_d ─────
    wire [2:0][31:0] t0, t1;
    for (genvar a = 0; a < 3; ++a) begin : g_slab
        VX_fma_unit #(
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .LATENCY        (F),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (1)
        ) fma_t0 (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (1'b1),
            .op_type (INST_FPU_MADD),
            .fmt     (FMT_ADD),
            .frm     (INST_FRM_RNE),
            .dataa   (dmn[a]),
            .datab   (inv_d_q[a]),
            .datac   (F32_NEG0),
            .result  (t0[a]),
            `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(
            .USE_DSP        (`VX_CFG_RTU_USE_DSP),
            .LATENCY        (F),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (1)
        ) fma_t1 (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (1'b1),
            .op_type (INST_FPU_MADD),
            .fmt     (FMT_ADD),
            .frm     (INST_FRM_RNE),
            .dataa   (dmx[a]),
            .datab   (inv_d_q[a]),
            .datac   (F32_NEG0),
            .result  (t1[a]),
            `UNUSED_PIN (fflags)
        );
    end

    // t_min/t_max delayed to the verdict stage
    wire [31:0] tmin_r, tmax_r;
    VX_shift_register #(
        .DATAW (64),
        .DEPTH (LAT_FMA + 3)
    ) sr_t (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({t_min,  t_max}),
        .data_out ({tmin_r, tmax_r})
    );

    // ── stage 4: per-axis lo/hi; an axis whose slabs are both NaN drops out ──
    reg [2:0][31:0] lo_r, hi_r;
    always_ff @(posedge clk) begin
        if (enable) begin
            for (integer a = 0; a < 3; ++a) begin
                lo_r[a] <= f32_min(t0[a], t1[a], F32_NINF);
                hi_r[a] <= f32_max(t0[a], t1[a], F32_PINF);
            end
        end
    end

    // ── stage 5/6: lo = max over axes, hi = min over axes ─────────────
    reg [31:0] near_a_r, near_b_r, far_a_r, far_b_r;
    reg [31:0] lo_all_r, hi_all_r;
    always_ff @(posedge clk) begin
        if (enable) begin
            near_a_r <= f32_max(lo_r[0], lo_r[1], F32_NINF);
            near_b_r <= lo_r[2];
            far_a_r  <= f32_min(hi_r[0], hi_r[1], F32_PINF);
            far_b_r  <= hi_r[2];
            lo_all_r <= f32_max(near_a_r, near_b_r, F32_NINF);
            hi_all_r <= f32_min(far_a_r, far_b_r, F32_PINF);
        end
    end

    // ── stage 7: hit = hi >= max(0, lo) && lo <= t_max; t_near = max(t_min, lo)
    // lo and hi are never NaN; t_near is non-negative for t_min >= 0 and +0 for
    // a zero, so the consumer may order it as an unsigned integer.
    wire        hi_ge0  = !hi_all_r[31] || (hi_all_r[30:0] == 31'd0);
    wire        hit_w   = hi_ge0 && f32_le(lo_all_r, hi_all_r) && f32_le(lo_all_r, tmax_r);
    wire [31:0] tnear_w = f32_max(tmin_r, lo_all_r, lo_all_r);
    reg         hit_r;
    reg  [31:0] t_near_r;
    always_ff @(posedge clk) begin
        if (enable) begin
            hit_r    <= hit_w;
            t_near_r <= (tnear_w[30:0] == 31'd0) ? 32'd0 : tnear_w;
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

    // carry the caller's tag alongside the datapath so streamed results can be
    // routed back to their originating context.
    // Split one stage off the end so the tag is also available a cycle early;
    // the two together are the same LATENCY stages under the same enable.
    wire [TAG_WIDTH-1:0] tag_out_pre_w;
    wire [TAG_WIDTH-1:0] tag_out_w;
    VX_shift_register #(
        .DATAW (TAG_WIDTH),
        .DEPTH (LATENCY - 1)
    ) sr_tag (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (tag_in),
        .data_out (tag_out_pre_w)
    );
    VX_shift_register #(
        .DATAW (TAG_WIDTH),
        .DEPTH (1)
    ) sr_tag_last (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (tag_out_pre_w),
        .data_out (tag_out_w)
    );

    assign valid_out   = valid_pipe_r[LATENCY-1];
    assign tag_out     = tag_out_w;
    assign tag_out_pre = tag_out_pre_w;
    assign hit         = hit_r;
    assign t_near      = t_near_r;

endmodule
