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

// VX_rtu_tri_pe — pipelined Möller-Trumbore ray-triangle intersector. Streams
// one triangle per cycle and emits {hit, t, u, v, back_facing} after a fixed
// latency.
//
//   e1 = v1 - v0      e2 = v2 - v0      T = origin - v0
//   P  = dir × e2     det = e1 · P      invDet = 1/det
//   u  = (T · P) * invDet
//   Q  = T × e1       v = (dir · Q) * invDet
//   t  = (e2 · Q) * invDet
//   hit = |det| >= EPS && 0<=u<=1 && 0<=v && u+v<=1 && tmin<=t<=tmax
//   back_facing = det < 0
//
// The FP datapath reuses VX_fma_unit (a*b±c), VX_fdivsqrt_unit (1/det) and
// VX_fncp_unit (compares); the dot/cross products are VX_rtu_fdot3 /
// VX_rtu_fcross3. Side-band operands are delayed through shift registers so
// each stage consumes time-aligned inputs, keeping the whole pipe at a fixed
// latency the scheduler tracks via valid_out.

`include "VX_define.vh"

module VX_rtu_tri_pe import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter LATENCY_FMA  = `VX_CFG_LATENCY_FMA,
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
    // VX_fncp_unit result latency is 1 (input pipe reg, OUT_REG=0); the LATENCY
    // param only sizes the unused mask pipe, so size it to 2 to avoid a
    // degenerate [-1:0] slice while the result still lands after one cycle.
    localparam FNCP_LAT  = 1;                 // result latency for alignment
    localparam FNCP_SIZE = 2;                 // mask-pipe sizing param
    localparam F  = LATENCY_FMA;
    localparam V  = LATENCY_FDIV;
    localparam LATENCY = 8*F + V + 2;

    localparam [INST_FMT_BITS-1:0] FMT_ADD = 2'b00;   // F32, a*b + c
    localparam [INST_FMT_BITS-1:0] FMT_SUB = 2'b10;   // F32, a*b - c

    localparam [31:0] FP_ZERO    = 32'h00000000;
    localparam [31:0] FP_ONE     = 32'h3F800000;
    localparam [31:0] FP_EPS     = 32'h358637BD;   //  1e-6
    localparam [31:0] FP_NEG_EPS = 32'hB58637BD;   // -1e-6

    // ── stage e (@F): edge vectors and ray-origin offset ──────────────
    wire [2:0][31:0] e1, e2, tvec;
    for (genvar a = 0; a < 3; ++a) begin : g_edges
        VX_fma_unit #(.LATENCY (F)) fma_e1 (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MADD), .fmt (FMT_SUB), .frm (INST_FRM_RNE),
            .dataa (v1[a]), .datab (FP_ONE), .datac (v0[a]),
            .result (e1[a]), `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(.LATENCY (F)) fma_e2 (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MADD), .fmt (FMT_SUB), .frm (INST_FRM_RNE),
            .dataa (v2[a]), .datab (FP_ONE), .datac (v0[a]),
            .result (e2[a]), `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(.LATENCY (F)) fma_t (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MADD), .fmt (FMT_SUB), .frm (INST_FRM_RNE),
            .dataa (origin[a]), .datab (FP_ONE), .datac (v0[a]),
            .result (tvec[a]), `UNUSED_PIN (fflags)
        );
    end

    // dir aligned to the cross/dot consumers
    wire [2:0][31:0] dir_f, dir_3f;
    VX_shift_register #(.DATAW (96), .DEPTH (F)) sr_dir_f (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (dir), .data_out (dir_f)
    );
    VX_shift_register #(.DATAW (96), .DEPTH (3*F)) sr_dir_3f (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (dir), .data_out (dir_3f)
    );

    // ── stage cross (@3F): P = dir × e2, Q = T × e1 ───────────────────
    wire [2:0][31:0] pvec, qvec;
    VX_rtu_fcross3 #(.LATENCY_FMA (F)) cross_p (
        .clk (clk), .reset (reset), .enable (enable),
        .a (dir_f), .b (e2), .result (pvec)
    );
    VX_rtu_fcross3 #(.LATENCY_FMA (F)) cross_q (
        .clk (clk), .reset (reset), .enable (enable),
        .a (tvec), .b (e1), .result (qvec)
    );

    // e1/e2/T aligned from @F to @3F to feed the dot products
    wire [2:0][31:0] e1_3f, e2_3f, t_3f;
    VX_shift_register #(.DATAW (96), .DEPTH (2*F)) sr_e1 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (e1), .data_out (e1_3f)
    );
    VX_shift_register #(.DATAW (96), .DEPTH (2*F)) sr_e2 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (e2), .data_out (e2_3f)
    );
    VX_shift_register #(.DATAW (96), .DEPTH (2*F)) sr_t (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (tvec), .data_out (t_3f)
    );

    // ── stage dot (@6F): det, and the un-scaled u/v/t numerators ──────
    wire [31:0] det, u_num, v_num, t_num;
    VX_rtu_fdot3 #(.LATENCY_FMA (F)) dot_det (
        .clk (clk), .reset (reset), .enable (enable),
        .a (e1_3f), .b (pvec), .result (det)
    );
    VX_rtu_fdot3 #(.LATENCY_FMA (F)) dot_u (
        .clk (clk), .reset (reset), .enable (enable),
        .a (t_3f), .b (pvec), .result (u_num)
    );
    VX_rtu_fdot3 #(.LATENCY_FMA (F)) dot_v (
        .clk (clk), .reset (reset), .enable (enable),
        .a (dir_3f), .b (qvec), .result (v_num)
    );
    VX_rtu_fdot3 #(.LATENCY_FMA (F)) dot_t (
        .clk (clk), .reset (reset), .enable (enable),
        .a (e2_3f), .b (qvec), .result (t_num)
    );

    // ── stage recip (@6F+V): invDet = 1/det ───────────────────────────
    wire [31:0] inv_det;
    VX_fdivsqrt_unit #(.LATENCY (V)) recip (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .fmt ('0), .frm (INST_FRM_RNE),
        .dataa (FP_ONE), .datab (det), .is_sqrt (1'b0),
        .result (inv_det), `UNUSED_PIN (fflags)
    );

    // numerators aligned from @6F to @6F+V
    wire [31:0] u_num_d, v_num_d, t_num_d;
    VX_shift_register #(.DATAW (96), .DEPTH (V)) sr_num (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({u_num,   v_num,   t_num}),
        .data_out ({u_num_d, v_num_d, t_num_d})
    );

    // ── stage scale (@7F+V): u/v/t = numerator * invDet ───────────────
    wire [31:0] u_w, v_w, t_w;
    VX_fma_unit #(.LATENCY (F)) fma_u (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (u_num_d), .datab (inv_det), .datac (FP_ZERO),
        .result (u_w), `UNUSED_PIN (fflags)
    );
    VX_fma_unit #(.LATENCY (F)) fma_v (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (v_num_d), .datab (inv_det), .datac (FP_ZERO),
        .result (v_w), `UNUSED_PIN (fflags)
    );
    VX_fma_unit #(.LATENCY (F)) fma_t2 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (t_num_d), .datab (inv_det), .datac (FP_ZERO),
        .result (t_w), `UNUSED_PIN (fflags)
    );

    // ── stage sum (@8F+V): uv = u + v ─────────────────────────────────
    wire [31:0] uv_w;
    VX_fma_unit #(.LATENCY (F)) fma_uv (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (u_w), .datab (FP_ONE), .datac (v_w),
        .result (uv_w), `UNUSED_PIN (fflags)
    );

    // u/v/t aligned from @7F+V to @8F+V
    wire [31:0] u_c, v_c, t_c;
    VX_shift_register #(.DATAW (96), .DEPTH (F)) sr_uvt (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({u_w, v_w, t_w}),
        .data_out ({u_c, v_c, t_c})
    );
    // det aligned from @6F to @8F+V
    wire [31:0] det_c;
    VX_shift_register #(.DATAW (32), .DEPTH (2*F + V)) sr_det (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (det), .data_out (det_c)
    );
    // t_min/t_max aligned from @0 to @8F+V
    wire [31:0] tmin_c, tmax_c;
    VX_shift_register #(.DATAW (64), .DEPTH (8*F + V)) sr_tmm (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({t_min,  t_max}),
        .data_out ({tmin_c, tmax_c})
    );

    // ── stage compare (@8F+V+1): bound and determinant tests ──────────
    wire [`VX_CFG_XLEN-1:0] cu0, cu1, cv0, cuv, ct0, ct1, cdp, cdn, bfc;
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) cmp_u0 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_CMP), .frm (3'd0 /*LE*/),
        .dataa (FP_ZERO), .datab (u_c), .result (cu0), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) cmp_u1 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_CMP), .frm (3'd0),
        .dataa (u_c), .datab (FP_ONE), .result (cu1), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) cmp_v0 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_CMP), .frm (3'd0),
        .dataa (FP_ZERO), .datab (v_c), .result (cv0), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) cmp_uv (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_CMP), .frm (3'd0),
        .dataa (uv_w), .datab (FP_ONE), .result (cuv), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) cmp_t0 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_CMP), .frm (3'd0),
        .dataa (tmin_c), .datab (t_c), .result (ct0), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) cmp_t1 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_CMP), .frm (3'd0),
        .dataa (t_c), .datab (tmax_c), .result (ct1), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) cmp_dp (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_CMP), .frm (3'd0),
        .dataa (FP_EPS), .datab (det_c), .result (cdp), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) cmp_dn (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_CMP), .frm (3'd0),
        .dataa (det_c), .datab (FP_NEG_EPS), .result (cdn), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) cmp_bf (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_CMP), .frm (3'd1 /*LT*/),
        .dataa (det_c), .datab (FP_ZERO), .result (bfc), `UNUSED_PIN (fflags));

    wire pass_w = cu0[0] & cu1[0] & cv0[0] & cuv[0]
                & ct0[0] & ct1[0] & (cdp[0] | cdn[0]);

    // u/v/t aligned from @8F+V to @8F+V+1 (one compare stage)
    wire [31:0] u_a, v_a, t_a;
    VX_shift_register #(.DATAW (96), .DEPTH (FNCP_LAT)) sr_uvt2 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({u_c, v_c, t_c}),
        .data_out ({u_a, v_a, t_a})
    );

    // ── stage commit (@8F+V+2): register the verdict and attributes ───
    reg        hit_r, bf_r;
    reg [31:0] u_r, v_r, t_r;
    always @(posedge clk) begin
        if (enable) begin
            hit_r <= pass_w;
            bf_r  <= bfc[0];
            u_r   <= u_a;
            v_r   <= v_a;
            t_r   <= t_a;
        end
    end

    reg [LATENCY-1:0] valid_pipe;
    always @(posedge clk) begin
        if (reset) begin
            valid_pipe <= '0;
        end else if (enable) begin
            valid_pipe <= {valid_pipe[LATENCY-2:0], valid_in};
        end
    end

    // carry the caller's tag alongside the datapath so streamed results can be
    // routed back to their originating context.
    wire [TAG_WIDTH-1:0] tag_out_w;
    VX_shift_register #(.DATAW (TAG_WIDTH), .DEPTH (LATENCY)) sr_tag (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (tag_in), .data_out (tag_out_w)
    );

    assign valid_out   = valid_pipe[LATENCY-1];
    assign tag_out     = tag_out_w;
    assign hit         = hit_r;
    assign t           = t_r;
    assign u           = u_r;
    assign v           = v_r;
    assign back_facing = bf_r;

endmodule
