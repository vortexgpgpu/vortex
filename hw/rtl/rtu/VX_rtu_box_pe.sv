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
// Streams one box per cycle; emits {hit, t_near} after a fixed latency.
//
//   dequant   mn[a] = origin[a] + qmin[a] * 2^exp[a]      (qmax symmetric)
//   slab      t0[a] = (mn[a] - ro[a]) * inv_d[a]          (t1 from mx)
//             lo[a] = min(t0,t1)   hi[a] = max(t0,t1)
//   reduce    t_near = max(t_min, lo[x], lo[y], lo[z])
//             t_far  = min(t_max, hi[x], hi[y], hi[z])
//   hit       = (t_near <= t_far)
//
// The slab subtracts the ray origin before multiplying by inv_d (rather than
// the algebraically-equal mn*inv_d - ro*inv_d) so axis-aligned rays — where
// inv_d is +/-inf — stay numerically correct: (mn-ro) is finite, so (mn-ro)*inf
// is a signed infinity that the min/max reduction treats as a non-constraining
// slab, instead of inf-inf = NaN. The uint8->fp32 and 2^exp dequant terms are
// combinational; the FP add/mul use VX_fma_unit (a*b±c) and the min/max/compare
// use VX_fncp_unit, register-balanced to the configured latencies.

`include "VX_define.vh"

module VX_rtu_box_pe import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter LATENCY_FMA  = `VX_CFG_LATENCY_FMA
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,
    input  wire        valid_in,

    // node common terms (broadcast across all children)
    input  wire [2:0][31:0] origin,
    input  wire [2:0][7:0]  exp,
    // this child's quantized AABB corners
    input  wire [2:0][7:0]  qmin,
    input  wire [2:0][7:0]  qmax,
    // raw (unquantized) AABB path — procedural-leaf boxes carry float min/max
    // directly instead of node-relative quantized corners. raw=0 is bit-
    // identical to the quantized path (BVH internal-node box tests).
    input  wire             raw,
    input  wire [2:0][31:0] raw_min,
    input  wire [2:0][31:0] raw_max,
    // ray terms (precomputed per ray)
    input  wire [2:0][31:0] ro,
    input  wire [2:0][31:0] inv_d,
    input  wire [31:0]      t_min,
    input  wire [31:0]      t_max,

    output wire        valid_out,
    output wire        hit,
    output wire [31:0] t_near
);
    // VX_fncp_unit result latency is 1 (one input pipe reg, OUT_REG=0); its
    // LATENCY param only sizes the internal mask pipe, not the result path, so
    // size it to 2 to avoid a degenerate [-1:0] mask-pipe slice while the result
    // still lands after one cycle.
    localparam FNCP_LAT    = 1;     // result latency for alignment
    localparam FNCP_SIZE   = 2;     // mask-pipe sizing param
    localparam LAT_ORIGIN  = LATENCY_FMA;             // origin - ro
    localparam LAT_DEQUANT = LATENCY_FMA;             // q*scale + (origin - ro)
    localparam LAT_SLAB    = LATENCY_FMA;             // (mn - ro)*inv_d
    localparam LAT_MINMAX  = FNCP_LAT;                // lo/hi per axis
    localparam LAT_REDUCE  = 2 * FNCP_LAT;            // 4-input min/max tree
    localparam LAT_CMP      = FNCP_LAT;               // t_near <= t_far
    localparam LATENCY      = LAT_ORIGIN + LAT_DEQUANT + LAT_SLAB + LAT_MINMAX + LAT_REDUCE + LAT_CMP;

    localparam [INST_FMT_BITS-1:0] FMT_ADD = 2'b00;   // F32, a*b + c
    localparam [INST_FMT_BITS-1:0] FMT_SUB = 2'b10;   // F32, a*b - c

    // ── combinational uint8 -> fp32 ───────────────────────────────────
    function automatic logic [31:0] u8_to_f32(input logic [7:0] n);
        logic [2:0]  msb;
        logic [7:0]  shifted;
        logic [22:0] man;
        if (n == 8'd0) begin
            u8_to_f32 = 32'd0;
        end else begin
            msb = 3'd0;
            for (integer b = 0; b < 8; ++b) begin
                if (n[b]) begin
                    msb = b[2:0];
                end
            end
            // normalize so the leading 1 sits at bit 7, then the 7 bits
            // below it become the top of the fp32 mantissa.
            shifted = n << (3'd7 - msb);
            man = {shifted[6:0], 16'd0};
            u8_to_f32 = {1'b0, (8'd127 + 8'(msb)), man};
        end
    endfunction

    // ── combinational 2^exp as fp32 (well-conditioned exponents) ──────
    function automatic logic [31:0] pow2_f32(input logic [7:0] e);
        logic signed [8:0] biased;
        biased = 9'sd127 + {e[7], e};   // sign-extend int8 exponent
        pow2_f32 = {1'b0, biased[7:0], 23'd0};
    endfunction

    // ── stage 0: prep per-axis float operands ─────────────────────────
    wire [2:0][31:0] qmin_f, qmax_f, scale;
    for (genvar a = 0; a < 3; ++a) begin : g_prep
        assign qmin_f[a] = u8_to_f32(qmin[a]);
        assign qmax_f[a] = u8_to_f32(qmax[a]);
        assign scale[a]  = pow2_f32(exp[a]);
    end

    // ── stage 1: origin - ro (per axis) ───────────────────────────────
    wire [2:0][31:0] oro;
    for (genvar a = 0; a < 3; ++a) begin : g_origin
        VX_fma_unit #(.LATENCY (LAT_ORIGIN)) fma_oro (
            .clk (clk), .reset (reset), .enable (enable), .mask (valid_in),
            .op_type (INST_FPU_MADD), .fmt (FMT_SUB), .frm (INST_FRM_RNE),
            .dataa (origin[a]), .datab (32'h3F800000 /*1.0*/), .datac (ro[a]),
            .result (oro[a]), `UNUSED_PIN (fflags)
        );
    end

    // quantized corners delayed to align with origin-ro
    wire [2:0][31:0] qmin_f_q, qmax_f_q, scale_q;
    VX_shift_register #(.DATAW (3*32*3), .DEPTH (LAT_ORIGIN)) sr_q (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({qmin_f,   qmax_f,   scale}),
        .data_out ({qmin_f_q, qmax_f_q, scale_q})
    );

    // raw-path operands delayed to align with the dequant-FMA inputs.
    wire             raw_d;
    wire [2:0][31:0] raw_min_d, raw_max_d, ro_d;
    VX_shift_register #(.DATAW (1 + 3*32*3), .DEPTH (LAT_ORIGIN)) sr_raw (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({raw,   raw_min,   raw_max,   ro}),
        .data_out ({raw_d, raw_min_d, raw_max_d, ro_d})
    );

    // ── stage 2: corners relative to the ray origin (mn-ro, mx-ro). Quantized:
    //    q*scale + (origin-ro). Raw procedural box: (min*1.0 - ro) directly,
    //    reusing the same FMAs (FMT_SUB). ──
    localparam [31:0] FP_ONE = 32'h3F800000;
    wire [2:0][31:0] dmn, dmx;
    for (genvar a = 0; a < 3; ++a) begin : g_dequant
        VX_fma_unit #(.LATENCY (LAT_DEQUANT)) fma_mn (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MADD), .fmt (raw_d ? FMT_SUB : FMT_ADD), .frm (INST_FRM_RNE),
            .dataa (raw_d ? raw_min_d[a] : qmin_f_q[a]),
            .datab (raw_d ? FP_ONE       : scale_q[a]),
            .datac (raw_d ? ro_d[a]      : oro[a]),
            .result (dmn[a]), `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(.LATENCY (LAT_DEQUANT)) fma_mx (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MADD), .fmt (raw_d ? FMT_SUB : FMT_ADD), .frm (INST_FRM_RNE),
            .dataa (raw_d ? raw_max_d[a] : qmax_f_q[a]),
            .datab (raw_d ? FP_ONE       : scale_q[a]),
            .datac (raw_d ? ro_d[a]      : oro[a]),
            .result (dmx[a]), `UNUSED_PIN (fflags)
        );
    end

    // inv_d delayed to align with the origin-relative corners
    wire [2:0][31:0] inv_d_q;
    VX_shift_register #(.DATAW (3*32), .DEPTH (LAT_ORIGIN + LAT_DEQUANT)) sr_invd (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (inv_d), .data_out (inv_d_q)
    );

    // ── stage 3: slab entry/exit per axis = (corner - ro) * inv_d ─────
    wire [2:0][31:0] t0, t1;
    for (genvar a = 0; a < 3; ++a) begin : g_slab
        VX_fma_unit #(.LATENCY (LAT_SLAB)) fma_t0 (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
            .dataa (dmn[a]), .datab (inv_d_q[a]), .datac (32'h0),
            .result (t0[a]), `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(.LATENCY (LAT_SLAB)) fma_t1 (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
            .dataa (dmx[a]), .datab (inv_d_q[a]), .datac (32'h0),
            .result (t1[a]), `UNUSED_PIN (fflags)
        );
    end

    // ── stage 4: per-axis lo/hi ───────────────────────────────────────
    wire [2:0][31:0] lo, hi;
    for (genvar a = 0; a < 3; ++a) begin : g_minmax
        VX_fncp_unit #(.LATENCY (FNCP_SIZE)) fncp_lo (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MISC), .frm (3'd6 /*FMIN*/),
            .dataa (t0[a]), .datab (t1[a]), .result (lo[a]), `UNUSED_PIN (fflags)
        );
        VX_fncp_unit #(.LATENCY (FNCP_SIZE)) fncp_hi (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MISC), .frm (3'd7 /*FMAX*/),
            .dataa (t0[a]), .datab (t1[a]), .result (hi[a]), `UNUSED_PIN (fflags)
        );
    end

    // t_min/t_max delayed to align with lo/hi
    wire [31:0] tmin_r, tmax_r;
    VX_shift_register #(.DATAW (64), .DEPTH (LAT_ORIGIN + LAT_DEQUANT + LAT_SLAB + LAT_MINMAX)) sr_t (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({t_min,  t_max}),
        .data_out ({tmin_r, tmax_r})
    );

    // ── stage 5: reduce — t_near = max(tmin, lo[*]), t_far = min(tmax, hi[*]) ──
    wire [31:0] near_a, near_b, far_a, far_b;     // first reduce level
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) r_near_a (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MISC), .frm (3'd7), .dataa (lo[0]), .datab (lo[1]),
        .result (near_a), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) r_near_b (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MISC), .frm (3'd7), .dataa (lo[2]), .datab (tmin_r),
        .result (near_b), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) r_far_a (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MISC), .frm (3'd6), .dataa (hi[0]), .datab (hi[1]),
        .result (far_a), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) r_far_b (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MISC), .frm (3'd6), .dataa (hi[2]), .datab (tmax_r),
        .result (far_b), `UNUSED_PIN (fflags));

    wire [31:0] t_near_w, t_far_w;                // second reduce level
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) r_near (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MISC), .frm (3'd7), .dataa (near_a), .datab (near_b),
        .result (t_near_w), `UNUSED_PIN (fflags));
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) r_far (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MISC), .frm (3'd6), .dataa (far_a), .datab (far_b),
        .result (t_far_w), `UNUSED_PIN (fflags));

    // ── stage 6: hit = (t_near <= t_far) ──────────────────────────────
    wire [`VX_CFG_XLEN-1:0] cmp_res;
    VX_fncp_unit #(.LATENCY (FNCP_SIZE)) fncp_cmp (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_CMP), .frm (3'd0 /*LE*/),
        .dataa (t_near_w), .datab (t_far_w), .result (cmp_res), `UNUSED_PIN (fflags));

    // carry t_near alongside the compare result, plus the overall valid pipe
    wire [31:0] t_near_cmp;
    VX_shift_register #(.DATAW (32), .DEPTH (LAT_CMP)) sr_tnear (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (t_near_w), .data_out (t_near_cmp)
    );

    reg [LATENCY-1:0] valid_pipe;
    always @(posedge clk) begin
        if (reset) begin
            valid_pipe <= '0;
        end else if (enable) begin
            valid_pipe <= {valid_pipe[LATENCY-2:0], valid_in};
        end
    end

    assign valid_out = valid_pipe[LATENCY-1];
    assign hit       = cmp_res[0];
    assign t_near    = t_near_cmp;

endmodule
