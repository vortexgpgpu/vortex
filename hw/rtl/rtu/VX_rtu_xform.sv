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

// VX_rtu_xform — world→object ray transform for a TLAS instance. Streams one
// instance's 3x4 affine transform + world ray and emits the object-space ray
// after a fixed latency.
//
//   obj_ro = R^T * (ro - t)        obj_rd = R^T * rd
//
// The instance transform is object→world; its inverse brings the world ray into
// object space. For the orthonormal rotation+translation transforms a TLAS
// carries (every instance in a valid scene), R is orthonormal so R^(-1) = R^T,
// which needs no determinant or division — a pure FMA pipeline. This is bit-
// equivalent to the SimX oracle's explicit cofactor inverse for any orthonormal
// R (the only kind the tests and a valid Vulkan TLAS produce); SimX's singular-
// matrix passthrough is moot here as there is no divide to guard.
//
// Layout of the 3x4 row-major transform (matches the shared host/SimX format):
//   xform[0..2]  = R row 0   xform[3]  = t.x
//   xform[4..6]  = R row 1   xform[7]  = t.y
//   xform[8..10] = R row 2   xform[11] = t.z
// obj_ro[i] = (column i of R) . (ro - t); column i of R = row i of R^T:
//   col0 = {xform[0], xform[4], xform[8]}, etc.
//
// The (ro - t) subtract reuses VX_fma_unit (a*1 - c); the matrix-vector products
// reuse VX_rtu_fdot3. Side-band operands are delayed through shift registers so
// every stage consumes time-aligned inputs at a fixed latency the scheduler
// tracks via valid_out — same structure as VX_rtu_tri_pe / VX_rtu_box_pe.

`include "VX_define.vh"

module VX_rtu_xform import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter LATENCY_FMA = `VX_CFG_LATENCY_FMA,
    parameter TAG_WIDTH   = 1
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             enable,
    input  wire             valid_in,
    input  wire [TAG_WIDTH-1:0] tag_in,    // caller side-band (e.g. context id)

    input  wire [11:0][31:0] xform,        // 3x4 row-major affine (object→world)
    input  wire [2:0][31:0]  ro,           // world ray origin
    input  wire [2:0][31:0]  rd,           // world ray direction

    output wire             valid_out,
    output wire [TAG_WIDTH-1:0] tag_out,
    output wire [2:0][31:0] obj_ro,        // object-space ray origin
    output wire [2:0][31:0] obj_rd         // object-space ray direction
);
    localparam F       = LATENCY_FMA;
    localparam LATENCY = 4 * F;            // (ro-t) subtract @F, then dot @3F

    localparam [INST_FMT_BITS-1:0] FMT_SUB = 2'b10;   // F32, a*b - c
    localparam [31:0] FP_ONE = 32'h3F800000;

    // translation vector t = {xform[3], xform[7], xform[11]}.
    wire [2:0][31:0] tvec;
    assign tvec[0] = xform[3];
    assign tvec[1] = xform[7];
    assign tvec[2] = xform[11];

    // R columns (rows of R^T): col_i[j] = xform[4*j + i].
    wire [2:0][2:0][31:0] col;
    for (genvar i = 0; i < 3; ++i) begin : g_col
        for (genvar j = 0; j < 3; ++j) begin : g_col_e
            assign col[i][j] = xform[4*j + i];
        end
    end

    // ── stage 1 (@F): d = ro - t (per axis), reusing the FMA as a*1 - c ──
    wire [2:0][31:0] d;
    for (genvar a = 0; a < 3; ++a) begin : g_sub
        VX_fma_unit #(.LATENCY (F)) fma_d (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MADD), .fmt (FMT_SUB), .frm (INST_FRM_RNE),
            .dataa (ro[a]), .datab (FP_ONE), .datac (tvec[a]),
            .result (d[a]), `UNUSED_PIN (fflags)
        );
    end

    // R columns aligned from @0 to @F to feed the dot products.
    wire [2:0][2:0][31:0] col_d;
    VX_shift_register #(.DATAW (9*32), .DEPTH (F)) sr_col (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (col), .data_out (col_d)
    );
    // rd aligned from @0 to @F so the direction dot starts in lock-step with d.
    wire [2:0][31:0] rd_d;
    VX_shift_register #(.DATAW (96), .DEPTH (F)) sr_rd (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (rd), .data_out (rd_d)
    );

    // ── stage 2 (@F+3F = @4F): obj_ro[i] = col_i . d, obj_rd[i] = col_i . rd ──
    for (genvar i = 0; i < 3; ++i) begin : g_dot
        VX_rtu_fdot3 #(.LATENCY_FMA (F)) dot_ro (
            .clk (clk), .reset (reset), .enable (enable),
            .a (col_d[i]), .b (d), .result (obj_ro[i])
        );
        VX_rtu_fdot3 #(.LATENCY_FMA (F)) dot_rd (
            .clk (clk), .reset (reset), .enable (enable),
            .a (col_d[i]), .b (rd_d), .result (obj_rd[i])
        );
    end

    // ── valid + tag pipe, sized to the whole datapath latency ─────────
    reg [LATENCY-1:0] valid_pipe;
    always @(posedge clk) begin
        if (reset) begin
            valid_pipe <= '0;
        end else if (enable) begin
            valid_pipe <= {valid_pipe[LATENCY-2:0], valid_in};
        end
    end

    wire [TAG_WIDTH-1:0] tag_out_w;
    VX_shift_register #(.DATAW (TAG_WIDTH), .DEPTH (LATENCY)) sr_tag (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (tag_in), .data_out (tag_out_w)
    );

    assign valid_out = valid_pipe[LATENCY-1];
    assign tag_out   = tag_out_w;

endmodule
