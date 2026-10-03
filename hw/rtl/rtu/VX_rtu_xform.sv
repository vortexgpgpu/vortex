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
// instance's world→object 3x4 matrix + world ray and emits the object-space ray
// after a fixed latency.
//
// The instance record carries the world→object matrix the source driver
// (lavapipe) builds, and the ray is transformed in the order it does: F32,
// every product rounded, then
//
//   obj_ro[i] = ((m[i][3] + ro.x*m[i][0]) + ro.y*m[i][1]) + ro.z*m[i][2]
//   obj_rd[i] =  (rd.x*m[i][0] + rd.y*m[i][1]) + rd.z*m[i][2]
//
// so the object ray matches it bit for bit for any affine instance (scale,
// shear included), with no inverse taken anywhere. Layout: m[i][j] = xform[4*i
// + j], row-major, translation in column 3.

`include "VX_define.vh"

module VX_rtu_xform import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter LATENCY_FMA = RTU_LATENCY_FMA,
    parameter TAG_WIDTH   = 1
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             enable,
    input  wire             valid_in,
    input  wire [TAG_WIDTH-1:0] tag_in,    // caller side-band (e.g. context id)

    input  wire [11:0][31:0] xform,        // 3x4 row-major affine (world→object)
    input  wire [2:0][31:0]  ro,           // world ray origin
    input  wire [2:0][31:0]  rd,           // world ray direction

    output wire             valid_out,
    output wire [TAG_WIDTH-1:0] tag_out,
    output wire [2:0][31:0] obj_ro,        // object-space ray origin
    output wire [2:0][31:0] obj_rd         // object-space ray direction
);
    localparam F       = LATENCY_FMA;
    localparam LATENCY = 4 * F;            // products, then three dependent adds

    localparam [INST_FMT_BITS-1:0] FMT_ADD = 2'b00;

    // ── @0 → @F: every product, rounded ───────────────────────────────
    wire [2:0][2:0][31:0] po, pd;          // [row][column]
    for (genvar i = 0; i < 3; ++i) begin : g_row
        for (genvar j = 0; j < 3; ++j) begin : g_col
            VX_fma_unit #(
                .USE_DSP        (`VX_CFG_RTU_USE_DSP),
                .LATENCY        (F),
                .SUBNORM_ENABLE (0),
                .EXCEPT_ENABLE  (1)
            ) fmul_o (
                .clk     (clk),
                .reset   (reset),
                .enable  (enable),
                .mask    (1'b1),
                .op_type (INST_FPU_MUL),
                .fmt     (FMT_ADD),
                .frm     (INST_FRM_RNE),
                .dataa   (ro[j]),
                .datab   (xform[4*i + j]),
                .datac   ('0),
                .result  (po[i][j]),
                `UNUSED_PIN (fflags)
            );
            VX_fma_unit #(
                .USE_DSP        (`VX_CFG_RTU_USE_DSP),
                .LATENCY        (F),
                .SUBNORM_ENABLE (0),
                .EXCEPT_ENABLE  (1)
            ) fmul_d (
                .clk     (clk),
                .reset   (reset),
                .enable  (enable),
                .mask    (1'b1),
                .op_type (INST_FPU_MUL),
                .fmt     (FMT_ADD),
                .frm     (INST_FRM_RNE),
                .dataa   (rd[j]),
                .datab   (xform[4*i + j]),
                .datac   ('0),
                .result  (pd[i][j]),
                `UNUSED_PIN (fflags)
            );
        end
    end

    wire [2:0][31:0] tr_d;                 // translation column @F
    VX_shift_register #(
        .DATAW (3*32),
        .DEPTH (F)
    ) sr_tr (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({xform[11], xform[7], xform[3]}),
        .data_out (tr_d)
    );

    // later addends held until their add issues
    wire [2:0][31:0] po1_d, po2_d, pd2_d;  // po[.][1] @2F, po[.][2] @3F, pd[.][2] @2F
    VX_shift_register #(
        .DATAW (2*3*32),
        .DEPTH (F)
    ) sr_p1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({po[2][1], po[1][1], po[0][1], pd[2][2], pd[1][2], pd[0][2]}),
        .data_out ({po1_d, pd2_d})
    );
    VX_shift_register #(
        .DATAW (3*32),
        .DEPTH (2 * F)
    ) sr_p2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({po[2][2], po[1][2], po[0][2]}),
        .data_out (po2_d)
    );

    // ── the dependent adds, one per F ─────────────────────────────────
    wire [2:0][31:0] o1, o2, d1, d2;
    for (genvar i = 0; i < 3; ++i) begin : g_sum
        VX_fma_unit #(.USE_DSP (`VX_CFG_RTU_USE_DSP), .LATENCY (F), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fadd_o1 (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_ADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
            .dataa (tr_d[i]), .datab (po[i][0]), .datac ('0),
            .result (o1[i]), `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(.USE_DSP (`VX_CFG_RTU_USE_DSP), .LATENCY (F), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fadd_o2 (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_ADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
            .dataa (o1[i]), .datab (po1_d[i]), .datac ('0),
            .result (o2[i]), `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(.USE_DSP (`VX_CFG_RTU_USE_DSP), .LATENCY (F), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fadd_o3 (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_ADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
            .dataa (o2[i]), .datab (po2_d[i]), .datac ('0),
            .result (obj_ro[i]), `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(.USE_DSP (`VX_CFG_RTU_USE_DSP), .LATENCY (F), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fadd_d1 (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_ADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
            .dataa (pd[i][0]), .datab (pd[i][1]), .datac ('0),
            .result (d1[i]), `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(.USE_DSP (`VX_CFG_RTU_USE_DSP), .LATENCY (F), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fadd_d2 (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_ADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
            .dataa (d1[i]), .datab (pd2_d[i]), .datac ('0),
            .result (d2[i]), `UNUSED_PIN (fflags)
        );
    end

    VX_shift_register #(
        .DATAW (3*32),
        .DEPTH (F)
    ) sr_d (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (d2),
        .data_out (obj_rd)
    );

    // ── valid + tag pipe, sized to the whole datapath latency ─────────
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

    assign valid_out = valid_pipe_r[LATENCY-1];
    assign tag_out   = tag_out_w;

endmodule
