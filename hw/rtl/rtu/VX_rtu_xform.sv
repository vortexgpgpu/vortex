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
// after a fixed latency: three dependent FMAs per component,
//
//   obj_ro[i] = fma(ro.z, m[i][2], fma(ro.y, m[i][1], fma(ro.x, m[i][0], m[i][3])))
//   obj_rd[i] = fma(rd.z, m[i][2], fma(rd.y, m[i][1], rd.x * m[i][0]))
//
// The instance record carries the world→object matrix itself, so no inverse
// is taken anywhere; the direction is not renormalised, so t is the same in
// both spaces. Layout: m[i][j] = xform[4*i + j], row-major, translation in
// column 3. Mirrors SimX rtu::world_to_object_ray.

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
    localparam LATENCY = 3 * F;            // three dependent FMAs

    localparam [INST_FMT_BITS-1:0] FMT_ADD = 2'b00;

    // column j of the matrix and the ray's j component, held until step j
    wire [2:0][2:0][31:0] col;             // [j][row]
    for (genvar j = 0; j < 3; ++j) begin : g_col
        for (genvar i = 0; i < 3; ++i) begin : g_row
            assign col[j][i] = xform[4*i + j];
        end
    end
    wire [2:0][31:0] col1_d, col2_d;
    wire [1:0][31:0] ray1_d, ray2_d;       // {rd, ro} component y, then z
    VX_shift_register #(
        .DATAW (3*32 + 2*32),
        .DEPTH (F)
    ) sr_step1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({col[1], rd[1], ro[1]}),
        .data_out ({col1_d, ray1_d})
    );
    VX_shift_register #(
        .DATAW (3*32 + 2*32),
        .DEPTH (2 * F)
    ) sr_step2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({col[2], rd[2], ro[2]}),
        .data_out ({col2_d, ray2_d})
    );

    // [row][0: origin, 1: direction] accumulators after each step
    wire [2:0][1:0][31:0] acc0, acc1, acc2;
    for (genvar i = 0; i < 3; ++i) begin : g_row
        for (genvar k = 0; k < 2; ++k) begin : g_ray
            // step 0: origin seeds with the translation, direction with -0
            // (so the product keeps its own zero sign)
            VX_fma_unit #(.USE_DSP (`VX_CFG_RTU_USE_DSP), .LATENCY (F), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fma_s0 (
                .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
                .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
                .dataa ((k == 0) ? ro[0] : rd[0]), .datab (col[0][i]),
                .datac ((k == 0) ? xform[4*i + 3] : 32'h80000000),
                .result (acc0[i][k]), `UNUSED_PIN (fflags)
            );
            VX_fma_unit #(.USE_DSP (`VX_CFG_RTU_USE_DSP), .LATENCY (F), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fma_s1 (
                .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
                .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
                .dataa (ray1_d[k]), .datab (col1_d[i]), .datac (acc0[i][k]),
                .result (acc1[i][k]), `UNUSED_PIN (fflags)
            );
            VX_fma_unit #(.USE_DSP (`VX_CFG_RTU_USE_DSP), .LATENCY (F), .SUBNORM_ENABLE (0), .EXCEPT_ENABLE (1)) fma_s2 (
                .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
                .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
                .dataa (ray2_d[k]), .datab (col2_d[i]), .datac (acc1[i][k]),
                .result (acc2[i][k]), `UNUSED_PIN (fflags)
            );
        end
        assign obj_ro[i] = acc2[i][0];
        assign obj_rd[i] = acc2[i][1];
    end

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
