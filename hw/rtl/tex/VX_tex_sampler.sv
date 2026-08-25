//!/bin/bash

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

`include "VX_tex_define.vh"

module VX_tex_sampler import VX_gpu_pkg::*, VX_tex_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter REQ_TAGW = 1,
    parameter NUM_LANES = 1
) (
    input wire clk,
    input wire reset,

    // inputs
    input wire                          req_valid,
    input wire [TEX_FORMAT_BITS-1:0]   req_format,
    input wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_BLEND_FRAC-1:0] req_blends,
    input wire [NUM_LANES-1:0][`VX_TEX_LOD_FRAC_BITS-1:0] req_lodfrac,
    input wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] req_data,
    input wire [REQ_TAGW-1:0]           req_tag,
    output wire                         req_ready,

    // ouputs
    output wire                         rsp_valid,
    output wire [NUM_LANES-1:0][31:0]   rsp_data,
    output wire [REQ_TAGW-1:0]          rsp_tag,
    input wire                          rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam LODFRAC_W = `VX_TEX_LOD_FRAC_BITS;

    wire valid_s0, valid_s1, valid_s2;
    wire [REQ_TAGW-1:0] req_tag_s0, req_tag_s1, req_tag_s2;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][31:0] texel_ul, texel_uh, texel_lod;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_BLEND_FRAC-1:0] req_blends_s0;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][`TEX_BLEND_FRAC-1:0] blend_v_s0, blend_v_s1;
    wire [NUM_LANES-1:0][LODFRAC_W-1:0] lodfrac_s0, lodfrac_s1, lodfrac_s2;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] fmt_texels, fmt_texels_s0;

    wire stall_out;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_fmt_texels
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            for (genvar j = 0; j < 4; ++j) begin : g_j
                VX_tex_format tex_format (
                    .format    (req_format),
                    .texel_in  (req_data[i][k][j]),
                    .texel_out (fmt_texels[i][k][j])
                );
            end
        end
    end

    VX_pipe_register #(
        .DATAW  (1 + REQ_TAGW + (NUM_LANES * (LODFRAC_W + TEX_NUM_LEVELS * (2 * `TEX_BLEND_FRAC + 4 * 32)))),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_tag,    req_blends,    req_lodfrac, fmt_texels}),
        .data_out ({valid_s0,  req_tag_s0, req_blends_s0, lodfrac_s0,  fmt_texels_s0})
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_tex_lerp_U
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            for (genvar j = 0; j < 4; ++j) begin : g_j
                VX_tex_lerp #(
                    .LATENCY (3)
                ) tex_lerp_ul (
                    .clk  (clk),
                    .reset(reset),
                    .enable(~stall_out),
                    .in1  (fmt_texels_s0[i][k][0][j*8 +: 8]),
                    .in2  (fmt_texels_s0[i][k][1][j*8 +: 8]),
                    .frac (req_blends_s0[i][k][0]),
                    .out  (texel_ul[i][k][j*8 +: 8])
                );
                VX_tex_lerp #(
                    .LATENCY (3)
                ) tex_lerp_uh (
                    .clk  (clk),
                    .reset(reset),
                    .enable(~stall_out),
                    .in1  (fmt_texels_s0[i][k][2][j*8 +: 8]),
                    .in2  (fmt_texels_s0[i][k][3][j*8 +: 8]),
                    .frac (req_blends_s0[i][k][0]),
                    .out  (texel_uh[i][k][j*8 +: 8])
                );
            end
        end
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_blend_v
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            assign blend_v_s0[i][k] = req_blends_s0[i][k][1];
        end
    end

    VX_shift_register #(
        .DATAW  (1 + REQ_TAGW + (NUM_LANES * (LODFRAC_W + TEX_NUM_LEVELS * `TEX_BLEND_FRAC))),
        .DEPTH  (3),
        .RESETW (1)
    ) shift_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s0, req_tag_s0, blend_v_s0, lodfrac_s0}),
        .data_out ({valid_s1, req_tag_s1, blend_v_s1, lodfrac_s1})
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_tex_lerp_V
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            for (genvar j = 0; j < 4; ++j) begin : g_j
                VX_tex_lerp #(
                    .LATENCY (3)
                ) tex_lerp_v (
                    .clk  (clk),
                    .reset(reset),
                    .enable(~stall_out),
                    .in1  (texel_ul[i][k][j*8 +: 8]),
                    .in2  (texel_uh[i][k][j*8 +: 8]),
                    .frac (blend_v_s1[i][k]),
                    .out  (texel_lod[i][k][j*8 +: 8])
                );
            end
        end
    end

    VX_shift_register #(
        .DATAW  (1 + REQ_TAGW + (NUM_LANES * LODFRAC_W)),
        .DEPTH  (3),
        .RESETW (1)
    ) shift_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s1, req_tag_s1, lodfrac_s1}),
        .data_out ({valid_s2, req_tag_s2, lodfrac_s2})
    );

    // Blend the two levels. The weight is a fraction of 256 rather than of 255,
    // which is the form the software sampler blends levels in; a sample that
    // moves between the two paths then keeps the same value.
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_tex_lerp_LOD
        for (genvar j = 0; j < 4; ++j) begin : g_j
            VX_tex_lerp #(
                .LATENCY    (3),
                .FRAC_SCALE (256)
            ) tex_lerp_lod (
                .clk  (clk),
                .reset(reset),
                .enable(~stall_out),
                .in1  (texel_lod[i][0][j*8 +: 8]),
                .in2  (texel_lod[i][1][j*8 +: 8]),
                .frac (lodfrac_s2[i]),
                .out  (rsp_data[i][j*8 +: 8])
            );
        end
    end

    assign stall_out = rsp_valid && ~rsp_ready;

    VX_shift_register #(
        .DATAW  (1 + REQ_TAGW),
        .DEPTH  (3),
        .RESETW (1)
    ) shift_reg3 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s2,  req_tag_s2}),
        .data_out ({rsp_valid, rsp_tag})
    );

    // can accept new request?
    assign req_ready = ~stall_out;

`ifdef DBG_TRACE_TEX
    // The trace macros reach two dimensions, so a level's taps are dumped as
    // their own array rather than one three-dimensional view.
    wire [NUM_LANES-1:0][3:0][31:0] trace_data [TEX_NUM_LEVELS];
    wire [NUM_LANES-1:0][1:0][`TEX_BLEND_FRAC-1:0] trace_blends [TEX_NUM_LEVELS];
    for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_trace
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_i
            assign trace_data[k][i] = req_data[i][k];
            assign trace_blends[k][i] = req_blends[i][k];
        end
    end

    always @(posedge clk) begin
        if (req_valid && req_ready) begin
            `TRACE(2, ("%d: %s-sampler-req: format=%0d, data=", $time, INSTANCE_ID, req_format))
            `TRACE_ARRAY2D(2, "0x%0h", trace_data[0], 4, NUM_LANES)
            `TRACE(2, (", data_up="))
            `TRACE_ARRAY2D(2, "0x%0h", trace_data[1], 4, NUM_LANES)
            `TRACE(2, (", blends="))
            `TRACE_ARRAY2D(2, "0x%0h", trace_blends[0], 2, NUM_LANES)
            `TRACE(2, (", lodfrac="))
            `TRACE_ARRAY1D(2, "0x%0h", req_lodfrac, NUM_LANES)
            `TRACE(2, (" (#%0d)\n", req_tag[REQ_TAGW-1 -: UUID_WIDTH]))
        end
        if (rsp_valid && rsp_ready) begin
            `TRACE(2, ("%d: %s-sampler-rsp: data=", $time, INSTANCE_ID))
            `TRACE_ARRAY1D(2, "0x%0h", rsp_data, NUM_LANES)
            `TRACE(2, (" (#%0d)\n", rsp_tag[REQ_TAGW-1 -: UUID_WIDTH]))
        end
    end
`endif

endmodule
