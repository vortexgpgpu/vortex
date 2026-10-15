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
    // Per lane: gather enable, and the component (0..3 = R,G,B,A) it returns.
    input wire [NUM_LANES-1:0][2:0]     req_gather,
    // VX_TEX_REDUCTION_*: the weighted average, or the min/max of the taps.
    input wire [1:0]                    req_reduction,
    input wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_BLEND_FRAC-1:0] req_blends,
    input wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0] req_border,
    input wire [31:0]                   req_bordercolor,
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
    localparam LERP_LAT  = 3;

    wire valid_s0, valid_s1, valid_s2;
    wire [REQ_TAGW-1:0] req_tag_s0, req_tag_s1, req_tag_s2;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][31:0] texel_ul, texel_uh, texel_lod;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_BLEND_FRAC-1:0] req_blends_s0;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][`TEX_BLEND_FRAC-1:0] blend_v_s0, blend_v_s1;
    wire [NUM_LANES-1:0][LODFRAC_W-1:0] lodfrac_s0, lodfrac_s1, lodfrac_s2;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] fmt_texels, sel_texels, fmt_texels_s0;
    wire [NUM_LANES-1:0][31:0] filtered;

    wire stall_out;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_fmt_texels
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            for (genvar j = 0; j < 4; ++j) begin : g_j
                VX_tex_format tex_format (
                    .format    (req_format),
                    .texel_in  (req_data[i][k][j]),
                    .texel_out (fmt_texels[i][k][j])
                );

                // A tap that left the texture takes the border colour instead of
                // what its address returned. It happens after the decode because
                // the border colour is already in the sampler's working format
                // while a fetched texel is still in the texture's.
                assign sel_texels[i][k][j] = req_border[i][k][j] ? req_bordercolor
                                                                 : fmt_texels[i][k][j];
            end
        end
    end

    // textureGather: the chosen channel of level 0's four decoded taps, as bytes
    // in GL gather order -- taps (u-,v+), (u+,v+), (u+,v-), (u-,v-). It skips the
    // blends and joins the result at the end: the input register plus the three
    // lerp levels (u, v, level).
    localparam GATHER_DELAY = 1 + 3 * LERP_LAT;
    wire [NUM_LANES-1:0][31:0] gather_texel, gather_texel_s;
    wire [NUM_LANES-1:0]       gather_en, gather_en_s;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_gather
        wire [1:0] comp = req_gather[i][1:0];
        wire [4:0] shift = (comp == 2'd0) ? 5'd16 : (comp == 2'd1) ? 5'd8 : (comp == 2'd2) ? 5'd0 : 5'd24;
        assign gather_texel[i] = {8'(sel_texels[i][0][0] >> shift), 8'(sel_texels[i][0][1] >> shift),
                                  8'(sel_texels[i][0][3] >> shift), 8'(sel_texels[i][0][2] >> shift)};
        assign gather_en[i] = req_gather[i][2];
    end

    VX_shift_register #(
        .DATAW  (NUM_LANES * 33),
        .DEPTH  (GATHER_DELAY),
        .RESETW (0)
    ) gather_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({gather_en, gather_texel}),
        .data_out ({gather_en_s, gather_texel_s})
    );

    wire [1:0] reduction_s0;

    VX_pipe_register #(
        .DATAW  (1 + REQ_TAGW + 2 + (NUM_LANES * (LODFRAC_W + TEX_NUM_LEVELS * (2 * `TEX_BLEND_FRAC + 4 * 32)))),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_tag,    req_reduction, req_blends,    req_lodfrac, sel_texels}),
        .data_out ({valid_s0,  req_tag_s0, reduction_s0,  req_blends_s0, lodfrac_s0,  fmt_texels_s0})
    );

    // Min/max reduction: per channel, over the taps whose blend weight is not
    // zero -- tap 0 always, tap 1 with a u weight, tap 2 with a v weight, tap 3
    // with both -- then over the two levels when the level weight is not zero.
    // It runs beside the blends and joins the result at the end, one register
    // later than gather starts.
    function automatic [31:0] reduce4x8(input [31:0] a, input [31:0] b, input is_min);
        for (integer c = 0; c < 4; ++c) begin
            reduce4x8[c*8 +: 8] = ((a[c*8 +: 8] < b[c*8 +: 8]) == is_min) ? a[c*8 +: 8] : b[c*8 +: 8];
        end
    endfunction

    wire [NUM_LANES-1:0][31:0] reduced_texel, reduced_texel_s;
    wire reduce_en = (reduction_s0 != 2'(`VX_TEX_REDUCTION_AVG));
    wire reduce_min = (reduction_s0 == 2'(`VX_TEX_REDUCTION_MIN));
    wire reduce_en_s;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_reduce
        wire [TEX_NUM_LEVELS-1:0][31:0] level_texel;
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            wire has_u = (req_blends_s0[i][k][0] != '0);
            wire has_v = (req_blends_s0[i][k][1] != '0);
            wire [31:0] t01 = has_u ? reduce4x8(fmt_texels_s0[i][k][0], fmt_texels_s0[i][k][1], reduce_min)
                                    : fmt_texels_s0[i][k][0];
            wire [31:0] t23 = has_u ? reduce4x8(fmt_texels_s0[i][k][2], fmt_texels_s0[i][k][3], reduce_min)
                                    : fmt_texels_s0[i][k][2];
            assign level_texel[k] = has_v ? reduce4x8(t01, t23, reduce_min) : t01;
        end
        assign reduced_texel[i] = (lodfrac_s0[i] != '0) ? reduce4x8(level_texel[0], level_texel[1], reduce_min)
                                                        : level_texel[0];
    end

    VX_shift_register #(
        .DATAW  (1 + NUM_LANES * 32),
        .DEPTH  (GATHER_DELAY - 1),
        .RESETW (0)
    ) reduce_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({reduce_en,   reduced_texel}),
        .data_out ({reduce_en_s, reduced_texel_s})
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_tex_lerp_U
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            for (genvar j = 0; j < 4; ++j) begin : g_j
                VX_tex_lerp #(
                    .LATENCY (LERP_LAT)
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
                    .LATENCY (LERP_LAT)
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
                    .LATENCY (LERP_LAT)
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

    // Blend the two levels. The level weight truncates rather than rounds,
    // the form the software sampler blends levels in; a sample that moves
    // between the two paths then keeps the same value.
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_tex_lerp_LOD
        for (genvar j = 0; j < 4; ++j) begin : g_j
            VX_tex_lerp #(
                .LATENCY (LERP_LAT),
                .ROUND   (0)
            ) tex_lerp_lod (
                .clk  (clk),
                .reset(reset),
                .enable(~stall_out),
                .in1  (texel_lod[i][0][j*8 +: 8]),
                .in2  (texel_lod[i][1][j*8 +: 8]),
                .frac (lodfrac_s2[i]),
                .out  (filtered[i][j*8 +: 8])
            );
        end
        assign rsp_data[i] = gather_en_s[i] ? gather_texel_s[i]
                           : (reduce_en_s ? reduced_texel_s[i] : filtered[i]);
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
