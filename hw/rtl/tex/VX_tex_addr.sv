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

// Texel addressing in texel space, for any texture size. Each axis's
// coordinate is folded by its wrap in normalized space -- repeat and mirror
// into [0,1), clamp and border into [-1,2) -- then scaled by the level's
// dimension, with bilinear moving it half a texel down. The two taps are
// floor(x) and floor(x)+1, each wrapped back into [0,dim) as an integer, and
// the fraction of x is the weight of the second.
module VX_tex_addr import VX_gpu_pkg::*, VX_tex_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter REQ_TAGW    = 1,
    parameter NUM_LANES   = 1,
    parameter W_ADDR_BITS = `TEX_ADDR_BITS + 6
) (
    input wire clk,
    input wire reset,

    // inputs

    input wire                          req_valid,
    input wire [NUM_LANES-1:0]          req_mask,
    input wire [1:0][NUM_LANES-1:0][`VX_TEX_FXD_BITS-1:0] req_coords,
    input wire [TEX_FORMAT_BITS-1:0]   req_format,
    input wire [NUM_LANES-1:0][TEX_FILTER_BITS-1:0] req_filter,
    input wire [1:0][TEX_WRAP_BITS-1:0] req_wraps,
    input wire [NUM_LANES-1:0][W_ADDR_BITS-1:0] req_baseaddr,
    input wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][TEX_LOD_BITS-1:0] req_miplevel,
    input wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][`TEX_MIPOFF_BITS-1:0] req_mipoff,
    input wire [NUM_LANES-1:0][`VX_TEX_LOD_FRAC_BITS-1:0] req_lodfrac,
    input wire [1:0][TEX_DIM_W-1:0]     req_dims,
    input wire [REQ_TAGW-1:0]           req_tag,
    output wire                         req_ready,

    // outputs

    output wire                         rsp_valid,
    output wire [NUM_LANES-1:0]         rsp_mask,
    output wire [NUM_LANES-1:0][TEX_FILTER_BITS-1:0] rsp_filter,
    output wire [`TEX_LGSTRIDE_BITS-1:0] rsp_lgstride,
    output wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][W_ADDR_BITS-1:0] rsp_baseaddr,
    output wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] rsp_addr,
    output wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_BLEND_FRAC-1:0] rsp_blends,
    output wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0] rsp_border,
    output wire [NUM_LANES-1:0][`VX_TEX_LOD_FRAC_BITS-1:0] rsp_lodfrac,
    output wire [REQ_TAGW-1:0]          rsp_tag,
    input wire                          rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // A folded coordinate is signed and spans [-1,2).
    localparam FOLD_W  = `TEX_FXD_FRAC + 2;
    localparam PROD_W  = FOLD_W + TEX_DIM_W + 1;
    // Texel-space position with TEX_BLEND_FRAC fraction bits.
    localparam X_W     = PROD_W - (`TEX_FXD_FRAC - `TEX_BLEND_FRAC);
    localparam IDX_W   = X_W - `TEX_BLEND_FRAC;
    localparam TEXEL_W = `VX_TEX_DIM_BITS;
    localparam PITCH_W = TEX_DIM_W + `TEX_LGSTRIDE_MAX;

    localparam signed [FOLD_W-1:0] FOLD_MIN = -FOLD_W'(`TEX_FXD_ONE);
    localparam signed [FOLD_W-1:0] FOLD_MAX = FOLD_W'(2 * `TEX_FXD_ONE - 1);

    wire stall_out;

    // stage 0: fold the coordinates, scale the dimensions to each level

    wire [`TEX_LGSTRIDE_BITS-1:0] log_stride;

    VX_tex_stride tex_stride (
        .format     (req_format),
        .log_stride (log_stride)
    );

    wire [NUM_LANES-1:0][1:0][FOLD_W-1:0] folded;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][TEX_DIM_W-1:0] level_dim;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][W_ADDR_BITS-1:0] mip_addr;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_fold
        for (genvar j = 0; j < 2; ++j) begin : g_j
            wire [`VX_TEX_FXD_BITS-1:0] c = req_coords[j][i];
            wire [`TEX_FXD_FRAC-1:0] frac = c[`TEX_FXD_FRAC-1:0];
            reg [FOLD_W-1:0] f;
            always @(*) begin
                case (req_wraps[j])
                    `VX_TEX_WRAP_REPEAT: f = FOLD_W'(frac);
                    `VX_TEX_WRAP_MIRROR: f = FOLD_W'(frac ^ {`TEX_FXD_FRAC{c[`TEX_FXD_FRAC]}});
                    default: begin
                        if ($signed(c) < $signed(`VX_TEX_FXD_BITS'(FOLD_MIN))) begin
                            f = FOLD_MIN;
                        end else if ($signed(c) > $signed(`VX_TEX_FXD_BITS'(FOLD_MAX))) begin
                            f = FOLD_MAX;
                        end else begin
                            f = c[FOLD_W-1:0];
                        end
                    end
                endcase
            end
            assign folded[i][j] = f;
        end
    end

    // A level past the end of the chain is one texel wide, which is what the
    // offset table, saturating on its last entry, addresses.
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_level_dim
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            for (genvar j = 0; j < 2; ++j) begin : g_j
                wire [TEX_DIM_W-1:0] d = req_dims[j] >> req_miplevel[i][k];
                assign level_dim[i][k][j] = (d != 0) ? d : TEX_DIM_W'(1);
            end
            assign mip_addr[i][k] = req_baseaddr[i] + W_ADDR_BITS'(req_mipoff[i][k]);
        end
    end

    wire                 valid_s0;
    wire [NUM_LANES-1:0] mask_s0;
    wire [NUM_LANES-1:0][TEX_FILTER_BITS-1:0] filter_s0;
    wire [1:0][TEX_WRAP_BITS-1:0] wraps_s0;
    wire [`TEX_LGSTRIDE_BITS-1:0] log_stride_s0;
    wire [NUM_LANES-1:0][1:0][FOLD_W-1:0] folded_s0;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][TEX_DIM_W-1:0] level_dim_s0;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][W_ADDR_BITS-1:0] mip_addr_s0;
    wire [NUM_LANES-1:0][`VX_TEX_LOD_FRAC_BITS-1:0] lodfrac_s0;
    wire [REQ_TAGW-1:0] req_tag_s0;

    VX_pipe_register #(
        .DATAW  (1 + NUM_LANES + 2 * TEX_WRAP_BITS + `TEX_LGSTRIDE_BITS + REQ_TAGW + NUM_LANES * (TEX_FILTER_BITS + `VX_TEX_LOD_FRAC_BITS + 2 * FOLD_W + TEX_NUM_LEVELS * (2 * TEX_DIM_W + W_ADDR_BITS))),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_mask, req_filter, req_wraps, log_stride,    req_tag,    folded,    level_dim,    mip_addr,    req_lodfrac}),
        .data_out ({valid_s0,  mask_s0,  filter_s0,  wraps_s0,  log_stride_s0, req_tag_s0, folded_s0, level_dim_s0, mip_addr_s0, lodfrac_s0})
    );

    // stage 1: scale to texels, pick the two taps of each axis and wrap them

    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][1:0][TEXEL_W-1:0] texel;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][1:0] texel_border;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_BLEND_FRAC-1:0] blends;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0] border_mask;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0] row1_zero, row1_same;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_taps
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            for (genvar j = 0; j < 2; ++j) begin : g_j
                wire signed [PROD_W-1:0] prod = $signed(folded_s0[i][j]) * $signed({1'b0, level_dim_s0[i][k][j]});
                wire signed [X_W-1:0] x_pos = X_W'(prod >>> (`TEX_FXD_FRAC - `TEX_BLEND_FRAC));
                wire signed [X_W-1:0] x = filter_s0[i][0] ? (x_pos - X_W'(`TEX_BLEND_ONE / 2)) : x_pos;
                wire signed [IDX_W-1:0] i0 = IDX_W'(x >>> `TEX_BLEND_FRAC);
                wire signed [IDX_W-1:0] dim = $signed(IDX_W'(level_dim_s0[i][k][j]));

                for (genvar t = 0; t < 2; ++t) begin : g_t
                    wire signed [IDX_W-1:0] it = i0 + IDX_W'(t);
                    wire below = (it < 0);
                    wire above = (it >= dim);
                    reg [IDX_W-1:0] wrapped;
                    always @(*) begin
                        if (~below && ~above) begin
                            wrapped = it;
                        end else if (wraps_s0[j] == `VX_TEX_WRAP_REPEAT) begin
                            wrapped = below ? (it + dim) : (it - dim);
                        end else begin
                            wrapped = below ? IDX_W'(0) : (dim - IDX_W'(1));
                        end
                    end
                    assign texel[i][k][j][t] = TEXEL_W'(wrapped);
                    assign texel_border[i][k][j][t] = (wraps_s0[j] == `VX_TEX_WRAP_BORDER) && (below || above);
                    `UNUSED_VAR (wrapped)
                end

                assign blends[i][k][j] = filter_s0[i][0] ? x[`TEX_BLEND_FRAC-1:0] : `TEX_BLEND_FRAC'(0);
                `UNUSED_VAR (prod)
                `UNUSED_VAR (x)
            end

            // Tap order is the one the addresses below are built in:
            // 0=(u0,v0) 1=(u1,v0) 2=(u0,v1) 3=(u1,v1). Under POINT only tap 0
            // is consumed.
            assign border_mask[i][k][0] = texel_border[i][k][0][0] || texel_border[i][k][1][0];
            assign border_mask[i][k][1] = texel_border[i][k][0][1] || texel_border[i][k][1][0];
            assign border_mask[i][k][2] = texel_border[i][k][0][0] || texel_border[i][k][1][1];
            assign border_mask[i][k][3] = texel_border[i][k][0][1] || texel_border[i][k][1][1];

            // The second row is the first one plus a pitch, the first one again
            // (an edge clamped both taps onto it), or row 0 (repeat wrapped it),
            // so it needs no multiplier of its own.
            assign row1_zero[i][k] = (texel[i][k][1][1] == TEXEL_W'(0));
            assign row1_same[i][k] = (texel[i][k][1][1] == texel[i][k][1][0]);
        end
    end

    wire                 valid_s1;
    wire [NUM_LANES-1:0] mask_s1;
    wire [NUM_LANES-1:0][TEX_FILTER_BITS-1:0] filter_s1;
    wire [`TEX_LGSTRIDE_BITS-1:0] log_stride_s1;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][1:0][TEXEL_W-1:0] texel_s1;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][TEX_DIM_W-1:0] width_s1;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0] row1_zero_s1, row1_same_s1;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_BLEND_FRAC-1:0] blends_s1;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0] border_mask_s1;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][W_ADDR_BITS-1:0] mip_addr_s1;
    wire [NUM_LANES-1:0][`VX_TEX_LOD_FRAC_BITS-1:0] lodfrac_s1;
    wire [REQ_TAGW-1:0] req_tag_s1;

    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][TEX_DIM_W-1:0] width_s0;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_width
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            assign width_s0[i][k] = level_dim_s0[i][k][0];
        end
    end

    VX_pipe_register #(
        .DATAW  (1 + NUM_LANES + `TEX_LGSTRIDE_BITS + REQ_TAGW + NUM_LANES * (TEX_FILTER_BITS + `VX_TEX_LOD_FRAC_BITS + TEX_NUM_LEVELS * (4 * TEXEL_W + TEX_DIM_W + 2 + 2 * `TEX_BLEND_FRAC + 4 + W_ADDR_BITS))),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s0, mask_s0, filter_s0, log_stride_s0, req_tag_s0, texel,    width_s0, row1_zero,    row1_same,    blends,    border_mask,    mip_addr_s0, lodfrac_s0}),
        .data_out ({valid_s1, mask_s1, filter_s1, log_stride_s1, req_tag_s1, texel_s1, width_s1, row1_zero_s1, row1_same_s1, blends_s1, border_mask_s1, mip_addr_s1, lodfrac_s1})
    );

    // stage 2: byte offsets of the four taps

    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] addr;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_addr
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            wire [PITCH_W-1:0] pitch = PITCH_W'(width_s1[i][k]) << log_stride_s1;
            wire [31:0] row0 = 32'(texel_s1[i][k][1][0]) * 32'(pitch);
            wire [31:0] row1 = row1_zero_s1[i][k] ? 32'(0)
                             : (row1_same_s1[i][k] ? row0 : (row0 + 32'(pitch)));
            wire [31:0] col0 = 32'(texel_s1[i][k][0][0]) << log_stride_s1;
            wire [31:0] col1 = 32'(texel_s1[i][k][0][1]) << log_stride_s1;
            assign addr[i][k][0] = row0 + col0;
            assign addr[i][k][1] = row0 + col1;
            assign addr[i][k][2] = row1 + col0;
            assign addr[i][k][3] = row1 + col1;
        end
    end

    assign stall_out = rsp_valid && ~rsp_ready;

    VX_pipe_register #(
        .DATAW  (1 + NUM_LANES + `TEX_LGSTRIDE_BITS + REQ_TAGW + NUM_LANES * (TEX_FILTER_BITS + `VX_TEX_LOD_FRAC_BITS + TEX_NUM_LEVELS * (4 + W_ADDR_BITS + 4 * 32 + 2 * `TEX_BLEND_FRAC))),
        .RESETW (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s1,  mask_s1,  filter_s1,  log_stride_s1, mip_addr_s1,  addr,     blends_s1,  border_mask_s1, lodfrac_s1,  req_tag_s1}),
        .data_out ({rsp_valid, rsp_mask, rsp_filter, rsp_lgstride,  rsp_baseaddr, rsp_addr, rsp_blends, rsp_border,     rsp_lodfrac, rsp_tag})
    );

    assign req_ready = ~stall_out;

`ifdef DBG_TRACE_TEX
    // The trace macros reach two dimensions, so each level's taps are viewed
    // separately rather than as one three-dimensional dump.
    wire [NUM_LANES-1:0][3:0][31:0] trace_addr [TEX_NUM_LEVELS];
    for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_trace_addr
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_i
            assign trace_addr[k][i] = rsp_addr[i][k];
        end
    end

    always @(posedge clk) begin
        if (req_valid && ~stall_out) begin
            `TRACE(2, ("%d: *** %s-addr: mip_addr=", $time, INSTANCE_ID))
            `TRACE_ARRAY2D(2, "0x%0h", mip_addr, TEX_NUM_LEVELS, NUM_LANES)
            `TRACE(2, (", req_dims="))
            `TRACE_ARRAY1D(2, "%0d", req_dims, 2)
            `TRACE(2, (", lodfrac="))
            `TRACE_ARRAY1D(2, "0x%0h", req_lodfrac, NUM_LANES)
            `TRACE(2, ("\n"))
        end

        if (rsp_valid && rsp_ready) begin
            `TRACE(2, ("%d: %s-addr: valid=%b, req_filter=0x%0h, lgstride=%0d, addr=", $time, INSTANCE_ID, rsp_mask, rsp_filter, rsp_lgstride))
            `TRACE_ARRAY2D(2, "0x%0h", trace_addr[0], 4, NUM_LANES)
            `TRACE(2, (", addr_up="))
            `TRACE_ARRAY2D(2, "0x%0h", trace_addr[1], 4, NUM_LANES)
            `TRACE(2, (" (#%0d)\n", rsp_tag[REQ_TAGW-1 -: UUID_WIDTH]))
        end
    end
`endif

endmodule
