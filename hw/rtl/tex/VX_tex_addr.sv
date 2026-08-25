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
    input wire [TEX_FILTER_BITS-1:0]   req_filter,
    input wire [1:0][TEX_WRAP_BITS-1:0] req_wraps,
    input wire [`TEX_ADDR_BITS-1:0]     req_baseaddr,
    input wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][TEX_LOD_BITS-1:0] req_miplevel,
    input wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][`TEX_MIPOFF_BITS-1:0] req_mipoff,
    input wire [NUM_LANES-1:0][`VX_TEX_LOD_FRAC_BITS-1:0] req_lodfrac,
    input wire [1:0][TEX_LOD_BITS-1:0] req_logdims,
    input wire [REQ_TAGW-1:0]           req_tag,
    output wire                         req_ready,

    // outputs

    output wire                         rsp_valid,
    output wire [NUM_LANES-1:0]         rsp_mask,
    output wire [TEX_FILTER_BITS-1:0]  rsp_filter,
    output wire [`TEX_LGSTRIDE_BITS-1:0] rsp_lgstride,
    output wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][W_ADDR_BITS-1:0] rsp_baseaddr,
    output wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] rsp_addr,
    output wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_BLEND_FRAC-1:0] rsp_blends,
    output wire [NUM_LANES-1:0][`VX_TEX_LOD_FRAC_BITS-1:0] rsp_lodfrac,
    output wire [REQ_TAGW-1:0]          rsp_tag,
    input wire                          rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam SHIFT_BITS = `CLOG2(`TEX_FXD_FRAC+1);
    localparam PITCH_BITS = `MAX(TEX_LOD_BITS, `TEX_LGSTRIDE_BITS) + 1;
    localparam SCALED_DIM = `TEX_FXD_FRAC + `VX_TEX_DIM_BITS;
    localparam SCALED_X_W = `VX_TEX_DIM_BITS + `TEX_BLEND_FRAC;
    localparam OFFSET_U_W = `VX_TEX_DIM_BITS + `TEX_LGSTRIDE_MAX;
    localparam OFFSET_V_W = `VX_TEX_DIM_BITS + `VX_TEX_DIM_BITS + `TEX_LGSTRIDE_MAX;

    // Every quantity below that depends on the mip level is computed once per
    // level, so one request carries both levels' tap sets. The levels differ
    // only in that dependence -- same coordinates, same wrap, same format.
    wire                 valid_s0;
    wire [NUM_LANES-1:0] mask_s0;
    wire [TEX_FILTER_BITS-1:0] filter_s0;
    wire [REQ_TAGW-1:0] req_tag_s0;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_FXD_FRAC-1:0] clamped_lo, clamped_lo_s0;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_FXD_FRAC-1:0] clamped_hi, clamped_hi_s0;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][SHIFT_BITS-1:0] dim_shift, dim_shift_s0;
    wire [`TEX_LGSTRIDE_BITS-1:0] log_stride, log_stride_s0;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][W_ADDR_BITS-1:0] mip_addr, mip_addr_s0;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][PITCH_BITS-1:0] log_pitch, log_pitch_s0;
    wire [NUM_LANES-1:0][`VX_TEX_LOD_FRAC_BITS-1:0] lodfrac_s0;

    wire stall_out;

    // stride

    VX_tex_stride tex_stride (
        .format     (req_format),
        .log_stride (log_stride)
    );

    // addressing mode

    // A sampler's lod clamp is independent of how long the mip chain is, so the
    // unit can be asked for a level below 1x1, and a two-level sample reaches
    // that one level sooner than a one-level one. `req_logdims - req_miplevel`
    // then goes negative and wraps. It needs no floor: the wrap only ever lands
    // dim_shift in [16, 30], every one of which shifts the whole coordinate out
    // of the integer field, leaving texel index zero -- the single texel such a
    // level has. The offset table saturates on its last entry for the same
    // reason, so the address is the top level's, which is what the software
    // sampler returns.
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_clamp
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            for (genvar j = 0; j < 2; ++j) begin  : g_j
                wire [`TEX_FXD_FRAC-1:0] delta = `TEX_FXD_FRAC'((SCALED_DIM'(`TEX_FXD_HALF) << req_miplevel[i][k]) >> req_logdims[j]);
                wire [`VX_TEX_FXD_BITS-1:0] coord_lo = req_filter[0] ? (req_coords[j][i] - `VX_TEX_FXD_BITS'(delta)) : req_coords[j][i];
                wire [`VX_TEX_FXD_BITS-1:0] coord_hi = req_filter[0] ? (req_coords[j][i] + `VX_TEX_FXD_BITS'(delta)) : req_coords[j][i];

                VX_tex_wrap tex_wrap_lo (
                    .wrap_i  (req_wraps[j]),
                    .coord_i (coord_lo),
                    .coord_o (clamped_lo[i][k][j])
                );

                VX_tex_wrap tex_wrap_hi (
                    .wrap_i  (req_wraps[j]),
                    .coord_i (coord_hi),
                    .coord_o (clamped_hi[i][k][j])
                );
            end
        end
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_dim_shift
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            for (genvar j = 0; j < 2; ++j) begin : g_j
                assign dim_shift[i][k][j] = SHIFT_BITS'(`TEX_FXD_FRAC - `TEX_BLEND_FRAC) - (req_logdims[j] - req_miplevel[i][k]);
            end
        end
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_log_pitch
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            assign log_pitch[i][k] = PITCH_BITS'(req_logdims[0] - req_miplevel[i][k]) + PITCH_BITS'(log_stride);
        end
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mip_addr
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            assign mip_addr[i][k] = {req_baseaddr, 6'b0} + W_ADDR_BITS'(req_mipoff[i][k]);
        end
    end

    VX_pipe_register #(
        .DATAW  (1 + NUM_LANES + TEX_FILTER_BITS + `TEX_LGSTRIDE_BITS + REQ_TAGW + NUM_LANES * (`VX_TEX_LOD_FRAC_BITS + TEX_NUM_LEVELS * (PITCH_BITS + 2 * SHIFT_BITS + W_ADDR_BITS + 2 * 2 * `TEX_FXD_FRAC))),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_mask, req_filter, log_stride,    req_tag,    log_pitch,    dim_shift,    mip_addr,    clamped_lo,    clamped_hi,    req_lodfrac}),
        .data_out ({valid_s0,  mask_s0,  filter_s0,  log_stride_s0, req_tag_s0, log_pitch_s0, dim_shift_s0, mip_addr_s0, clamped_lo_s0, clamped_hi_s0, lodfrac_s0})
    );

    // addresses generation

    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][SCALED_X_W-1:0] scaled_lo;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][SCALED_X_W-1:0] scaled_hi;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][OFFSET_U_W-1:0] offset_u_lo;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][OFFSET_U_W-1:0] offset_u_hi;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][OFFSET_V_W-1:0] offset_v_lo;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][OFFSET_V_W-1:0] offset_v_hi;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_BLEND_FRAC-1:0] blends;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] addr;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_scaled
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            for (genvar j = 0; j < 2; ++j) begin : g_j
                assign scaled_lo[i][k][j] = SCALED_X_W'(clamped_lo_s0[i][k][j] >> dim_shift_s0[i][k][j]);
                assign scaled_hi[i][k][j] = SCALED_X_W'(clamped_hi_s0[i][k][j] >> dim_shift_s0[i][k][j]);
                assign blends[i][k][j] = filter_s0[0] ? scaled_lo[i][k][j][`TEX_BLEND_FRAC-1:0] : `TEX_BLEND_FRAC'(0);
            end
        end
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_offset
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            assign offset_u_lo[i][k] = OFFSET_U_W'(scaled_lo[i][k][0][`TEX_BLEND_FRAC +: `VX_TEX_DIM_BITS]) << log_stride_s0;
            assign offset_u_hi[i][k] = OFFSET_U_W'(scaled_hi[i][k][0][`TEX_BLEND_FRAC +: `VX_TEX_DIM_BITS]) << log_stride_s0;
            assign offset_v_lo[i][k] = OFFSET_V_W'(scaled_lo[i][k][1][`TEX_BLEND_FRAC +: `VX_TEX_DIM_BITS]) << log_pitch_s0[i][k];
            assign offset_v_hi[i][k] = OFFSET_V_W'(scaled_hi[i][k][1][`TEX_BLEND_FRAC +: `VX_TEX_DIM_BITS]) << log_pitch_s0[i][k];
        end
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_addr
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            assign addr[i][k][0] = 32'(offset_v_lo[i][k]) + 32'(offset_u_lo[i][k]);
            assign addr[i][k][1] = 32'(offset_v_lo[i][k]) + 32'(offset_u_hi[i][k]);
            assign addr[i][k][2] = 32'(offset_v_hi[i][k]) + 32'(offset_u_lo[i][k]);
            assign addr[i][k][3] = 32'(offset_v_hi[i][k]) + 32'(offset_u_hi[i][k]);
        end
    end

    assign stall_out = rsp_valid && ~rsp_ready;

    VX_pipe_register #(
        .DATAW  (1 + NUM_LANES + TEX_FILTER_BITS + `TEX_LGSTRIDE_BITS + REQ_TAGW + NUM_LANES * (`VX_TEX_LOD_FRAC_BITS + TEX_NUM_LEVELS * (W_ADDR_BITS + 4 * 32 + 2 * `TEX_BLEND_FRAC))),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s0,  mask_s0,  filter_s0,  log_stride_s0, mip_addr_s0,  addr,     blends,     lodfrac_s0,  req_tag_s0}),
        .data_out ({rsp_valid, rsp_mask, rsp_filter, rsp_lgstride,  rsp_baseaddr, rsp_addr, rsp_blends, rsp_lodfrac, rsp_tag})
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
            `TRACE(2, ("%d: *** %s-addr: log_pitch=", $time, INSTANCE_ID))
            `TRACE_ARRAY2D(2, "0x%0h", log_pitch, TEX_NUM_LEVELS, NUM_LANES)
            `TRACE(2, (", mip_addr="))
            `TRACE_ARRAY2D(2, "0x%0h", mip_addr, TEX_NUM_LEVELS, NUM_LANES)
            `TRACE(2, (", req_logdims="))
            `TRACE_ARRAY1D(2, "0x%0h", req_logdims, 2)
            `TRACE(2, (", lodfrac="))
            `TRACE_ARRAY1D(2, "0x%0h", req_lodfrac, NUM_LANES)
            `TRACE(2, ("\n"))
        end

        if (valid_s0 && ~stall_out) begin
            `TRACE(2, ("%d: *** %s-addr: offset_u_lo=", $time, INSTANCE_ID))
            `TRACE_ARRAY2D(2, "0x%0h", offset_u_lo, TEX_NUM_LEVELS, NUM_LANES)
            `TRACE(2, (", offset_u_hi="))
            `TRACE_ARRAY2D(2, "0x%0h", offset_u_hi, TEX_NUM_LEVELS, NUM_LANES)
            `TRACE(2, (", offset_v_lo="))
            `TRACE_ARRAY2D(2, "0x%0h", offset_v_lo, TEX_NUM_LEVELS, NUM_LANES)
            `TRACE(2, (", offset_v_hi="))
            `TRACE_ARRAY2D(2, "0x%0h", offset_v_hi, TEX_NUM_LEVELS, NUM_LANES)
            `TRACE(2, ("\n"))
        end

        if (rsp_valid && rsp_ready) begin
            `TRACE(2, ("%d: %s-addr: valid=%b, req_filter=%0d, lgstride=%0d, addr=", $time, INSTANCE_ID, rsp_mask, rsp_filter, rsp_lgstride))
            `TRACE_ARRAY2D(2, "0x%0h", trace_addr[0], 4, NUM_LANES)
            `TRACE(2, (", addr_up="))
            `TRACE_ARRAY2D(2, "0x%0h", trace_addr[1], 4, NUM_LANES)
            `TRACE(2, (" (#%0d)\n", rsp_tag[REQ_TAGW-1 -: UUID_WIDTH]))
        end
    end
`endif

endmodule
