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

module VX_tex_addr #(
    parameter `STRING INSTANCE_ID = "",
    parameter REQ_INFOW = 1,
    parameter NUM_LANES = 1,
    parameter W_ADDR_BITS = `TEX_ADDR_BITS + 6
) (
    input wire clk,
    input wire reset,

    // inputs

    input wire                          req_valid,    
    input wire [NUM_LANES-1:0]          req_mask,
    input wire [1:0][NUM_LANES-1:0][`VX_TEX_FXD_BITS-1:0] req_coords,    
    input wire [`TEX_FORMAT_BITS-1:0]   req_format,
    input wire [`TEX_FILTER_BITS-1:0]   req_filter,
    input wire [1:0][`TEX_WRAP_BITS-1:0] req_wraps,
    input wire [`TEX_ADDR_BITS-1:0]     req_baseaddr,
    input wire [NUM_LANES-1:0][`VX_TEX_LOD_BITS-1:0] req_miplevel,
    input wire [NUM_LANES-1:0][`TEX_MIPOFF_BITS-1:0] req_mipoff,    
    input wire [1:0][`VX_TEX_LOD_BITS-1:0] req_logdims,
    input wire [REQ_INFOW-1:0]          req_info,    
    output wire                         req_ready,

    // outputs

    output wire                         rsp_valid, 
    output wire [NUM_LANES-1:0]         rsp_mask,
    output wire [`TEX_FILTER_BITS-1:0]  rsp_filter,
    output wire [`TEX_LGSTRIDE_BITS-1:0] rsp_lgstride,
    output wire [NUM_LANES-1:0][W_ADDR_BITS-1:0] rsp_baseaddr,
    output wire [NUM_LANES-1:0][3:0][31:0] rsp_addr,
    output wire [NUM_LANES-1:0][1:0][`TEX_BLEND_FRAC-1:0] rsp_blends,
    output wire [REQ_INFOW-1:0]         rsp_info,  
    input wire                          rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam SHIFT_BITS = `CLOG2(`VX_TEX_FXD_FRAC+1);
    localparam PITCH_BITS = `MAX(`VX_TEX_LOD_BITS, `TEX_LGSTRIDE_BITS) + 1;
    localparam SCALED_DIM = `VX_TEX_FXD_FRAC + `VX_TEX_DIM_BITS;
    localparam SCALED_X_W = `VX_TEX_DIM_BITS + `TEX_BLEND_FRAC;
    localparam OFFSET_U_W = `VX_TEX_DIM_BITS + `TEX_LGSTRIDE_MAX;
    localparam OFFSET_V_W = `VX_TEX_DIM_BITS + `VX_TEX_DIM_BITS + `TEX_LGSTRIDE_MAX;

    wire                valid_s0;   
    wire [NUM_LANES-1:0] mask_s0; 
    wire [`TEX_FILTER_BITS-1:0] filter_s0;
    wire [REQ_INFOW-1:0] req_info_s0;
    wire [NUM_LANES-1:0][1:0][`VX_TEX_FXD_FRAC-1:0] clamped_lo, clamped_lo_s0;
    wire [NUM_LANES-1:0][1:0][`VX_TEX_FXD_FRAC-1:0] clamped_hi, clamped_hi_s0;
    wire [NUM_LANES-1:0][1:0][SHIFT_BITS-1:0] dim_shift, dim_shift_s0;
    wire [`TEX_LGSTRIDE_BITS-1:0] log_stride, log_stride_s0;
    wire [NUM_LANES-1:0][W_ADDR_BITS-1:0] mip_addr, mip_addr_s0;
    wire [NUM_LANES-1:0][PITCH_BITS-1:0] log_pitch, log_pitch_s0;
    
    wire stall_out;

    // stride   

    VX_tex_stride tex_stride (
        .format     (req_format),
        .log_stride (log_stride)
    );

    // addressing mode

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            wire [`VX_TEX_FXD_FRAC-1:0] delta = `VX_TEX_FXD_FRAC'((SCALED_DIM'(`TEX_FXD_HALF) << req_miplevel[i]) >> req_logdims[j]);
            wire [`VX_TEX_FXD_BITS-1:0] coord_lo = req_filter ? (req_coords[j][i] - `VX_TEX_FXD_BITS'(delta)) : req_coords[j][i];
            wire [`VX_TEX_FXD_BITS-1:0] coord_hi = req_filter ? (req_coords[j][i] + `VX_TEX_FXD_BITS'(delta)) : req_coords[j][i];

            VX_tex_wrap tex_wrap_lo (
                .wrap_i  (req_wraps[j]),
                .coord_i (coord_lo),
                .coord_o (clamped_lo[i][j])
            );

            VX_tex_wrap tex_wrap_hi (
                .wrap_i  (req_wraps[j]),
                .coord_i (coord_hi),
                .coord_o (clamped_hi[i][j])
            );

            assign dim_shift[i][j] = SHIFT_BITS'(`VX_TEX_FXD_FRAC - `TEX_BLEND_FRAC) - (req_logdims[j] - req_miplevel[i]);
        end
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign log_pitch[i] = PITCH_BITS'(req_logdims[0] - req_miplevel[i]) + PITCH_BITS'(log_stride);        
        assign mip_addr[i]  = {req_baseaddr, 6'b0} + W_ADDR_BITS'(req_mipoff[i]);
    end

    VX_pipe_register #(
        .DATAW  (1 + NUM_LANES + `TEX_FILTER_BITS + `TEX_LGSTRIDE_BITS + REQ_INFOW + NUM_LANES * (PITCH_BITS + 2 * SHIFT_BITS + W_ADDR_BITS + 2 * 2 * `VX_TEX_FXD_FRAC)),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_mask, req_filter, log_stride,    req_info,    log_pitch,    dim_shift,    mip_addr,    clamped_lo,    clamped_hi}),
        .data_out ({valid_s0,  mask_s0,  filter_s0,  log_stride_s0, req_info_s0, log_pitch_s0, dim_shift_s0, mip_addr_s0, clamped_lo_s0, clamped_hi_s0})
    );
    
    // addresses generation

    wire [NUM_LANES-1:0][1:0][SCALED_X_W-1:0] scaled_lo;
    wire [NUM_LANES-1:0][1:0][SCALED_X_W-1:0] scaled_hi;
    wire [NUM_LANES-1:0][OFFSET_U_W-1:0] offset_u_lo;
    wire [NUM_LANES-1:0][OFFSET_U_W-1:0] offset_u_hi;
    wire [NUM_LANES-1:0][OFFSET_V_W-1:0] offset_v_lo;
    wire [NUM_LANES-1:0][OFFSET_V_W-1:0] offset_v_hi;
    wire [NUM_LANES-1:0][1:0][`TEX_BLEND_FRAC-1:0] blends;
    wire [NUM_LANES-1:0][3:0][31:0] addr;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin  
            assign scaled_lo[i][j] = SCALED_X_W'(clamped_lo_s0[i][j] >> dim_shift_s0[i][j]);
            assign scaled_hi[i][j] = SCALED_X_W'(clamped_hi_s0[i][j] >> dim_shift_s0[i][j]);          
            assign blends[i][j] = filter_s0 ? scaled_lo[i][j][`TEX_BLEND_FRAC-1:0] : `TEX_BLEND_FRAC'(0);
        end
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign offset_u_lo[i] = OFFSET_U_W'(scaled_lo[i][0][`TEX_BLEND_FRAC +: `VX_TEX_DIM_BITS]) << log_stride_s0;
        assign offset_u_hi[i] = OFFSET_U_W'(scaled_hi[i][0][`TEX_BLEND_FRAC +: `VX_TEX_DIM_BITS]) << log_stride_s0;        
        assign offset_v_lo[i] = OFFSET_V_W'(scaled_lo[i][1][`TEX_BLEND_FRAC +: `VX_TEX_DIM_BITS]) << log_pitch_s0[i];
        assign offset_v_hi[i] = OFFSET_V_W'(scaled_hi[i][1][`TEX_BLEND_FRAC +: `VX_TEX_DIM_BITS]) << log_pitch_s0[i];
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign addr[i][0] = 32'(offset_v_lo[i]) + 32'(offset_u_lo[i]);
        assign addr[i][1] = 32'(offset_v_lo[i]) + 32'(offset_u_hi[i]);
        assign addr[i][2] = 32'(offset_v_hi[i]) + 32'(offset_u_lo[i]);
        assign addr[i][3] = 32'(offset_v_hi[i]) + 32'(offset_u_hi[i]);
    end

    assign stall_out = rsp_valid && ~rsp_ready;

    VX_pipe_register #(
        .DATAW  (1 + NUM_LANES + `TEX_FILTER_BITS + `TEX_LGSTRIDE_BITS + (NUM_LANES * W_ADDR_BITS) + (NUM_LANES * 4 * 32) + (2 * NUM_LANES * `TEX_BLEND_FRAC) + REQ_INFOW),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s0,  mask_s0,  filter_s0,  log_stride_s0, mip_addr_s0,  addr,     blends,     req_info_s0}),
        .data_out ({rsp_valid, rsp_mask, rsp_filter, rsp_lgstride,  rsp_baseaddr, rsp_addr, rsp_blends, rsp_info})
    );

    assign req_ready = ~stall_out;

`ifdef DBG_TRACE_TEX    
    always @(posedge clk) begin
        if (req_valid && ~stall_out) begin
            `TRACE(2, ("%d: *** %s-addr: log_pitch=", $time, INSTANCE_ID)); 
            `TRACE_ARRAY1D(2, log_pitch, NUM_LANES);
            `TRACE(2, (", mip_addr=0x"));
            `TRACE_ARRAY1D(2, mip_addr, NUM_LANES);
            `TRACE(2, (", req_logdims="));
            `TRACE_ARRAY1D(2, req_logdims, 2);  
            `TRACE(2, (", clamped_lo="));
            `TRACE_ARRAY2D(2, clamped_lo, 2, NUM_LANES);    
            `TRACE(2, (", clamped_hi="));
            `TRACE_ARRAY2D(2, clamped_hi, 2, NUM_LANES);
            `TRACE(2, (", mip_addr=0x"));
            `TRACE_ARRAY1D(2, mip_addr, NUM_LANES);
            `TRACE(2, ("\n"));
        end

        if (valid_s0 && ~stall_out) begin
            `TRACE(2, ("%d: *** %s-addr: scaled_lo=", $time, INSTANCE_ID)); 
            `TRACE_ARRAY2D(2, scaled_lo, 2, NUM_LANES);
            `TRACE(2, (", scaled_hi="));
            `TRACE_ARRAY2D(2, scaled_hi, 2, NUM_LANES);  
            `TRACE(2, (", offset_u_lo="));
            `TRACE_ARRAY1D(2, offset_u_lo, NUM_LANES);
            `TRACE(2, (", offset_u_hi="));
            `TRACE_ARRAY1D(2, offset_u_hi, NUM_LANES);    
            `TRACE(2, (", offset_v_lo="));
            `TRACE_ARRAY1D(2, offset_v_lo, NUM_LANES);
            `TRACE(2, (", offset_v_hi="));
            `TRACE_ARRAY1D(2, offset_v_hi, NUM_LANES);
            `TRACE(2, ("\n"));
        end

        if (rsp_valid && rsp_ready) begin
            `TRACE(2, ("%d: %s-addr: valid=%b, req_filter=%0d, lgstride=%0d, addr=", $time, INSTANCE_ID, rsp_mask, rsp_filter, rsp_lgstride));
            `TRACE_ARRAY2D(2, rsp_addr, 4, NUM_LANES);
            `TRACE(2, (" (#%0d)\n", rsp_info[REQ_INFOW-1 -: `UUID_WIDTH]));
        end
    end
`endif

endmodule
