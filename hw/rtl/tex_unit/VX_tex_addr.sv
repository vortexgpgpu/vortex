`include "VX_tex_define.vh"

module VX_tex_addr #(
    parameter CORE_ID   = 0,
    parameter REQ_INFOW = 1,
    parameter NUM_REQS  = 1
) (
    input wire clk,
    input wire reset,

    // inputs

    input wire                          req_valid,    
    input wire [NUM_REQS-1:0]           req_tmask,
    input wire [1:0][NUM_REQS-1:0][31:0] req_coords,    
    input wire [`TEX_FORMAT_BITS-1:0]   req_format,
    input wire [`TEX_FILTER_BITS-1:0]   req_filter,
    input wire [1:0][`TEX_WRAP_BITS-1:0] req_wraps,
    input wire [`TEX_ADDR_BITS-1:0]     req_baseaddr,
    input wire [NUM_REQS-1:0][`TEX_MIPOFF_BITS-1:0] req_mipoff,    
    input wire [NUM_REQS-1:0][1:0][`TEX_DIM_BITS-1:0] req_logdims,
    input wire [REQ_INFOW-1:0]          req_info,    
    output wire                         req_ready,

    // outputs

    output wire                         rsp_valid, 
    output wire [NUM_REQS-1:0]          rsp_tmask,
    output wire [`TEX_FILTER_BITS-1:0]  rsp_filter,
    output wire [`TEX_STRIDE_BITS-1:0]  rsp_stride,
    output wire [NUM_REQS-1:0][3:0][31:0] rsp_addr,
    output wire [NUM_REQS-1:0][1:0][`BLEND_FRAC-1:0] rsp_blends,
    output wire [REQ_INFOW-1:0]         rsp_info,  
    input wire                          rsp_ready
);

    `UNUSED_PARAM (CORE_ID)

    localparam PITCH_BITS = `MAX(`TEX_DIM_BITS, `TEX_STRIDE_BITS) + 1;
    localparam SCALED_U_W = `FIXED_INT + `TEX_STRIDE_BITS;
    localparam SCALED_X_W = (2 * `FIXED_INT);
    localparam SCALED_V_W = SCALED_X_W + `TEX_STRIDE_BITS;

    wire                valid_s0;   
    wire [NUM_REQS-1:0] tmask_s0; 
    wire [`TEX_FILTER_BITS-1:0] filter_s0;
    wire [REQ_INFOW-1:0] req_info_s0;
    wire [NUM_REQS-1:0][1:0][`FIXED_FRAC-1:0] clamped_lo, clamped_lo_s0;
    wire [NUM_REQS-1:0][1:0][`FIXED_FRAC-1:0] clamped_hi, clamped_hi_s0;
    wire [`TEX_STRIDE_BITS-1:0] log_stride, log_stride_s0;
    wire [NUM_REQS-1:0][31:0] mip_addr, mip_addr_s0;
    wire [NUM_REQS-1:0][1:0][`TEX_DIM_BITS-1:0] log_dims_s0;
    wire [NUM_REQS-1:0][PITCH_BITS-1:0] log_pitch, log_pitch_s0;

    wire stall_out;

    // stride   

    VX_tex_stride #(
        .CORE_ID (CORE_ID)
    ) tex_stride (
        .format     (req_format),
        .log_stride (log_stride)
    );

    // addressing mode

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            wire [`FIXED_FRAC-1:0] delta = (`FIXED_HALF >> req_logdims[i][j]);
            wire [31:0] coord_lo = req_filter ? (req_coords[j][i] - 32'(delta)) : req_coords[j][i];
            wire [31:0] coord_hi = req_filter ? (req_coords[j][i] + 32'(delta)) : req_coords[j][i];

            VX_tex_wrap #(
                .CORE_ID (CORE_ID)
            ) tex_wrap_lo (
                .wrap_i  (req_wraps[j]),
                .coord_i (coord_lo),
                .coord_o (clamped_lo[i][j])
            );

            VX_tex_wrap #(
                .CORE_ID (CORE_ID)
            ) tex_wrap_hi (
                .wrap_i  (req_wraps[j]),
                .coord_i (coord_hi),
                .coord_o (clamped_hi[i][j])
            );
        end
        assign log_pitch[i] = PITCH_BITS'(req_logdims[i][0]) + PITCH_BITS'(log_stride);        
        assign mip_addr[i]  = req_baseaddr + 32'(req_mipoff[i]);
    end

    VX_pipe_register #(
        .DATAW  (1 + NUM_REQS + `TEX_FILTER_BITS + `TEX_STRIDE_BITS + REQ_INFOW + NUM_REQS * (PITCH_BITS + 2 * `TEX_DIM_BITS + 32 + 2 * 2 * `FIXED_FRAC)),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_tmask, req_filter, log_stride,    req_info,    log_pitch,    req_logdims, mip_addr,    clamped_lo,    clamped_hi}),
        .data_out ({valid_s0,  tmask_s0,  filter_s0,  log_stride_s0, req_info_s0, log_pitch_s0, log_dims_s0, mip_addr_s0, clamped_lo_s0, clamped_hi_s0})
    );
    
    // addresses generation

    wire [NUM_REQS-1:0][1:0][`FIXED_INT-1:0] scaled_lo;
    wire [NUM_REQS-1:0][1:0][`FIXED_INT-1:0] scaled_hi;
    wire [NUM_REQS-1:0][1:0][`BLEND_FRAC-1:0] blends;
    wire [NUM_REQS-1:0][3:0][31:0] addr;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin  
            assign scaled_lo[i][j] = scale_to_dim(clamped_lo_s0[i][j], log_dims_s0[i][j]);
            assign scaled_hi[i][j] = scale_to_dim(clamped_hi_s0[i][j], log_dims_s0[i][j]);          
            assign blends[i][j] = filter_s0 ? clamped_lo_s0[i][j][`BLEND_FRAC-1:0] : `BLEND_FRAC'(0);
        end
    end

    `UNUSED_VAR (log_pitch_s0)

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        wire [SCALED_U_W-1:0] offset_u_lo = SCALED_U_W'(scaled_lo[i][0]) << log_stride_s0;
        wire [SCALED_U_W-1:0] offset_u_hi = SCALED_U_W'(scaled_hi[i][0]) << log_stride_s0;
        
        wire [SCALED_V_W-1:0] offset_v_lo = SCALED_V_W'(scaled_lo[i][1]) << log_pitch_s0[i];
        wire [SCALED_V_W-1:0] offset_v_hi = SCALED_V_W'(scaled_hi[i][1]) << log_pitch_s0[i];

        wire [31:0] base_addr_lo = mip_addr_s0[i] + 32'(offset_v_lo);
        wire [31:0] base_addr_hi = mip_addr_s0[i] + 32'(offset_v_hi);

        assign addr[i][0] = base_addr_lo + 32'(offset_u_lo);
        assign addr[i][1] = base_addr_lo + 32'(offset_u_hi);
        assign addr[i][2] = base_addr_hi + 32'(offset_u_lo);
        assign addr[i][3] = base_addr_hi + 32'(offset_u_hi);
    end

    assign stall_out = rsp_valid && ~rsp_ready;

    VX_pipe_register #(
        .DATAW  (1 + NUM_REQS + `TEX_FILTER_BITS + `TEX_STRIDE_BITS + (NUM_REQS * 4 * 32) + (2 * NUM_REQS * `BLEND_FRAC) + REQ_INFOW),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s0,  tmask_s0,  filter_s0,  log_stride_s0, addr,     blends,     req_info_s0}),
        .data_out ({rsp_valid, rsp_tmask, rsp_filter, rsp_stride,    rsp_addr, rsp_blends, rsp_info})
    );

    assign req_ready = ~stall_out;

`ifdef DBG_TRACE_TEX
    wire [`NW_BITS-1:0] rsp_wid;
    wire [31:0]         rsp_PC;

    assign {rsp_wid, rsp_PC} = rsp_info[`NW_BITS+32-1:0];
    
    always @(posedge clk) begin
        if (rsp_valid && rsp_ready) begin
            dpi_trace("%d: core%0d-tex-addr: wid=%0d, PC=%0h, tmask=%b, req_filter=%0d, tride=%0d, addr=", 
                    $time, CORE_ID, rsp_wid, rsp_PC, rsp_tmask, rsp_filter, rsp_stride);
            `TRACE_ARRAY2D(rsp_addr, 4, NUM_REQS);
            dpi_trace("\n");
        end
    end
`endif

function logic [`FIXED_INT-1:0] scale_to_dim (input logic [`FIXED_FRAC-1:0] src, 
                                              input logic [`TEX_DIM_BITS-1:0] dim);
`IGNORE_WARNINGS_BEGIN
    logic [`FIXED_BITS-1:0] out;
`IGNORE_WARNINGS_END
    out = `FIXED_BITS'(src) << dim;
    return out[`FIXED_FRAC +: `FIXED_INT];
endfunction

endmodule