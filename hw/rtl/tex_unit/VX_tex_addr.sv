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
    input wire [1:0][NUM_REQS-1:0][`TEX_FXD_BITS-1:0] req_coords,    
    input wire [`TEX_FORMAT_BITS-1:0]   req_format,
    input wire [`TEX_FILTER_BITS-1:0]   req_filter,
    input wire [1:0][`TEX_WRAP_BITS-1:0] req_wraps,
    input wire [`TEX_ADDR_BITS-1:0]     req_baseaddr,
    input wire [NUM_REQS-1:0][`TEX_LOD_BITS-1:0] mip_level,
    input wire [NUM_REQS-1:0][`TEX_MIPOFF_BITS-1:0] req_mipoff,    
    input wire [NUM_REQS-1:0][1:0][`TEX_LOD_BITS-1:0] req_logdims,
    input wire [REQ_INFOW-1:0]          req_info,    
    output wire                         req_ready,

    // outputs

    output wire                         rsp_valid, 
    output wire [NUM_REQS-1:0]          rsp_tmask,
    output wire [`TEX_FILTER_BITS-1:0]  rsp_filter,
    output wire [`TEX_LGSTRIDE_BITS-1:0] rsp_lgstride,
    output wire [NUM_REQS-1:0][31:0]    rsp_baseaddr,
    output wire [NUM_REQS-1:0][3:0][31:0] rsp_addr,
    output wire [NUM_REQS-1:0][1:0][`TEX_BLEND_FRAC-1:0] rsp_blends,
    output wire [REQ_INFOW-1:0]         rsp_info,  
    input wire                          rsp_ready
);

    `UNUSED_PARAM (CORE_ID)

    localparam SHIFT_BITS = $clog2(`TEX_FXD_FRAC+1);
    localparam PITCH_BITS = `MAX(`TEX_LOD_BITS, `TEX_LGSTRIDE_BITS) + 1;
    localparam SCALED_DIM = `TEX_FXD_FRAC + `TEX_DIM_BITS;
    localparam SCALED_X_W = `TEX_DIM_BITS + `TEX_BLEND_FRAC;
    localparam OFFSET_U_W = `TEX_DIM_BITS + `TEX_LGSTRIDE_MAX;
    localparam OFFSET_V_W = `TEX_DIM_BITS + `TEX_DIM_BITS + `TEX_LGSTRIDE_MAX;

    wire                valid_s0;   
    wire [NUM_REQS-1:0] tmask_s0; 
    wire [`TEX_FILTER_BITS-1:0] filter_s0;
    wire [REQ_INFOW-1:0] req_info_s0;
    wire [NUM_REQS-1:0][1:0][`TEX_FXD_FRAC-1:0] clamped_lo, clamped_lo_s0;
    wire [NUM_REQS-1:0][1:0][`TEX_FXD_FRAC-1:0] clamped_hi, clamped_hi_s0;
    wire [NUM_REQS-1:0][1:0][SHIFT_BITS-1:0] dim_shift, dim_shift_s0;
    wire [`TEX_LGSTRIDE_BITS-1:0] log_stride, log_stride_s0;
    wire [NUM_REQS-1:0][31:0] mip_addr, mip_addr_s0;
    wire [NUM_REQS-1:0][PITCH_BITS-1:0] log_pitch, log_pitch_s0;
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
            wire [`TEX_FXD_FRAC-1:0] delta    = `TEX_FXD_FRAC'((SCALED_DIM'(`TEX_FXD_HALF) << mip_level[i]) >> req_logdims[i][j]);
            wire [`TEX_FXD_BITS-1:0] coord_lo = req_filter ? (req_coords[j][i] - `TEX_FXD_BITS'(delta)) : req_coords[j][i];
            wire [`TEX_FXD_BITS-1:0] coord_hi = req_filter ? (req_coords[j][i] + `TEX_FXD_BITS'(delta)) : req_coords[j][i];

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

            assign dim_shift[i][j] = (`TEX_FXD_FRAC - `TEX_BLEND_FRAC - (req_logdims[i][j] - mip_level[i]));
        end
        assign log_pitch[i] = PITCH_BITS'(req_logdims[i][0] - mip_level[i]) + PITCH_BITS'(log_stride);        
        assign mip_addr[i]  = req_baseaddr + `TEX_ADDR_BITS'(req_mipoff[i]);
    end

    VX_pipe_register #(
        .DATAW  (1 + NUM_REQS + `TEX_FILTER_BITS + `TEX_LGSTRIDE_BITS + REQ_INFOW + NUM_REQS * (PITCH_BITS + 2 * SHIFT_BITS + `TEX_ADDR_BITS + 2 * 2 * `TEX_FXD_FRAC)),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_tmask, req_filter, log_stride,    req_info,    log_pitch,    dim_shift,    mip_addr,    clamped_lo,    clamped_hi}),
        .data_out ({valid_s0,  tmask_s0,  filter_s0,  log_stride_s0, req_info_s0, log_pitch_s0, dim_shift_s0, mip_addr_s0, clamped_lo_s0, clamped_hi_s0})
    );
    
    // addresses generation

    wire [NUM_REQS-1:0][1:0][SCALED_X_W-1:0] scaled_lo;
    wire [NUM_REQS-1:0][1:0][SCALED_X_W-1:0] scaled_hi;
    wire [NUM_REQS-1:0][OFFSET_U_W-1:0] offset_u_lo;
    wire [NUM_REQS-1:0][OFFSET_U_W-1:0] offset_u_hi;
    wire [NUM_REQS-1:0][OFFSET_V_W-1:0] offset_v_lo;
    wire [NUM_REQS-1:0][OFFSET_V_W-1:0] offset_v_hi;
    wire [NUM_REQS-1:0][1:0][`TEX_BLEND_FRAC-1:0] blends;
    wire [NUM_REQS-1:0][3:0][31:0] addr;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin  
            assign scaled_lo[i][j] = SCALED_X_W'(clamped_lo_s0[i][j] >> dim_shift_s0[i][j]);
            assign scaled_hi[i][j] = SCALED_X_W'(clamped_hi_s0[i][j] >> dim_shift_s0[i][j]);          
            assign blends[i][j] = filter_s0 ? scaled_lo[i][j][`TEX_BLEND_FRAC-1:0] : `TEX_BLEND_FRAC'(0);
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign offset_u_lo[i] = OFFSET_U_W'(scaled_lo[i][0][`TEX_BLEND_FRAC +: `TEX_DIM_BITS]) << log_stride_s0;
        assign offset_u_hi[i] = OFFSET_U_W'(scaled_hi[i][0][`TEX_BLEND_FRAC +: `TEX_DIM_BITS]) << log_stride_s0;
        
        assign offset_v_lo[i] = OFFSET_V_W'(scaled_lo[i][1][`TEX_BLEND_FRAC +: `TEX_DIM_BITS]) << log_pitch_s0[i];
        assign offset_v_hi[i] = OFFSET_V_W'(scaled_hi[i][1][`TEX_BLEND_FRAC +: `TEX_DIM_BITS]) << log_pitch_s0[i];

        assign addr[i][0] = 32'(offset_v_lo[i]) + 32'(offset_u_lo[i]);
        assign addr[i][1] = 32'(offset_v_lo[i]) + 32'(offset_u_hi[i]);
        assign addr[i][2] = 32'(offset_v_hi[i]) + 32'(offset_u_lo[i]);
        assign addr[i][3] = 32'(offset_v_hi[i]) + 32'(offset_u_hi[i]);
    end

    assign stall_out = rsp_valid && ~rsp_ready;

    VX_pipe_register #(
        .DATAW  (1 + NUM_REQS + `TEX_FILTER_BITS + `TEX_LGSTRIDE_BITS + (NUM_REQS * 32) + (NUM_REQS * 4 * 32) + (2 * NUM_REQS * `TEX_BLEND_FRAC) + REQ_INFOW),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s0,  tmask_s0,  filter_s0,  log_stride_s0, mip_addr_s0,  addr,     blends,     req_info_s0}),
        .data_out ({rsp_valid, rsp_tmask, rsp_filter, rsp_lgstride,  rsp_baseaddr, rsp_addr, rsp_blends, rsp_info})
    );

    assign req_ready = ~stall_out;

`ifdef DBG_TRACE_TEX
    wire [`NW_BITS-1:0] rsp_wid;
    wire [31:0]         rsp_PC;

    assign {rsp_wid, rsp_PC} = rsp_info[`NW_BITS+32-1:0];
    
    always @(posedge clk) begin
        if (req_valid && ~stall_out) begin
            dpi_trace("%d: *** log_pitch=", $time); 
            `TRACE_ARRAY1D(log_pitch, NUM_REQS);
            dpi_trace(", mip_addr=");
            `TRACE_ARRAY1D(mip_addr, NUM_REQS);
            dpi_trace(", req_logdims=");
            `TRACE_ARRAY2D(req_logdims, 2, NUM_REQS);  
            dpi_trace(", clamped_lo=");
            `TRACE_ARRAY2D(clamped_lo, 2, NUM_REQS);    
            dpi_trace(", clamped_hi=");
            `TRACE_ARRAY2D(clamped_hi, 2, NUM_REQS);
            dpi_trace(", mip_addr=");
            `TRACE_ARRAY1D(mip_addr, NUM_REQS);
            dpi_trace("\n");
        end

        if (valid_s0 && ~stall_out) begin
            dpi_trace("%d: *** scaled_lo=", $time); 
            `TRACE_ARRAY2D(scaled_lo, 2, NUM_REQS);
            dpi_trace(", scaled_hi=");
            `TRACE_ARRAY2D(scaled_hi, 2, NUM_REQS);  
            dpi_trace(", offset_u_lo=");
            `TRACE_ARRAY1D(offset_u_lo, NUM_REQS);
            dpi_trace(", offset_u_hi=");
            `TRACE_ARRAY1D(offset_u_hi, NUM_REQS);    
            dpi_trace(", offset_v_lo=");
            `TRACE_ARRAY1D(offset_v_lo, NUM_REQS);
            dpi_trace(", offset_v_hi=");
            `TRACE_ARRAY1D(offset_v_hi, NUM_REQS);
            dpi_trace("\n");
        end

        if (rsp_valid && rsp_ready) begin
            dpi_trace("%d: core%0d-tex-addr: wid=%0d, PC=%0h, tmask=%b, req_filter=%0d, lgstride=%0d, addr=", 
                    $time, CORE_ID, rsp_wid, rsp_PC, rsp_tmask, rsp_filter, rsp_lgstride);
            `TRACE_ARRAY2D(rsp_addr, 4, NUM_REQS);
            dpi_trace("\n");
        end
    end
`endif

endmodule