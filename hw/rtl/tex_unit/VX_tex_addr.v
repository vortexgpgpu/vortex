`include "VX_tex_define.vh"

module VX_tex_addr #(
    parameter CORE_ID        = 0,
    parameter REQ_INFO_WIDTH = 1,
    parameter NUM_REQS       = 1
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
    input wire [REQ_INFO_WIDTH-1:0]     req_info,    
    output wire                         req_ready,

    // outputs

    output wire                         rsp_valid, 
    output wire [NUM_REQS-1:0]          rsp_tmask,
    output wire [`TEX_FILTER_BITS-1:0]  rsp_filter,
    output wire [`TEX_STRIDE_BITS-1:0]  rsp_stride,
    output wire [NUM_REQS-1:0][3:0][31:0] rsp_addr,
    output wire [NUM_REQS-1:0][1:0][`BLEND_FRAC-1:0] rsp_blends,
    output wire [REQ_INFO_WIDTH-1:0]    rsp_info,  
    input wire                          rsp_ready
);

    `UNUSED_PARAM (CORE_ID)

    wire                        valid_s0;   
    wire [NUM_REQS-1:0]         tmask_s0; 
    wire [`TEX_FILTER_BITS-1:0] filter_s0;
    wire [REQ_INFO_WIDTH-1:0]   req_info_s0;
    wire [NUM_REQS-1:0][1:0][31:0] coord_lo, coord_lo_s0;
    wire [NUM_REQS-1:0][1:0][31:0] coord_hi, coord_hi_s0;
    wire [`TEX_STRIDE_BITS-1:0] log_stride, log_stride_s0;
    wire [NUM_REQS-1:0][31:0] mip_addr, mip_addr_s0;
    wire [NUM_REQS-1:0][1:0][`TEX_DIM_BITS-1:0] log_dims_s0;
    wire [1:0][`TEX_WRAP_BITS-1:0] req_wraps_s0;

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
            assign coord_lo[i][j] = req_filter ? (req_coords[j][i] - (`FIXED_HALF >> req_logdims[i][j])) : req_coords[j][i];
            assign coord_hi[i][j] = req_filter ? (req_coords[j][i] + (`FIXED_HALF >> req_logdims[i][j])) : req_coords[j][i];
        end
        assign mip_addr[i] = req_baseaddr + 32'(req_mipoff[i]);
    end

    VX_pipe_register #(
        .DATAW  (1 + NUM_REQS + `TEX_FILTER_BITS + `TEX_STRIDE_BITS + REQ_INFO_WIDTH + 2 * `TEX_WRAP_BITS + NUM_REQS * (2 * `TEX_DIM_BITS + 32 + 2 * 2 * 32)),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_tmask, req_filter, log_stride,    req_info,    req_wraps,    req_logdims, mip_addr,    coord_lo,    coord_hi}),
        .data_out ({valid_s0,  tmask_s0,  filter_s0,  log_stride_s0, req_info_s0, req_wraps_s0, log_dims_s0, mip_addr_s0, coord_lo_s0, coord_hi_s0})
    );
    
    // addresses generation

    wire [NUM_REQS-1:0][1:0][`FIXED_INT-1:0] scaled_lo;
    wire [NUM_REQS-1:0][1:0][`FIXED_INT-1:0] scaled_hi;
    wire [NUM_REQS-1:0][1:0][`BLEND_FRAC-1:0] blends;
    wire [NUM_REQS-1:0][3:0][31:0] addr;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            wire [`FIXED_FRAC-1:0] clamped_lo;
            wire [`FIXED_FRAC-1:0] clamped_hi;

            VX_tex_wrap #(
                .CORE_ID (CORE_ID)
            ) tex_wrap_lo (
                .wrap_i  (req_wraps_s0[j]),
                .coord_i (coord_lo_s0[i][j]),
                .coord_o (clamped_lo)
            );

            VX_tex_wrap #(
                .CORE_ID (CORE_ID)
            ) tex_wrap_hi (
                .wrap_i  (req_wraps_s0[j]),
                .coord_i (coord_hi_s0[i][j]),
                .coord_o (clamped_hi)
            );

            assign scaled_lo[i][j] = `FIXED_INT'(clamped_lo >> ((`FIXED_FRAC) - log_dims_s0[i][j])); 
            assign scaled_hi[i][j] = `FIXED_INT'(clamped_hi >> ((`FIXED_FRAC) - log_dims_s0[i][j]));
            assign blends[i][j]    = filter_s0 ? clamped_lo[`BLEND_FRAC-1:0] : `BLEND_FRAC'(0);
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign addr[i][0] = mip_addr_s0[i] + (32'(scaled_lo[i][0]) + (32'(scaled_lo[i][1]) << log_dims_s0[i][0])) << log_stride_s0;
        assign addr[i][1] = mip_addr_s0[i] + (32'(scaled_hi[i][0]) + (32'(scaled_lo[i][1]) << log_dims_s0[i][0])) << log_stride_s0;
        assign addr[i][2] = mip_addr_s0[i] + (32'(scaled_lo[i][0]) + (32'(scaled_hi[i][1]) << log_dims_s0[i][0])) << log_stride_s0;
        assign addr[i][3] = mip_addr_s0[i] + (32'(scaled_hi[i][0]) + (32'(scaled_hi[i][1]) << log_dims_s0[i][0])) << log_stride_s0;
    end

    assign stall_out = rsp_valid && ~rsp_ready;

    VX_pipe_register #(
        .DATAW  (1 + NUM_REQS + `TEX_FILTER_BITS + `TEX_STRIDE_BITS + (NUM_REQS * 4 * 32) + (2 * NUM_REQS * `BLEND_FRAC) + REQ_INFO_WIDTH),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s0,  tmask_s0,  filter_s0,  log_stride_s0, addr,     blends,     req_info_s0}),
        .data_out ({rsp_valid, rsp_tmask, rsp_filter, rsp_stride,    rsp_addr, rsp_blends, rsp_info})
    );

    assign req_ready = ~stall_out;

 `ifdef DBG_PRINT_TEX   
    wire [`NW_BITS-1:0] rsp_wid;
    wire [31:0]         rsp_PC;
    assign {rsp_wid, rsp_PC} = rsp_info[`NW_BITS+32-1:0];
    
    always @(posedge clk) begin
        if (rsp_valid && rsp_ready) begin
            $write("%t: core%0d-tex-addr: wid=%0d, PC=%0h, tmask=%b, req_filter=%0d, tride=%0d, addr=", 
                    $time, CORE_ID, rsp_wid, rsp_PC, rsp_tmask, rsp_filter, rsp_stride);
            `PRINT_ARRAY2D(rsp_addr, 4, NUM_REQS);
            $write("\n");
        end
    end
`endif

endmodule