`include "VX_tex_define.vh"

module VX_tex_addr #(
    parameter CORE_ID = 0,
    parameter REQ_INFO_WIDTH = 1
) (
    input wire  clk,
    input wire  reset,

    // handshake
    
    input wire  valid_in,
    output wire ready_in,

    // inputs

    input wire [`NW_BITS-1:0]           req_wid,
    input wire [`NUM_THREADS-1:0]       req_tmask,
    input wire [31:0]                   req_PC,
    input wire [REQ_INFO_WIDTH-1:0]     req_info,

    input wire [`TEX_FORMAT_BITS-1:0]   format,
    input wire [`TEX_FILTER_BITS-1:0]   filter,
    input wire [`TEX_WRAP_BITS-1:0]     wrap_u,
    input wire [`TEX_WRAP_BITS-1:0]     wrap_v,

    input wire [`TEX_ADDR_BITS-1:0]     base_addr,

    input wire [`NUM_THREADS-1:0][`TEX_MIPOFF_BITS-1:0] mip_offsets,    
    input wire [`NUM_THREADS-1:0][`TEX_DIM_BITS-1:0] log_widths,
    input wire [`NUM_THREADS-1:0][`TEX_DIM_BITS-1:0] log_heights,
    
    input wire [`NUM_THREADS-1:0][31:0] coord_u,
    input wire [`NUM_THREADS-1:0][31:0] coord_v,

    // outputs

    output wire                     rsp_valid,  
    output wire [`NW_BITS-1:0]      rsp_wid,
    output wire [`NUM_THREADS-1:0]  rsp_tmask,
    output wire [31:0]              rsp_PC,
    output wire [`TEX_FILTER_BITS-1:0] rsp_filter,
    output wire [`TEX_STRIDE_BITS-1:0] rsp_stride,
    output wire [`NUM_THREADS-1:0][3:0][31:0] rsp_addr,
    output wire [`NUM_THREADS-1:0][`BLEND_FRAC-1:0] rsp_blend_u,
    output wire [`NUM_THREADS-1:0][`BLEND_FRAC-1:0] rsp_blend_v,
    output wire [REQ_INFO_WIDTH-1:0] rsp_info,  
    input wire                      rsp_ready
);

    `UNUSED_PARAM (CORE_ID)

    wire [`NUM_THREADS-1:0][1:0][`FIXED_FRAC-1:0] clamped_u, clamped_v, clamped_u_s0, clamped_v_s0;
    wire [`TEX_STRIDE_BITS-1:0] log_stride, log_stride_s0;
    wire [`NUM_THREADS-1:0][31:0] mip_addr, mip_addr_s0;

    wire                        valid_in_s0;
    wire [`NW_BITS-1:0]         req_wid_s0;
    wire [`NUM_THREADS-1:0]     req_tmask_s0;
    wire [31:0]                 req_PC_s0;
    wire [REQ_INFO_WIDTH-1:0]   req_info_s0;
    wire [`TEX_FILTER_BITS-1:0] filter_s0;
    wire [`NUM_THREADS-1:0][`TEX_DIM_BITS-1:0] log_widths_s0;
    wire [`NUM_THREADS-1:0][`TEX_DIM_BITS-1:0] log_heights_s0;

    wire stall_out;

    // stride   

    VX_tex_stride #(
        .CORE_ID (CORE_ID)
    ) tex_stride (
        .format (format),
        .log_stride (log_stride)
    );

    // addressing mode

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [1:0][31:0] fu, fv;

        assign fu[0] = coord_u[i] - (filter ? (`FIXED_HALF >> log_widths[i]) : 0); 
        assign fu[1] = coord_u[i] + (filter ? (`FIXED_HALF >> log_widths[i]) : 0);

        assign fv[0] = coord_v[i] - (filter ? (`FIXED_HALF >> log_heights[i]) : 0);        
        assign fv[1] = coord_v[i] + (filter ? (`FIXED_HALF >> log_heights[i]) : 0);        

        VX_tex_wrap #(
            .CORE_ID (CORE_ID)
        ) tex_wrap_u0 (
            .wrap_i  (wrap_u),
            .coord_i (fu[0]),
            .coord_o (clamped_u[i][0])
        );

        VX_tex_wrap #(
            .CORE_ID (CORE_ID)
        ) tex_wrap_u1 (
            .wrap_i  (wrap_u),
            .coord_i (fu[1]),
            .coord_o (clamped_u[i][1])
        );

        VX_tex_wrap #(
            .CORE_ID (CORE_ID)
        ) tex_wrap_v0 (
            .wrap_i  (wrap_v),
            .coord_i (fv[0]),
            .coord_o (clamped_v[i][0])
        );

        VX_tex_wrap #(
            .CORE_ID (CORE_ID)
        ) tex_wrap_v1 (
            .wrap_i  (wrap_v),
            .coord_i (fv[1]),
            .coord_o (clamped_v[i][1])
        );

        assign mip_addr[i] = base_addr + 32'(mip_offsets[i]);
    end

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `TEX_FILTER_BITS + `TEX_STRIDE_BITS + REQ_INFO_WIDTH + `NUM_THREADS * (2 * `TEX_DIM_BITS + 32 + 2 * 2 * `FIXED_FRAC)),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_in,    req_wid,    req_tmask,    req_PC,    filter,    log_stride,    req_info,    log_widths,    log_heights,    mip_addr,    clamped_u,    clamped_v}),
        .data_out ({valid_in_s0, req_wid_s0, req_tmask_s0, req_PC_s0, filter_s0, log_stride_s0, req_info_s0, log_widths_s0, log_heights_s0, mip_addr_s0, clamped_u_s0, clamped_v_s0})
    );
    
    // addresses generation

    wire [`NUM_THREADS-1:0][`BLEND_FRAC-1:0] blend_u, blend_v;
    wire [`NUM_THREADS-1:0][3:0][31:0] addr;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [1:0][`FIXED_INT-1:0] x, y;

        assign x[0] = `FIXED_INT'(clamped_u_s0[i][0] >> ((`FIXED_FRAC) - log_widths_s0[i])); 
        assign x[1] = `FIXED_INT'(clamped_u_s0[i][1] >> ((`FIXED_FRAC) - log_widths_s0[i])); 
        assign y[0] = `FIXED_INT'(clamped_v_s0[i][0] >> ((`FIXED_FRAC) - log_heights_s0[i]));         
        assign y[1] = `FIXED_INT'(clamped_v_s0[i][1] >> ((`FIXED_FRAC) - log_heights_s0[i])); 

        assign addr[i][0] = mip_addr_s0[i] + (32'(x[0]) + (32'(y[0]) << log_widths_s0[i])) << log_stride_s0;
        assign addr[i][1] = mip_addr_s0[i] + (32'(x[1]) + (32'(y[0]) << log_widths_s0[i])) << log_stride_s0;
        assign addr[i][2] = mip_addr_s0[i] + (32'(x[0]) + (32'(y[1]) << log_widths_s0[i])) << log_stride_s0;
        assign addr[i][3] = mip_addr_s0[i] + (32'(x[1]) + (32'(y[1]) << log_widths_s0[i])) << log_stride_s0;
    end

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign blend_u[i] = clamped_u_s0[i][0][`BLEND_FRAC-1:0];
        assign blend_v[i] = clamped_v_s0[i][0][`BLEND_FRAC-1:0];
    end

    assign stall_out = rsp_valid && ~rsp_ready;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `TEX_FILTER_BITS + `TEX_STRIDE_BITS + (`NUM_THREADS * 4 * 32) + (2*`NUM_THREADS * `BLEND_FRAC) + REQ_INFO_WIDTH),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_in_s0, req_wid_s0, req_tmask_s0, req_PC_s0, filter_s0,  log_stride_s0, addr,     blend_u,     blend_v,     req_info_s0}),
        .data_out ({rsp_valid,   rsp_wid,    rsp_tmask,    rsp_PC,    rsp_filter, rsp_stride,    rsp_addr, rsp_blend_u, rsp_blend_v, rsp_info})
    );

    assign ready_in = ~stall_out;

 `ifdef DBG_PRINT_TEX   
    always @(posedge clk) begin
        if (rsp_valid && rsp_ready) begin
            $write("%t: core%0d-tex-addr: wid=%0d, PC=%0h, tmask=%b, filter=%0d, tride=%0d, addr=", 
                    $time, CORE_ID, rsp_wid, rsp_PC, rsp_tmask, rsp_filter, rsp_stride);
            `PRINT_ARRAY2D(rsp_addr, 4, `NUM_THREADS);
            $write("\n");
        end
    end
`endif

endmodule