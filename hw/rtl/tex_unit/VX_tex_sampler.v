`include "VX_tex_define.vh"

module VX_tex_sampler #(
    parameter CORE_ID = 0    
) (
    input wire clk,
    input wire reset,

    // inputs
    input wire                          req_valid,
    input wire [`NW_BITS-1:0]           req_wid,
    input wire [`NUM_THREADS-1:0]       req_tmask,
    input wire [31:0]                   req_PC,
    input wire [`NR_BITS-1:0]           req_rd,   
    input wire                          req_wb,
    input wire [`TEX_FILTER_BITS-1:0]   req_filter,
    input wire [`TEX_FORMAT_BITS-1:0]   req_format,
    input wire [`NUM_THREADS-1:0][`FIXED_FRAC-1:0] req_u,
    input wire [`NUM_THREADS-1:0][`FIXED_FRAC-1:0] req_v,
    input wire [`NUM_THREADS-1:0][3:0][31:0] req_texels,
    output wire                         req_ready,

    // ouputs
    output wire                          rsp_valid,
    output wire [`NW_BITS-1:0]           rsp_wid,
    output wire [`NUM_THREADS-1:0]       rsp_tmask,
    output wire [31:0]                   rsp_PC,
    output wire [`NR_BITS-1:0]           rsp_rd,   
    output wire                          rsp_wb,
    output wire [`NUM_THREADS-1:0][31:0] rsp_data,
    input wire                           rsp_ready
);
    
    `UNUSED_PARAM (CORE_ID)

    wire [`NUM_THREADS-1:0][31:0] req_data;

    wire stall_out;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin

        wire [3:0][31:0] fmt_texels;
        wire [31:0] texel_ul, texel_uh, texel_v;

        wire [`BLEND_FRAC-1:0] blend_u = req_u[i][`BLEND_FRAC-1:0];
        wire [`BLEND_FRAC-1:0] blend_v = req_v[i][`BLEND_FRAC-1:0];

        for (genvar j = 0; j < 4; j++) begin
            VX_tex_format #(
                .CORE_ID (CORE_ID)
            ) tex_format (
                .format    (req_format),
                .texel_in  (req_texels[i][j]),            
                .texel_out (fmt_texels[j])
            );
        end 

        VX_tex_lerp #(
        ) tex_lerp_ul (
            .blend (blend_u), 
            .in1 (fmt_texels[0]),
            .in2 (fmt_texels[1]),
            .out (texel_ul)
        );  

        VX_tex_lerp #(
        ) tex_lerp_uh (
            .blend (blend_u), 
            .in1 (fmt_texels[2]),
            .in2 (fmt_texels[3]),
            .out (texel_uh)
        );  

        VX_tex_lerp #(
        ) tex_lerp_v (
            .blend (blend_v), 
            .in1 (texel_ul),
            .in2 (texel_uh),
            .out (texel_v)
        );

        assign req_data[i] = req_filter ? texel_v : fmt_texels[0];
    end

    assign stall_out = rsp_valid && ~rsp_ready;
    
    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_wid, req_tmask, req_PC, req_rd, req_wb, req_data}),
        .data_out ({rsp_valid, rsp_wid, rsp_tmask, rsp_PC, rsp_rd, rsp_wb, rsp_data})
    );

    // can accept new request?
    assign req_ready = ~stall_out;     

endmodule