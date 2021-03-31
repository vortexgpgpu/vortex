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

        wire [31:0]                 req_data_bilerp;
        wire [3:0][63:0]            formatted_data;
        wire [31:0]                 formatted_pt_data;
        wire [`NUM_COLOR_CHANNEL-1:0] color_enable;

        VX_tex_format #(
            .CORE_ID (CORE_ID),
            .NUM_TEXELS (4)
        ) tex_format_texel (
            .texel_data  (req_texels[i]),
            .format (req_format),

            .color_enable (color_enable),
            .formatted_lerp_texel(formatted_data),
            .formatted_pt_texel(formatted_pt_data)
        );  

        VX_tex_bilerp #(
            .CORE_ID (CORE_ID)
        ) tex_bilerp (
            .blendU (req_u[i][`BLEND_FRAC_64-1:0]),
            .blendV (req_v[i][`BLEND_FRAC_64-1:0]),

            .color_enable (color_enable),
            .texels (formatted_data),
            
            .sampled_data (req_data_bilerp)
        );    

        assign req_data[i] = (req_filter == `TEX_FILTER_BITS'(0)) ? formatted_pt_data : req_data_bilerp;

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