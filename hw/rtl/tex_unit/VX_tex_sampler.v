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
    input wire [`NUM_THREADS-1:0][3:0][31:0] req_data,
    input wire [`NUM_THREADS-1:0][`BLEND_FRAC-1:0] req_blend_u,
    input wire [`NUM_THREADS-1:0][`BLEND_FRAC-1:0] req_blend_v,
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

    wire [`NUM_THREADS-1:0][31:0] result;

    wire stall_out;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin

        wire [3:0][31:0] fmt_texels;
        wire [31:0] texel_ul, texel_uh, texel_v;

        for (genvar j = 0; j < 4; j++) begin
            VX_tex_format #(
                .CORE_ID (CORE_ID)
            ) tex_format (
                .format    (req_format),
                .texel_in  (req_data[i][j]),            
                .texel_out (fmt_texels[j])
            );
        end 

        VX_tex_lerp #(
        ) tex_lerp_ul (
            .blend (req_blend_u[i]), 
            .in1 (fmt_texels[0]),
            .in2 (fmt_texels[1]),
            .out (texel_ul)
        );  

        VX_tex_lerp #(
        ) tex_lerp_uh (
            .blend (req_blend_u[i]), 
            .in1 (fmt_texels[2]),
            .in2 (fmt_texels[3]),
            .out (texel_uh)
        );  

        VX_tex_lerp #(
        ) tex_lerp_v (
            .blend (req_blend_v[i]), 
            .in1 (texel_ul),
            .in2 (texel_uh),
            .out (texel_v)
        );

        assign result[i] = req_filter ? texel_v : fmt_texels[0];
    end

    assign stall_out = rsp_valid && ~rsp_ready;
    
    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_wid, req_tmask, req_PC, req_rd, req_wb, result}),
        .data_out ({rsp_valid, rsp_wid, rsp_tmask, rsp_PC, rsp_rd, rsp_wb, rsp_data})
    );

    // can accept new request?
    assign req_ready = ~stall_out;   

`ifdef DBG_PRINT_TEX
   always @(posedge clk) begin        
        if (req_valid && req_ready) begin
            $write("%t: core%0d-tex-sampler-req: wid=%0d, PC=%0h, tmask=%b, filter=%0d, format=%0d, data=", 
                    $time, CORE_ID, req_wid, req_PC, req_tmask, req_filter, req_format);
            `PRINT_ARRAY2D(req_data, 4, `NUM_THREADS);
            $write("u0=");
            `PRINT_ARRAY1D(req_blend_u, `NUM_THREADS);
            $write("v0=");
            `PRINT_ARRAY1D(req_blend_v, `NUM_THREADS);
            $write("\n");
        end
        if (rsp_valid && rsp_ready) begin
            $write("%t: core%0d-tex-sampler-rsp: wid=%0d, PC=%0h, tmask=%b, data=", 
                    $time, CORE_ID, rsp_wid, rsp_PC, rsp_tmask);
            `PRINT_ARRAY1D(rsp_data, `NUM_THREADS);
            $write("\n");
        end        
    end
`endif  

endmodule