`include "VX_tex_define.vh"

module VX_tex_sampler #(
    parameter CORE_ID   = 0,
    parameter REQ_INFOW = 1,
    parameter NUM_REQS  = 1   
) (
    input wire clk,
    input wire reset,

    // inputs
    input wire                          req_valid,   
    input wire [NUM_REQS-1:0]           req_tmask, 
    input wire [`TEX_FORMAT_BITS-1:0]   req_format,    
    input wire [NUM_REQS-1:0][1:0][`TEX_BLEND_FRAC-1:0] req_blends,
    input wire [NUM_REQS-1:0][3:0][31:0] req_data,
    input wire [REQ_INFOW-1:0]          req_info,
    output wire                         req_ready,

    // ouputs
    output wire                         rsp_valid,
    output wire [NUM_REQS-1:0]          rsp_tmask, 
    output wire [NUM_REQS-1:0][31:0]    rsp_data,
    output wire [REQ_INFOW-1:0]         rsp_info,    
    input wire                          rsp_ready
);
    
    `UNUSED_PARAM (CORE_ID)
   
    wire valid_s0, valid_s1;
    wire [NUM_REQS-1:0] req_tmask_s0, req_tmask_s1; 
    wire [REQ_INFOW-1:0] req_info_s0, req_info_s1;
    wire [NUM_REQS-1:0][31:0] texel_ul, texel_uh;
    wire [NUM_REQS-1:0][31:0] texel_ul_s1, texel_uh_s1;
    wire [NUM_REQS-1:0][1:0][`TEX_BLEND_FRAC-1:0] req_blends_s0;
    wire [NUM_REQS-1:0][`TEX_BLEND_FRAC-1:0] blend_v, blend_v_s1;
    wire [NUM_REQS-1:0][31:0] texel_v;
    wire [NUM_REQS-1:0][3:0][31:0] fmt_texels, fmt_texels_s0;

    wire stall_out;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        for (genvar j = 0; j < 4; ++j) begin
            VX_tex_format #(
                .CORE_ID (CORE_ID)
            ) tex_format (
                .format    (req_format),
                .texel_in  (req_data[i][j]),            
                .texel_out (fmt_texels[i][j])
            );
        end
    end

    VX_pipe_register #(
        .DATAW  (1 + NUM_REQS + REQ_INFOW + (NUM_REQS * 2 * `TEX_BLEND_FRAC) + (NUM_REQS * 4 * 32)),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_tmask,    req_info,    req_blends,    fmt_texels}),
        .data_out ({valid_s0,  req_tmask_s0, req_info_s0, req_blends_s0, fmt_texels_s0})
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        VX_tex_lerp #(
        ) tex_lerp_ul (
            .in1   (fmt_texels_s0[i][0]),
            .in2   (fmt_texels_s0[i][1]),
            .frac  (req_blends_s0[i][0]),
            .out   (texel_ul[i])
        );  

        VX_tex_lerp #(
        ) tex_lerp_uh (
            .in1   (fmt_texels_s0[i][2]),
            .in2   (fmt_texels_s0[i][3]),
            .frac  (req_blends_s0[i][0]),
            .out   (texel_uh[i])
        );

        assign blend_v[i] = req_blends_s0[i][1];
    end

    VX_pipe_register #(
        .DATAW  (1 + NUM_REQS + REQ_INFOW + (NUM_REQS * `TEX_BLEND_FRAC) + (2 * NUM_REQS * 32)),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s0, req_tmask_s0, req_info_s0, blend_v,    texel_ul,    texel_uh}),
        .data_out ({valid_s1, req_tmask_s1, req_info_s1, blend_v_s1, texel_ul_s1, texel_uh_s1})
    );

    for (genvar i = 0; i < NUM_REQS; i++) begin
        VX_tex_lerp #(
        ) tex_lerp_v (
            .in1   (texel_ul_s1[i]),
            .in2   (texel_uh_s1[i]),
            .frac  (blend_v_s1[i]),
            .out   (texel_v[i])
        );
    end

    assign stall_out = rsp_valid && ~rsp_ready;
    
    VX_pipe_register #(
        .DATAW  (1 + NUM_REQS + REQ_INFOW + (NUM_REQS * 32)),
        .RESETW (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s1,  req_tmask_s1, req_info_s1, texel_v}),
        .data_out ({rsp_valid, rsp_tmask,    rsp_info,    rsp_data})
    );

    // can accept new request?
    assign req_ready = ~stall_out;   

`ifdef DBG_TRACE_TEX
    wire [`NW_BITS-1:0] req_wid, rsp_wid;
    wire [31:0]         req_PC, rsp_PC;

    assign {req_wid, req_PC} = req_info[`NW_BITS+32-1:0];
    assign {rsp_wid, rsp_PC} = rsp_info[`NW_BITS+32-1:0];

    always @(posedge clk) begin        
        if (req_valid && req_ready) begin
            dpi_trace("%d: core%0d-tex-sampler-req: wid=%0d, PC=%0h, tmask=%b, format=%0d, data=", 
                    $time, CORE_ID, req_wid, req_PC, req_tmask, req_format);
            `TRACE_ARRAY2D(req_data, 4, NUM_REQS);
            dpi_trace(", u0=");
            `TRACE_ARRAY1D(req_blends[0], NUM_REQS);
            dpi_trace(", v0=");
            `TRACE_ARRAY1D(req_blends[1], NUM_REQS);
            dpi_trace("\n");
        end
        if (rsp_valid && rsp_ready) begin
            dpi_trace("%d: core%0d-tex-sampler-rsp: wid=%0d, PC=%0h, tmask=%b, data=", 
                    $time, CORE_ID, rsp_wid, rsp_PC, rsp_tmask);
            `TRACE_ARRAY1D(rsp_data, NUM_REQS);
            dpi_trace("\n");
        end        
    end
`endif  

endmodule