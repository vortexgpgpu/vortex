`include "VX_tex_define.vh"

module VX_tex_sampler #(
    parameter `STRING INSTANCE_ID = "",
    parameter REQ_INFOW = 1,
    parameter NUM_LANES = 1   
) (
    input wire clk,
    input wire reset,

    // inputs
    input wire                          req_valid,
    input wire [`TEX_FORMAT_BITS-1:0]   req_format,    
    input wire [NUM_LANES-1:0][1:0][`TEX_BLEND_FRAC-1:0] req_blends,
    input wire [NUM_LANES-1:0][3:0][31:0] req_data,
    input wire [REQ_INFOW-1:0]          req_info,
    output wire                         req_ready,

    // ouputs
    output wire                         rsp_valid,
    output wire [NUM_LANES-1:0][31:0]   rsp_data,
    output wire [REQ_INFOW-1:0]         rsp_info,    
    input wire                          rsp_ready
);   
    `UNUSED_SPARAM (INSTANCE_ID)
    
    wire valid_s0, valid_s1;
    wire [REQ_INFOW-1:0] req_info_s0, req_info_s1;
    wire [NUM_LANES-1:0][31:0] texel_ul, texel_uh;
    wire [NUM_LANES-1:0][1:0][`TEX_BLEND_FRAC-1:0] req_blends_s0;
    wire [NUM_LANES-1:0][`TEX_BLEND_FRAC-1:0] blend_v_s0, blend_v_s1;
    wire [NUM_LANES-1:0][3:0][31:0] fmt_texels, fmt_texels_s0;

    wire stall_out;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        for (genvar j = 0; j < 4; ++j) begin
            VX_tex_format tex_format (
                .format    (req_format),
                .texel_in  (req_data[i][j]),            
                .texel_out (fmt_texels[i][j])
            );
        end
    end

    VX_pipe_register #(
        .DATAW  (1 + REQ_INFOW + (NUM_LANES * 2 * `TEX_BLEND_FRAC) + (NUM_LANES * 4 * 32)),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valid, req_info,    req_blends,    fmt_texels}),
        .data_out ({valid_s0,  req_info_s0, req_blends_s0, fmt_texels_s0})
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        for (genvar j = 0; j < 4; ++j) begin
            VX_tex_lerp #(
                .LATENCY (3)
            ) tex_lerp_ul (
                .clk  (clk),
                .reset(reset),
                .enable(~stall_out),
                .in1  (fmt_texels_s0[i][0][j*8 +: 8]),
                .in2  (fmt_texels_s0[i][1][j*8 +: 8]),
                .frac (req_blends_s0[i][0]),
                .out  (texel_ul[i][j*8 +: 8])
            );                
            VX_tex_lerp #(
                .LATENCY (3)
            ) tex_lerp_uh (
                .clk  (clk),
                .reset(reset),
                .enable(~stall_out),
                .in1  (fmt_texels_s0[i][2][j*8 +: 8]),
                .in2  (fmt_texels_s0[i][3][j*8 +: 8]),
                .frac (req_blends_s0[i][0]),
                .out  (texel_uh[i][j*8 +: 8])
            );
        end        
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign blend_v_s0[i] = req_blends_s0[i][1];
    end

    VX_shift_register #(
        .DATAW  (1 + REQ_INFOW + (NUM_LANES * `TEX_BLEND_FRAC)),
        .DEPTH  (3),
        .RESETW (1)
    ) shift_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s0, req_info_s0, blend_v_s0}),
        .data_out ({valid_s1, req_info_s1, blend_v_s1})
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        for (genvar j = 0; j < 4; ++j) begin
            VX_tex_lerp #(
                .LATENCY (3)
            ) tex_lerp_v (
                .clk  (clk),
                .reset(reset),
                .enable(~stall_out),
                .in1  (texel_ul[i][j*8 +: 8]),
                .in2  (texel_uh[i][j*8 +: 8]),
                .frac (blend_v_s1[i]),
                .out  (rsp_data[i][j*8 +: 8])
            );
        end
    end

    assign stall_out = rsp_valid && ~rsp_ready;
    
    VX_shift_register #(
        .DATAW  (1 + REQ_INFOW),
        .DEPTH  (3),
        .RESETW (1)
    ) shift_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_s1,  req_info_s1}),
        .data_out ({rsp_valid, rsp_info})
    );

    // can accept new request?
    assign req_ready = ~stall_out;

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin        
        if (req_valid && req_ready) begin
            `TRACE(2, ("%d: %s-sampler-req: format=%0d, data=", $time, INSTANCE_ID, req_format));
            `TRACE_ARRAY2D(2, req_data, 4, NUM_LANES);
            `TRACE(2, (", u0="));
            `TRACE_ARRAY1D(2, req_blends[0], NUM_LANES);
            `TRACE(2, (", v0="));
            `TRACE_ARRAY1D(2, req_blends[1], NUM_LANES);
            `TRACE(2, (" (#%0d)\n", req_info[REQ_INFOW-1 -: `UUID_BITS]));
        end
        if (rsp_valid && rsp_ready) begin
            `TRACE(2, ("%d: %s-sampler-rsp: data=", $time, INSTANCE_ID));
            `TRACE_ARRAY1D(2, rsp_data, NUM_LANES);
            `TRACE(2, (" (#%0d)\n", rsp_info[REQ_INFOW-1 -: `UUID_BITS]));
        end        
    end
`endif  

endmodule
