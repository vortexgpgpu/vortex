`include "VX_tex_define.vh"

module VX_tex_bilerp #(
    parameter CORE_ID = 0
) (
    input wire [`BLEND_FRAC_64-1:0]   blendU,
    input wire [`BLEND_FRAC_64-1:0]   blendV,

    input wire [3:0][63:0]            texels,
    input wire [`NUM_COLOR_CHANNEL-1:0] color_enable,

    output wire [31:0]                sampled_data
);  
    `UNUSED_PARAM (CORE_ID)

    wire [63:0] UL_lerp;
    wire [63:0] UH_lerp;
    wire [63:0] V_lerp;
    reg [31:0] sampled_r;

    VX_tex_lerp #(
    ) tex_lerp_UL (
        .blend(blendU), 
        .in_texels({texels[1], texels[0]}),
        .lerp_texel(UL_lerp)
    );  

    VX_tex_lerp #(
    ) tex_lerp_UH (
        .blend(blendU), 
        .in_texels({texels[3], texels[2]}),
        .lerp_texel(UH_lerp)
    );  

    VX_tex_lerp #(
    ) tex_lerp_V (
        .blend(blendV), 
        .in_texels({UH_lerp, UL_lerp}),
        .lerp_texel(V_lerp)
    );  

    `UNUSED_VAR (V_lerp[63:56])

    always @(*) begin
        if (color_enable[3]==1'b1) //R
            sampled_r[31:24] = V_lerp[55:48];
        else
            sampled_r[31:24] = {`TEX_COLOR_BITS{1'b0}};

        if (color_enable[2]==1'b1) //G
            sampled_r[23:16] = V_lerp[39:32];
        else
            sampled_r[23:16] = {`TEX_COLOR_BITS{1'b0}};

        if (color_enable[1]==1'b1) //B
            sampled_r[15:8] = V_lerp[23:16];
        else
            sampled_r[15:8] = {`TEX_COLOR_BITS{1'b0}};

        if (color_enable[0]==1'b1) //A
            sampled_r[7:0] = V_lerp[7:0];
        else
            sampled_r[7:0] = {`TEX_COLOR_BITS{1'b1}};
    end

    assign sampled_data = sampled_r;

endmodule