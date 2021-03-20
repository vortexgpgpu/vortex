`include "VX_tex_define.vh"

module VX_tex_format #(
    parameter CORE_ID = 0
) (
    input wire [31:0]                    texel_data,
    input wire [`TEX_FORMAT_BITS-1:0]    format,

    output wire [`NUM_COLOR_CHANNEL-1:0] color_enable,
    output wire [`MAX_COLOR_BITS-1:0]    R,
    output wire [`MAX_COLOR_BITS-1:0]    G,
    output wire [`MAX_COLOR_BITS-1:0]    B,
    output wire [`MAX_COLOR_BITS-1:0]    A
);  
    `UNUSED_PARAM (CORE_ID)

    reg [`NUM_COLOR_CHANNEL-1:0] color_enable_r;
    reg [`MAX_COLOR_BITS-1:0]    R_r;
    reg [`MAX_COLOR_BITS-1:0]    G_r;
    reg [`MAX_COLOR_BITS-1:0]    B_r;
    reg [`MAX_COLOR_BITS-1:0]    A_r;

    always @(*) begin
        case (format)
            `R5G6B5:
                R_r = `MAX_COLOR_BITS'(texel_data[15:11]);
                G_r = `MAX_COLOR_BITS'(texel_data[10:5]);
                B_r = `MAX_COLOR_BITS'(texel_data[4:0]);
                A_r = {`MAX_COLOR_BITS{1'b0}};
                color_enable_r = 4'b1110;

            `R8G8B8:
                R_r = `MAX_COLOR_BITS'(texel_data[23:16]);
                G_r = `MAX_COLOR_BITS'(texel_data[15:8]);
                B_r = `MAX_COLOR_BITS'(texel_data[7:0]);
                A_r = {`MAX_COLOR_BITS{1'b0}};
                color_enable_r = 4'b1110;

            `R8G8B8A8:
                R_r = `MAX_COLOR_BITS'(texel_data[31:24]);
                G_r = `MAX_COLOR_BITS'(texel_data[23:16]);
                B_r = `MAX_COLOR_BITS'(texel_data[15:8]);
                A_r = `MAX_COLOR_BITS'(texel_data[7:0]);
                color_enable_r = 4'b1111;  

            default: 
                R_r = `MAX_COLOR_BITS'(texel_data[23:16]);
                G_r = `MAX_COLOR_BITS'(texel_data[15:8]);
                B_r = `MAX_COLOR_BITS'(texel_data[7:0]);
                A_r = {`MAX_COLOR_BITS{1'b0}};
                color_enable_r = 4'b1110;
        endcase
    end

    assign color_enable = color_enable_r;
    assign R = R_r;
    assign G = G_r;
    assign B = B_r;
    assign A = A_r;   

endmodule