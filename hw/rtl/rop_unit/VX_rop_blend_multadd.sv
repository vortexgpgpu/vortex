`include "VX_rop_define.vh"

module VX_rop_blend_multadd #(  
    parameter CORE_ID = 0
) (
    // Mode used to determine which values to multiply and whether to use addition or subtraction.
    input wire [`ROP_BLEND_MODE_BITS-1:0] mode_rgb,
    input wire [`ROP_BLEND_MODE_BITS-1:0] mode_a,

    input wire [31:0] src_color,
    input wire [31:0] dst_color,

    input wire [31:0] src_blend_factor,
    input wire [31:0] dst_blend_factor,

    output wire [31:0] color_out
);

    wire [7:0] src_red, src_green, src_blue, src_alpha, dst_red, dst_green, dst_blue, dst_alpha;
    wire [7:0] src_blend_red, src_blend_green, src_blend_blue, src_blend_alpha, dst_blend_red, dst_blend_green, dst_blend_blue, dst_blend_alpha;
    wire [15:0] red_combined, green_combined, blue_combined, alpha_combined;

    assign src_red     = src_color[31:24];
    assign src_green   = src_color[23:16];
    assign src_blue    = src_color[15:8];
    assign src_alpha   = src_color[7:0];

    assign dst_red     = dst_color[31:24];
    assign dst_green   = dst_color[23:16];
    assign dst_blue    = dst_color[15:8];
    assign dst_alpha   = dst_color[7:0];

    assign src_blend_red     = src_blend_factor[31:24];
    assign src_blend_green   = src_blend_factor[23:16];
    assign src_blend_blue    = src_blend_factor[15:8];
    assign src_blend_alpha   = src_blend_factor[7:0];

    assign dst_blend_red     = dst_blend_factor[31:24];
    assign dst_blend_green   = dst_blend_factor[23:16];
    assign dst_blend_blue    = dst_blend_factor[15:8];
    assign dst_blend_alpha   = dst_blend_factor[7:0];

    always @(*) begin
        // RGB blending
        case(mode_rgb)
            `ROP_BLEND_MODE_FUNC_ADD: begin
                red_combined   = (src_red * src_blend_red) + (dst_red * dst_blend_red);
                green_combined = (src_green * src_blend_green) + (dst_green * dst_blend_green);
                blue_combined  = (src_blue * src_blend_blue) + (dst_blue * dst_blend_blue);
            end
            `ROP_BLEND_MODE_FUNC_SUBTRACT: begin
                red_combined   = (src_red * src_blend_red) - (dst_red * dst_blend_red);
                green_combined = (src_green * src_blend_green) - (dst_green * dst_blend_green);
                blue_combined  = (src_blue * src_blend_blue) - (dst_blue * dst_blend_blue); 
            end
            `ROP_BLEND_MODE_FUNC_REVERSE_SUBTRACT: begin
                red_combined   = (dst_red * src_blend_red) - (src_red * dst_blend_red);
                green_combined = (dst_green * src_blend_green) - (src_green * dst_blend_green);
                blue_combined  = (dst_blue * src_blend_blue) - (src_blue * dst_blend_blue);
            end
            default: begin
                red_combined   = {src_red, src_red};
                green_combined = {src_green, src_green};
                blue_combined  = {src_blue, src_blue};
            end
        endcase
        // Alpha blending
        case(mode_a)
            `ROP_BLEND_MODE_FUNC_ADD: begin
                alpha_combined = (src_alpha * src_blend_alpha) + (dst_alpha * dst_blend_alpha);
            end
            `ROP_BLEND_MODE_FUNC_SUBTRACT: begin
                alpha_combined = (src_alpha * src_blend_alpha) - (dst_alpha * dst_blend_alpha);
            end
            `ROP_BLEND_MODE_FUNC_REVERSE_SUBTRACT: begin
                alpha_combined = (dst_alpha * src_blend_alpha) - (src_alpha * dst_blend_alpha);
            end
            default: begin
                alpha_combined = {src_alpha, src_alpha};
            end
        endcase
    end

    assign color_out = {red_combined[15:8], green_combined[15:8], blue_combined[15:8], alpha_combined[15:8]};

    // How to do math because the color constant is float between [0, 1] and 
    // the RGBA values are 8 bit ints. Even if we do fixed point math, how is 
    // this accomplished? How many bits could the answer be and how do we trim 
    // that down to an 8 bit result?

    // I need to know what type of number and in what form the color constant 
    // will be coming in, I need to know how to handle overflow like when I 
    // multiply src_rgba by itself and get a larger than 8 bit number. These
    // are all possible cases. src_blend_factor/dst_blend_factor could be 
    // set to the orginal src/dst values or to a color constant.

    // Answer:
    // Simply treat the constant color value the same as a regular color,
    // despite what the OpenGL documentation says. AFAIK, it will be handled in
    // the software stack or higher-level modules.
    // As for the multiplication, I think you'd want to take the upeer half,
    // which is consistent with VX_tex_lerp.

endmodule