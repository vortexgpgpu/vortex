`include "VX_rop_define.vh"

module VX_rop_mult_add #(  
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

    // How to do math because the color constant is float between [0, 1] and 
    // the RGBA values are 8 bit ints. Even if we do fixed point math, how is 
    // this accomplished? How many bits could the answer be and how do we trim 
    // that down to an 8 bit result?

    // I need to know what type of number and in what form the color constant 
    // will be coming in, I need to know how to handle overflow like when I 
    // multiply src_rgba by itself and get a larger than 8 bit number. These
    // are all possible cases. src_blend_factor/dst_blend_factor could be 
    // set to the orginal src/dst values or to a color constant.

endmodule