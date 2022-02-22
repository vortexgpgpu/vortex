`include "VX_rop_define.vh"

module VX_rop_blend #(  
    parameter CORE_ID = 0
) (
    input wire                  clk,
    input wire                  reset,
    input wire                  enable,

    // Blend Equation (TODO: replace with CSRs)
    input wire [:0]             mode_rgb,
    input wire [:0]             mode_a,

    // CSR Parameters
    rop_csrs_t reg_csrs,

    // Color Values (TODO: replace with REQs)
    input wire [31:0]           dst_color,
    input wire [31:0]           src_color,
    output wire [31:0]          out_color,
);

    // Blend Color
    wire [7:0]             const_red;
    wire [7:0]             const_green;
    wire [7:0]             const_blue;
    wire [7:0]             const_alpha;

    wire [`ROP_BLEND_FACTOR_BITS-1:0]  pipe_factor_src_rgb, pipe_factor_rgb_a,
                                       pipe_factor_dst_rgb, pipe_factor_dst_a;
    wire [`ROP_BLEND_FACTOR_BITS-1:0]  factor_src_rgb, factor_src_a,
                                       factor_dst_rgb, factor_dst_a;

    assign const_red   = reg_csrs.blend_const[31:24];
    assign const_green = reg_csrs.blend_const[23:16];
    assign const_blue  = reg_csrs.blend_const[15:8];
    assign const_alpha = reg_csrs.blend_const[7:0];

    // Combinational logic, not sequential. We want the pipe reg to store the output from this block. Reset will be tied to the pipe reg.
    always @(*) begin
        // Blending Functions
        case (func_src_rgb)
            `ROP_BLEND_FACTOR_ZERO:                 factor_src_rgb = 
            `ROP_BLEND_FACTOR_ONE:                  factor_src_rgb = 
            `ROP_BLEND_FACTOR_SRC_RGB:              factor_src_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_src_rgb = 
            `ROP_BLEND_FACTOR_DST_RGB:              factor_src_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_src_rgb = 
            `ROP_BLEND_FACTOR_DST_A:                factor_src_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_rgb = 
            `ROP_BLEND_FACTOR_DST_A:                factor_src_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_rgb = 
            `ROP_BLEND_FACTOR_CONST_RGB:            factor_src_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_src_rgb = 
            `ROP_BLEND_FACTOR_CONST_A:              factor_src_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_src_rgb = 
            `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_src_rgb = 
            default: 
        endcase
        case (func_src_a)
            `ROP_BLEND_FACTOR_ZERO:                 factor_src_a = 
            `ROP_BLEND_FACTOR_ONE:                  factor_src_a = 
            `ROP_BLEND_FACTOR_SRC_RGB:              factor_src_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_src_a = 
            `ROP_BLEND_FACTOR_DST_RGB:              factor_src_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_src_a = 
            `ROP_BLEND_FACTOR_DST_A:                factor_src_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_a = 
            `ROP_BLEND_FACTOR_DST_A:                factor_src_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_a = 
            `ROP_BLEND_FACTOR_CONST_RGB:            factor_src_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_src_a = 
            `ROP_BLEND_FACTOR_CONST_A:              factor_src_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_src_a = 
            `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_src_a = 
            default: 
        endcase
        case (func_dst_rgb)
            `ROP_BLEND_FACTOR_ZERO:                 factor_dst_rgb = 
            `ROP_BLEND_FACTOR_ONE:                  factor_dst_rgb = 
            `ROP_BLEND_FACTOR_SRC_RGB:              factor_dst_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_dst_rgb = 
            `ROP_BLEND_FACTOR_DST_RGB:              factor_dst_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_dst_rgb = 
            `ROP_BLEND_FACTOR_DST_A:                factor_dst_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_rgb = 
            `ROP_BLEND_FACTOR_DST_A:                factor_dst_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_rgb = 
            `ROP_BLEND_FACTOR_CONST_RGB:            factor_dst_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_dst_rgb = 
            `ROP_BLEND_FACTOR_CONST_A:              factor_dst_rgb = 
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_dst_rgb = 
            `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_dst_rgb = 
            default: 
        endcase
        case (func_dst_a)
            `ROP_BLEND_FACTOR_ZERO:                 factor_dst_a = 
            `ROP_BLEND_FACTOR_ONE:                  factor_dst_a = 
            `ROP_BLEND_FACTOR_SRC_RGB:              factor_dst_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_dst_a = 
            `ROP_BLEND_FACTOR_DST_RGB:              factor_dst_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_dst_a = 
            `ROP_BLEND_FACTOR_DST_A:                factor_dst_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_a = 
            `ROP_BLEND_FACTOR_DST_A:                factor_dst_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_a = 
            `ROP_BLEND_FACTOR_CONST_RGB:            factor_dst_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_dst_a = 
            `ROP_BLEND_FACTOR_CONST_A:              factor_dst_a = 
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_dst_a = 
            `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_dst_a = 
            default: 
        endcase
    end

    // Store the original source color in pipe_color in case blending is turned off
    wire enable_out, reset_out;
    wire [31:0] pipe_color;

    VX_pipe_register #(
        .DATAW  (1 + 1 + (`ROP_BLEND_FACTOR_BITS * 4) + 32),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({enable,     reset,     factor_src_rgb,      factor_rgb_a,      factor_dst_rgb,      factor_dst_a,      source_color}),
        .data_out ({enable_out, reset_out, pipe_factor_src_rgb, pipe_factor_rgb_a, pipe_factor_dst_rgb, pipe_factor_dst_a, pipe_color})
    );

    always @(*) begin
        // Blend Equations (TODO: ask about logic op and whether blending is disabled or not when logic op is used as per GL spec)
        case (mode_rgb)
            `ROP_BLEND_MODE_FUNC_ADD: begin
                out_red <=
                out_blue <=
                out_green <=
            end
            `ROP_BLEND_MODE_FUNC_SUBTRACT:
            `ROP_BLEND_MODE_FUNC_REVERSE_SUBTRACT:
            `ROP_BLEND_MODE_MIN:
            `ROP_BLEND_MODE_MAX:
            `ROP_BLEND_MODE_LOGIC_OP:
            default: 
        endcase
        case (mode_a)
            `ROP_BLEND_MODE_FUNC_ADD: begin
                out_a <=
            end
            `ROP_BLEND_MODE_FUNC_SUBTRACT:
            `ROP_BLEND_MODE_FUNC_REVERSE_SUBTRACT:
            `ROP_BLEND_MODE_MIN:
            `ROP_BLEND_MODE_MAX:
            `ROP_BLEND_MODE_LOGIC_OP:
            default: 
        endcase
    end

    // If enable is off, pass the original source color (now stored in pipe_color) on to the next stage and ignore blending
    wire [31:0] pipe_color_out;
    assign pipe_color_out = (enable_out) ? {out_red, out_green, out_blue, out_a} : pipe_color;

    VX_pipe_register #(
        .DATAW  (1 + 1 + 32),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset_out),
        .enable   (enable_out),
        .data_in  ({enable_out, reset_out, pipe_color_out,      ,      ,      }),
        .data_out ({          ,          , out_color,           ,      ,      })
    );

endmodule