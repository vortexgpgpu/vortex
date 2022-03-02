`include "VX_rop_define.vh"

module VX_rop_blend #(  
    parameter CORE_ID = 0
) (
    input wire                  clk,
    input wire                  reset,
    input wire                  enable,

    // CSR Parameters
    rop_csrs_t                  reg_csrs,

    // Color Values 
    input wire [31:0]           dst_color,
    input wire [31:0]           src_color,
    output wire [31:0]          out_color
);

    // Blend Color
    wire [7:0]             const_red;
    wire [7:0]             const_green;
    wire [7:0]             const_blue;
    wire [7:0]             const_alpha;

    wire [23:0] factor_src_rgb, factor_dst_rgb;
    wire [7:0] factor_src_a, factor_dst_a;

    // // considering mapping out input color channels for readability
    // wire [7:0] src_red, src_green, src_blue, src_alpha;
    // wire [7:0] dst_red, dst_green, dst_blue, dst_alpha;

    // assign src_red     = src_color[31:24];
    // assign src_green   = src_color[23:16];
    // assign src_blue    = src_color[15:8];
    // assign src_alpha   = src_color[7:0];

    // assign dst_red     = dst_color[31:24];
    // assign dst_green   = dst_color[23:16];
    // assign dst_blue    = dst_color[15:8];
    // assign dst_alpha   = dst_color[7:0];

    assign const_red   = reg_csrs.blend_const[31:24];
    assign const_green = reg_csrs.blend_const[23:16];
    assign const_blue  = reg_csrs.blend_const[15:8];
    assign const_alpha = reg_csrs.blend_const[7:0];

    // Combinational logic, not sequential. We want the pipe reg to store the output from this block. Reset will be tied to the pipe reg.
    always @(*) begin
        // Blending Functions
        case (reg_csrs.blend_func_src_rgb)
            `ROP_BLEND_FACTOR_ZERO:                 factor_src_rgb = {24{1'b0}};
            `ROP_BLEND_FACTOR_ONE:                  factor_src_rgb = {24{1'b1}};
            `ROP_BLEND_FACTOR_SRC_RGB:              factor_src_rgb = src_color[31:8];
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_src_rgb = {8'hFF - src_color[31:24], 8'hFF - src_color[23:16], 8'hFF - src_color[15:8]};
            `ROP_BLEND_FACTOR_DST_RGB:              factor_src_rgb = dst_color[31:8];
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_src_rgb = {8'hFF - dst_color[31:24], 8'hFF - dst_color[23:16], 8'hFF - dst_color[15:8]};
            `ROP_BLEND_FACTOR_SRC_A:                factor_src_rgb = {src_color[7:0], src_color[7:0], src_color[7:0]};
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_A:      factor_src_rgb = {8'hFF - src_color[7:0], 8'hFF - src_color[7:0], 8'hFF - src_color[7:0]};
            `ROP_BLEND_FACTOR_DST_A:                factor_src_rgb = {dst_color[7:0], dst_color[7:0], dst_color[7:0]};
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_rgb = {8'hFF - dst_color[7:0], 8'hFF - dst_color[7:0], 8'hFF - dst_color[7:0]};
            `ROP_BLEND_FACTOR_CONST_RGB:            factor_src_rgb = {const_red, const_green, const_blue};
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_src_rgb = {8'hFF - const_red, 8'hFF - const_green, 8'hFF - const_blue};
            `ROP_BLEND_FACTOR_CONST_A:              factor_src_rgb = {const_alpha, const_alpha, const_alpha};
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_src_rgb = {8'hFF - const_alpha, 8'hFF - const_alpha, 8'hFF - const_alpha};
            `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_src_rgb = 
            default:                                factor_src_rgb = {24{1'b1}};
        endcase
        case (reg_csrs.blend_func_src_a)
            `ROP_BLEND_FACTOR_ZERO:                 factor_src_a = 8'h00;
            `ROP_BLEND_FACTOR_ONE:                  factor_src_a = 8'hFF;
            `ROP_BLEND_FACTOR_SRC_RGB:              factor_src_a = src_color[7:0];
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_src_a = 8'hFF - src_color[7:0];
            `ROP_BLEND_FACTOR_DST_RGB:              factor_src_a = dst_color[7:0];
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_src_a = 8'hFF - dst_color[7:0];
            `ROP_BLEND_FACTOR_SRC_A:                factor_src_a = src_color[7:0];
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_A:      factor_src_a = 8'hFF - src_color[7:0];
            `ROP_BLEND_FACTOR_DST_A:                factor_src_a = dst_color[7:0];
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_a = 8'hFF - dst_color[7:0];
            `ROP_BLEND_FACTOR_CONST_RGB:            factor_src_a = const_alpha;
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_src_a = 8'hFF - const_alpha;
            `ROP_BLEND_FACTOR_CONST_A:              factor_src_a = const_alpha;
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_src_a = 8'hFF - const_alpha;
            `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_src_a = 8'hFF;
            default:                                factor_src_a = {8{1'b1}};
        endcase
        case (reg_csrs.blend_func_dst_rgb)
            `ROP_BLEND_FACTOR_ZERO:                 factor_dst_rgb = {24{1'b0}};
            `ROP_BLEND_FACTOR_ONE:                  factor_dst_rgb = {24{1'b1}};
            `ROP_BLEND_FACTOR_SRC_RGB:              factor_dst_rgb = src_color[31:8];
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_dst_rgb = {8'hFF - src_color[31:24], 8'hFF - src_color[23:16], 8'hFF - src_color[15:8]};
            `ROP_BLEND_FACTOR_DST_RGB:              factor_dst_rgb = dst_color[31:8];
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_dst_rgb = {8'hFF - dst_color[31:24], 8'hFF - dst_color[23:16], 8'hFF - dst_color[15:8]};
            `ROP_BLEND_FACTOR_SRC_A:                factor_dst_rgb = {src_color[7:0], src_color[7:0], src_color[7:0]};
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_A:      factor_dst_rgb = {8'hFF - src_color[7:0], 8'hFF - src_color[7:0], 8'hFF - src_color[7:0]};
            `ROP_BLEND_FACTOR_DST_A:                factor_dst_rgb = {dst_color[7:0], dst_color[7:0], dst_color[7:0]};
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_rgb = {8'hFF - dst_color[7:0], 8'hFF - dst_color[7:0], 8'hFF - dst_color[7:0]};
            `ROP_BLEND_FACTOR_CONST_RGB:            factor_dst_rgb = {const_red, const_green, const_blue};
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_dst_rgb = {8'hFF - const_red, 8'hFF - const_green, 8'hFF - const_blue};
            `ROP_BLEND_FACTOR_CONST_A:              factor_dst_rgb = {const_alpha, const_alpha, const_alpha};
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_dst_rgb = {8'hFF - const_alpha, 8'hFF - const_alpha, 8'hFF - const_alpha};
            `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_dst_rgb = 
            default:                                factor_dst_rgb = {24{1'b0}};
        endcase
        case (reg_csrs.blend_func_dst_a)
            `ROP_BLEND_FACTOR_ZERO:                 factor_dst_a = 8'h00;
            `ROP_BLEND_FACTOR_ONE:                  factor_dst_a = 8'hFF;
            `ROP_BLEND_FACTOR_SRC_RGB:              factor_dst_a = src_color[7:0];
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_dst_a = 8'hFF - src_color[7:0];
            `ROP_BLEND_FACTOR_DST_RGB:              factor_dst_a = dst_color[7:0];
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_dst_a = 8'hFF - dst_color[7:0];
            `ROP_BLEND_FACTOR_SRC_A:                factor_dst_a = src_color[7:0];
            `ROP_BLEND_FACTOR_ONE_MINUS_SRC_A:      factor_dst_a = 8'hFF - src_color[7:0];
            `ROP_BLEND_FACTOR_DST_A:                factor_dst_a = dst_color[7:0];
            `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_a = 8'hFF - dst_color[7:0];
            `ROP_BLEND_FACTOR_CONST_RGB:            factor_dst_a = const_alpha;
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_dst_a = 8'hFF - const_alpha;
            `ROP_BLEND_FACTOR_CONST_A:              factor_dst_a = const_alpha;
            `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_dst_a = 8'hFF - const_alpha;
            `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_dst_a = 8'hFF;
            default:                                factor_dst_a = 8'h00;
        endcase
    end

    // Store the original source color in pipe_color in case blending is turned off
    wire pipe_enable, pipe_reset;
    wire [31:0] pipe_src_color, pipe_dst_color;
    wire [23:0] pipe_factor_src_rgb, pipe_factor_dst_rgb;
    wire [7:0] pipe_factor_src_a, pipe_factor_dst_a;
    wire [`ROP_LOGIC_OP_BITS-1:0] pipe_logic_op;
    wire [`ROP_BLEND_MODE_BITS-1:0] pipe_blend_mode_rgb, pipe_blend_mode_a;

    VX_pipe_register #(
        .DATAW  (1 + 1 + `ROP_LOGIC_OP_BITS + 2*`ROP_BLEND_MODE_BITS + 24 + 8 + 24 + 8 + 32 + 32),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({enable,      reset,      reg_csrs.logic_op, reg_csrs.blend_mode_rgb, reg_csrs.blend_mode_a, factor_src_rgb,      factor_src_a,      factor_dst_rgb,      factor_dst_a,      src_color,      dst_color}),
        .data_out ({pipe_enable, pipe_reset, pipe_logic_op,     pipe_blend_mode_rgb,     pipe_blend_mode_a,     pipe_factor_src_rgb, pipe_factor_src_a, pipe_factor_dst_rgb, pipe_factor_dst_a, pipe_src_color, pipe_dst_color})
    );

    wire [31:0] mult_add_color_out, min_color_out, max_color_out, rop_logic_color_out;
    wire [31:0] color_out;

    VX_rop_mult_add #(
    ) mult_add (
        .mode_rgb(pipe_blend_mode_rgb),
        .mode_a(pipe_blend_mode_a),
        .src_color(pipe_src_color),
        .dst_color(pipe_dst_color),
        .src_blend_factor({pipe_factor_src_rgb, pipe_factor_src_a}),
        .dst_blend_factor({pipe_factor_dst_rgb, pipe_factor_dst_a}),
        .color_out(mult_add_color_out)
    );

    VX_rop_min_max #(
    ) min_max (
        .src_color(pipe_src_color),
        .dst_color(pipe_dst_color),
        .min_color_out(min_color_out),
        .max_color_out(max_color_out)
    );

    VX_rop_logic #(
    ) rop_logic (
        .logic_op(pipe_logic_op),
        .src_color(pipe_src_color),
        .dst_color(pipe_dst_color),
        .color_out(rop_logic_color_out)
    );

    always @(*) begin
        // Blend Equations
        // RGB Component
        case (pipe_blend_mode_rgb)
            `ROP_BLEND_MODE_FUNC_ADD: begin
                color_out[31:8] = mult_add_color_out[31:8];
                end
            `ROP_BLEND_MODE_FUNC_SUBTRACT: begin
                color_out[31:8] = mult_add_color_out[31:8];
                end
            `ROP_BLEND_MODE_FUNC_REVERSE_SUBTRACT: begin
                color_out[31:8] = mult_add_color_out[31:8];
                end
            `ROP_BLEND_MODE_MIN: begin
                color_out[31:8] = min_color_out[31:8];
                end
            `ROP_BLEND_MODE_MAX: begin
                color_out[31:8] = max_color_out[31:8];
                end
            `ROP_BLEND_MODE_LOGIC_OP: begin
                color_out[31:8] = rop_logic_color_out[31:8];
                end
            default: begin
                // Figure out default
                end
        endcase
        // Alpha Component
        case (pipe_blend_mode_a)
            `ROP_BLEND_MODE_FUNC_ADD: begin
                color_out[7:0] = mult_add_color_out[7:0];
                end
            `ROP_BLEND_MODE_FUNC_SUBTRACT: begin
                color_out[7:0] = mult_add_color_out[7:0];
                end
            `ROP_BLEND_MODE_FUNC_REVERSE_SUBTRACT: begin
                color_out[7:0] = mult_add_color_out[7:0];
                end
            `ROP_BLEND_MODE_MIN: begin
                color_out[7:0] = min_color_out[7:0];
                end
            `ROP_BLEND_MODE_MAX: begin
                color_out[7:0] = max_color_out[7:0];
                end
            `ROP_BLEND_MODE_LOGIC_OP: begin
                color_out[7:0] = rop_logic_color_out[7:0];
                end
            default: begin
                // Figure out defualt
                end
        endcase
    end

    // If enable is off, pass the original source color (now stored in pipe_color) on to the next stage and ignore blending
    wire [31:0] pipe_color_out;

    assign pipe_color_out = (enable_out) ? color_out : pipe_src_color;

    VX_pipe_register #(
        .DATAW  (32),
        .RESETW (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (pipe_reset),
        .enable   (pipe_enable),
        .data_in  ({pipe_color_out}),
        .data_out ({out_color})
    );

endmodule