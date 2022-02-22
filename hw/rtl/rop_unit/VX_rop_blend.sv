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
    rop_csrs_t reg_csrs;
    wire [`ROP_BLEND_FACTOR_BITS-1:0]  pipe_factor_src_rgb, pipe_factor_rgb_a,
                                       pipe_factor_dst_rgb, pipe_factor_dst_a;
    logic [`ROP_BLEND_FACTOR_BITS-1:0] factor_src_rgb, factor_src_a,
                                       factor_dst_rgb, factor_dst_a;

    assign const_red   = reg_csrs.blend_const[31:24];
    assign const_green = reg_csrs.blend_const[23:16];
    assign const_blue  = reg_csrs.blend_const[15:8];
    assign const_alpha = reg_csrs.blend_const[7:0];

    always @(posedge clock) begin
        if (reset) begin
            // reset code
        end
        else if (enable) begin
            // Blending Functions
            case (func_src_rgb)
                `ROP_BLEND_FACTOR_ZERO:                 factor_src_rgb <= 
                `ROP_BLEND_FACTOR_ONE:                  factor_src_rgb <= 
                `ROP_BLEND_FACTOR_SRC_RGB:              factor_src_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_src_rgb <= 
                `ROP_BLEND_FACTOR_DST_RGB:              factor_src_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_src_rgb <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_src_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_rgb <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_src_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_rgb <= 
                `ROP_BLEND_FACTOR_CONST_RGB:            factor_src_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_src_rgb <= 
                `ROP_BLEND_FACTOR_CONST_A:              factor_src_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_src_rgb <= 
                `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_src_rgb <= 
                default: 
            endcase
            case (func_src_a)
                `ROP_BLEND_FACTOR_ZERO:                 factor_src_a <= 
                `ROP_BLEND_FACTOR_ONE:                  factor_src_a <= 
                `ROP_BLEND_FACTOR_SRC_RGB:              factor_src_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_src_a <= 
                `ROP_BLEND_FACTOR_DST_RGB:              factor_src_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_src_a <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_src_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_a <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_src_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_a <= 
                `ROP_BLEND_FACTOR_CONST_RGB:            factor_src_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_src_a <= 
                `ROP_BLEND_FACTOR_CONST_A:              factor_src_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_src_a <= 
                `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_src_a <= 
                default: 
            endcase
            case (func_dst_rgb)
                `ROP_BLEND_FACTOR_ZERO:                 factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_ONE:                  factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_SRC_RGB:              factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_DST_RGB:              factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_CONST_RGB:            factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_CONST_A:              factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_dst_rgb <= 
                `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_dst_rgb <= 
                default: 
            endcase
            case (func_dst_a)
                `ROP_BLEND_FACTOR_ZERO:                 factor_dst_a <= 
                `ROP_BLEND_FACTOR_ONE:                  factor_dst_a <= 
                `ROP_BLEND_FACTOR_SRC_RGB:              factor_dst_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_dst_a <= 
                `ROP_BLEND_FACTOR_DST_RGB:              factor_dst_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_dst_a <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_dst_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_a <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_dst_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_a <= 
                `ROP_BLEND_FACTOR_CONST_RGB:            factor_dst_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_dst_a <= 
                `ROP_BLEND_FACTOR_CONST_A:              factor_dst_a <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_dst_a <= 
                `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_dst_a <= 
                default: 
            endcase

            // Blend Equations
            // Need enumerated modes for which equation to use, add to vx_rop_define (these are the values for func_add)
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
        else begin  // if blending disabled, pass on values
            out_color <= source_color;
        end
    end

    VX_pipe_register #(
        .DATAW  (1 + `UUID_BITS + (`ROP_BLEND_FACTOR_BITS * 4)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (valid_in),
        .data_in  ({valid_in, uuid_in, factor_src_rgb, factor_rgb_a, factor_dst_rgb, factor_dst_a}),
        .data_out ({valid_out, uuid_out, pipe_factor_src_rgb, pipe_factor_rgb_a,
                                         pipe_factor_dst_rgb, pipe_factor_dst_a})
    );

    VX_rop_csr #(  
        .CORE_ID (CORE_ID)
    ) rop_csr (
        .clk            (clk),
        .reset          (reset),
        .rop_csr_if     (rop_csr_if),
        .rop_req_if     (rop_req_if),
        .rop_csrs       (reg_csrs)
    );

endmodule