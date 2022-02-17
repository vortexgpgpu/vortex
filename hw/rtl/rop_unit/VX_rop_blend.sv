`include "VX_rop_define.vh"

module VX_rop_blend #(  
    parameter CORE_ID = 0
) (
    input wire                  clk,
    input wire                  reset,
    input wire                  enable,

    // Blend Equation (TODO: replace with CSRs)
    input wire [:0]             mode_rgb,
    input wire [:0]             mode_alpha,

    // Blend Function (TODO: replace with CSRs)
    input wire [`ROP_BLEND_FACTOR_BITS-1:0] func_src_rgb,
    input wire [`ROP_BLEND_FACTOR_BITS-1:0] func_dst_rgb,
    input wire [`ROP_BLEND_FACTOR_BITS-1:0] func_src_alpha,
    input wire [`ROP_BLEND_FACTOR_BITS-1:0] func_dst_alpha,

    // Blend Color
    input wire [:0]             const_red,
    input wire [:0]             const_green,
    input wire [:0]             const_blue,
    input wire [:0]             const_alpha,

    // Color Values (TODO: replace with REQs)
    input wire [31:0]           dst_color,
    input wire [31:0]           src_color,
    output wire [31:0]          out_color,

    VX_rop_csr_if.slave         rop_csr_if,
    VX_rop_req_if.slave         rop_req_src_if,
    VX_rop_req_if.slave         rop_req_dst_if,
    VX_rop_req_if.master        rop_req_out_if
);

    rop_csrs_t reg_csrs;

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
            case (func_src_alpha)
                `ROP_BLEND_FACTOR_ZERO:                 factor_src_alpha <= 
                `ROP_BLEND_FACTOR_ONE:                  factor_src_alpha <= 
                `ROP_BLEND_FACTOR_SRC_RGB:              factor_src_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_src_alpha <= 
                `ROP_BLEND_FACTOR_DST_RGB:              factor_src_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_src_alpha <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_src_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_alpha <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_src_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_src_alpha <= 
                `ROP_BLEND_FACTOR_CONST_RGB:            factor_src_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_src_alpha <= 
                `ROP_BLEND_FACTOR_CONST_A:              factor_src_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_src_alpha <= 
                `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_src_alpha <= 
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
            case (func_dst_alpha)
                `ROP_BLEND_FACTOR_ZERO:                 factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_ONE:                  factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_SRC_RGB:              factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB:    factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_DST_RGB:              factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB:    factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_DST_A:                factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_DST_A:      factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_CONST_RGB:            factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB:  factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_CONST_A:              factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_ONE_MINUS_CONST_A:    factor_dst_alpha <= 
                `ROP_BLEND_FACTOR_ALPHA_SAT:            factor_dst_alpha <= 
                default: 
            endcase

            // Blend Equations
            case (mode_rgb)
                `ROP_BLEND_EQ_x: begin
                    out_red <=
                    out_blue <=
                    out_green <=
                end 
                default: 
            endcase
            case (mode_alpha)
                `ROP_BLEND_EQ_x: begin
                    out_alpha <=
                end 
                default: 
            endcase
        end
        else begin  // if blending disabled, pass on values
            out_color <= source_color;
        end
    end

    VX_pipe_register #(
        .DATAW  (1 + `UUID_BITS + ),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (valid_in),
        .data_in  ({valid_in, uuid_in, }),
        .data_out ({valid_out, uuid_out, })
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