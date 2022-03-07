`include "VX_rop_define.vh"

module VX_rop_blend #(
    //--
) (
    input wire          clk,
    input wire          reset,   

    // Handshake
    output wire         ready_in,
    input wire          valid_in,
    input wire          ready_out,
    output wire         valid_out,

    // DCRs
    input rop_dcrs_t    dcrs,

    // Inputs    
    input rgba_t        src_color,
    input rgba_t        dst_color,
    
    // Outputs
    output rgba_t       color_out
);
    `UNUSED_VAR (dcrs)

    wire stall = ~ready_out && valid_out;
    
    assign ready_in = ~stall;
    
    rgba_t src_factor, dst_factor;

    VX_rop_blend_func #(
    ) src_blend_func (
        .func_rgb   (dcrs.blend_src_rgb),
        .func_a     (dcrs.blend_src_a),
        .src_color  (src_color),
        .dst_color  (dst_color),
        .cst_color  (dcrs.blend_const),
        .factor_out (src_factor)
    );

    VX_rop_blend_func #(
    ) dst_blend_func (
        .func_rgb   (dcrs.blend_dst_rgb),
        .func_a     (dcrs.blend_dst_a),
        .src_color  (src_color),
        .dst_color  (dst_color),
        .cst_color  (dcrs.blend_const),
        .factor_out (dst_factor)
    );

    wire valid_s1;
    rgba_t src_color_s1, dst_color_s1, src_factor_s1, dst_factor_s1;

    VX_pipe_register #(
        .DATAW  (1 + 32 * 4),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_in, src_color,    dst_color,    src_factor,    dst_factor}),
        .data_out ({valid_s1, src_color_s1, dst_color_s1, src_factor_s1, dst_factor_s1})
    );

    rgba_t mult_add_color_out, min_color_out, max_color_out, logic_op_color_out;
    
    VX_rop_blend_multadd #(
    ) mult_add (
        .mode_rgb   (dcrs.blend_mode_rgb),
        .mode_a     (dcrs.blend_mode_a),
        .src_color  (src_color_s1),
        .dst_color  (dst_color_s1),
        .src_factor (src_factor_s1),
        .dst_factor (dst_factor_s1),
        .color_out  (mult_add_color_out)
    );

    VX_rop_blend_minmax #(
    ) min_max (
        .src_color (src_color_s1),
        .dst_color (dst_color_s1),
        .min_out   (min_color_out),
        .max_out   (max_color_out)
    );

    VX_rop_logic_op #(
    ) logic_op (
        .op        (dcrs.logic_op),
        .src_color (src_color_s1),
        .dst_color (dst_color_s1),
        .color_out (logic_op_color_out)
    );

    rgba_t color_out_s1;

    always @(*) begin
        // RGB Component
        case (dcrs.blend_mode_rgb)
            `ROP_BLEND_MODE_ADD, 
            `ROP_BLEND_MODE_SUB, 
            `ROP_BLEND_MODE_REV_SUB: begin
                color_out_s1.r = mult_add_color_out.r;
                color_out_s1.g = mult_add_color_out.g;
                color_out_s1.b = mult_add_color_out.b;
                end
            `ROP_BLEND_MODE_MIN: begin
                color_out_s1.r = min_color_out.r;
                color_out_s1.g = min_color_out.g;
                color_out_s1.b = min_color_out.b;
                end
            `ROP_BLEND_MODE_MAX: begin
                color_out_s1.r = max_color_out.r;
                color_out_s1.g = max_color_out.g;
                color_out_s1.b = max_color_out.b;
                end
            `ROP_BLEND_MODE_LOGICOP: begin
                color_out_s1.r = logic_op_color_out.r;
                color_out_s1.g = logic_op_color_out.g;
                color_out_s1.b = logic_op_color_out.b;
                end
            default: begin
                color_out_s1.r = 'x;
                color_out_s1.g = 'x;
                color_out_s1.b = 'x;
                end
        endcase
        // Alpha Component
        case (dcrs.blend_mode_a)
            `ROP_BLEND_MODE_ADD, 
            `ROP_BLEND_MODE_SUB, 
            `ROP_BLEND_MODE_REV_SUB: begin
                color_out_s1.a = mult_add_color_out.a;
                end
            `ROP_BLEND_MODE_MIN: begin
                color_out_s1.a = min_color_out.a;
                end
            `ROP_BLEND_MODE_MAX: begin
                color_out_s1.a = max_color_out.a;
                end
            `ROP_BLEND_MODE_LOGICOP: begin
                color_out_s1.a = logic_op_color_out.a;
                end
            default: begin
                color_out_s1.a = 'x;
                end
        endcase
    end

    VX_pipe_register #(
        .DATAW  (1 + 32),
        .RESETW (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_s1,  color_out_s1}),
        .data_out ({valid_out, color_out})
    );

endmodule