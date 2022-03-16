`include "VX_rop_define.vh"

module VX_rop_blend #(
    parameter CLUSTER_ID = 0,
    parameter NUM_LANES  = 4
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
    input rgba_t [NUM_LANES-1:0] src_color,
    input rgba_t [NUM_LANES-1:0] dst_color,
    
    // Outputs
    output rgba_t [NUM_LANES-1:0] color_out
);
    `UNUSED_VAR (dcrs)

    wire stall = ~ready_out && valid_out;
    
    assign ready_in = ~stall;
    
    rgba_t [NUM_LANES-1:0]  src_factor;
    rgba_t [NUM_LANES-1:0]  dst_factor;

    generate
        for(genvar i = 0; i < NUM_LANES; i++) begin : blend_func_inst

            VX_rop_blend_func #(
            ) src_blend_func (
                .func_rgb   (dcrs.blend_src_rgb),
                .func_a     (dcrs.blend_src_a),
                .src_color  (src_color[i]),
                .dst_color  (dst_color[i]),
                .cst_color  (dcrs.blend_const),
                .factor_out (src_factor[i])
            );

            VX_rop_blend_func #(
            ) dst_blend_func (
                .func_rgb   (dcrs.blend_dst_rgb),
                .func_a     (dcrs.blend_dst_a),
                .src_color  (src_color[i]),
                .dst_color  (dst_color[i]),
                .cst_color  (dcrs.blend_const),
                .factor_out (dst_factor[i])
            );
        end
    endgenerate

    wire valid_s1;

    rgba_t [NUM_LANES-1:0] src_color_s1;
    rgba_t [NUM_LANES-1:0] dst_color_s1;
    rgba_t [NUM_LANES-1:0] src_factor_s1;
    rgba_t [NUM_LANES-1:0] dst_factor_s1;

    VX_pipe_register #(
        .DATAW  (1 + 32 * 4 * NUM_LANES),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_in, src_color,    dst_color,    src_factor,    dst_factor}),
        .data_out ({valid_s1, src_color_s1, dst_color_s1, src_factor_s1, dst_factor_s1})
    );

    rgba_t [NUM_LANES-1:0] mult_add_color_out;
    rgba_t [NUM_LANES-1:0] min_color_out;
    rgba_t [NUM_LANES-1:0] max_color_out;
    rgba_t [NUM_LANES-1:0] logic_op_color_out;
    
    generate
        for(genvar i = 0; i < NUM_LANES; i++) begin : blend_op_inst

            VX_rop_blend_multadd #(
            ) mult_add (
                .mode_rgb   (dcrs.blend_mode_rgb),
                .mode_a     (dcrs.blend_mode_a),
                .src_color  (src_color_s1[i]),
                .dst_color  (dst_color_s1[i]),
                .src_factor (src_factor_s1[i]),
                .dst_factor (dst_factor_s1[i]),
                .color_out  (mult_add_color_out[i])
            );

            VX_rop_blend_minmax #(
            ) min_max (
                .src_color (src_color_s1[i]),
                .dst_color (dst_color_s1[i]),
                .min_out   (min_color_out[i]),
                .max_out   (max_color_out[i])
            );

            VX_rop_logic_op #(
            ) logic_op (
                .op        (dcrs.logic_op),
                .src_color (src_color_s1[i]),
                .dst_color (dst_color_s1[i]),
                .color_out (logic_op_color_out[i])
            );
        end
    endgenerate

    rgba_t [NUM_LANES-1:0] color_out_s1;

    generate
        for(genvar i = 0; i < NUM_LANES; i++) begin : blend_color_out_inst
            always @(*) begin
                // RGB Component
                case (dcrs.blend_mode_rgb)
                    `ROP_BLEND_MODE_ADD, 
                    `ROP_BLEND_MODE_SUB, 
                    `ROP_BLEND_MODE_REV_SUB: begin
                        color_out_s1[i].r = mult_add_color_out[i].r;
                        color_out_s1[i].g = mult_add_color_out[i].g;
                        color_out_s1[i].b = mult_add_color_out[i].b;
                        end
                    `ROP_BLEND_MODE_MIN: begin
                        color_out_s1[i].r = min_color_out[i].r;
                        color_out_s1[i].g = min_color_out[i].g;
                        color_out_s1[i].b = min_color_out[i].b;
                        end
                    `ROP_BLEND_MODE_MAX: begin
                        color_out_s1[i].r = max_color_out[i].r;
                        color_out_s1[i].g = max_color_out[i].g;
                        color_out_s1[i].b = max_color_out[i].b;
                        end
                    `ROP_BLEND_MODE_LOGICOP: begin
                        color_out_s1[i].r = logic_op_color_out[i].r;
                        color_out_s1[i].g = logic_op_color_out[i].g;
                        color_out_s1[i].b = logic_op_color_out[i].b;
                        end
                    default: begin
                        color_out_s1[i].r = 'x;
                        color_out_s1[i].g = 'x;
                        color_out_s1[i].b = 'x;
                        end
                endcase
                // Alpha Component
                case (dcrs.blend_mode_a)
                    `ROP_BLEND_MODE_ADD, 
                    `ROP_BLEND_MODE_SUB, 
                    `ROP_BLEND_MODE_REV_SUB: begin
                        color_out_s1[i].a = mult_add_color_out[i].a;
                        end
                    `ROP_BLEND_MODE_MIN: begin
                        color_out_s1[i].a = min_color_out[i].a;
                        end
                    `ROP_BLEND_MODE_MAX: begin
                        color_out_s1[i].a = max_color_out[i].a;
                        end
                    `ROP_BLEND_MODE_LOGICOP: begin
                        color_out_s1[i].a = logic_op_color_out[i].a;
                        end
                    default: begin
                        color_out_s1[i].a = 'x;
                        end
                endcase
            end
        end
    endgenerate

    VX_pipe_register #(
        .DATAW  (1 + 32 * NUM_LANES),
        .RESETW (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_s1,  color_out_s1}),
        .data_out ({valid_out, color_out})
    );

endmodule