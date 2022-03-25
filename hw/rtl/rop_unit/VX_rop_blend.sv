`include "VX_rop_define.vh"

module VX_rop_blend #(
    parameter CLUSTER_ID = 0,
    parameter NUM_LANES  = 4,
    parameter TAG_WIDTH  = 1
) (
    input wire  clk,
    input wire  reset,   

    // Handshake
    input wire                  valid_in,
    input wire [TAG_WIDTH-1:0]  tag_in,
    output wire                 ready_in,    
    
    output wire                 valid_out,
    output wire [TAG_WIDTH-1:0] tag_out,
    input wire                  ready_out,

    // Configuration states
    input wire [`ROP_BLEND_MODE_BITS-1:0] blend_mode_rgb,
    input wire [`ROP_BLEND_MODE_BITS-1:0] blend_mode_a,
    input wire [`ROP_BLEND_FUNC_BITS-1:0] blend_src_rgb,
    input wire [`ROP_BLEND_FUNC_BITS-1:0] blend_src_a,
    input wire [`ROP_BLEND_FUNC_BITS-1:0] blend_dst_rgb,
    input wire [`ROP_BLEND_FUNC_BITS-1:0] blend_dst_a,
    input wire rgba_t                     blend_const,
    input wire [`ROP_LOGIC_OP_BITS-1:0]   logic_op,

    // Input values 
    input rgba_t [NUM_LANES-1:0] src_color,
    input rgba_t [NUM_LANES-1:0] dst_color,
    
    // Output values
    output rgba_t [NUM_LANES-1:0] color_out
);
    wire stall = ~ready_out && valid_out;
    
    assign ready_in = ~stall;
    
    rgba_t [NUM_LANES-1:0]  src_factor;
    rgba_t [NUM_LANES-1:0]  dst_factor;

    for (genvar i = 0; i < NUM_LANES; i++) begin : blend_func_inst
        VX_rop_blend_func #(
        ) rop_blend_func_src (
            .func_rgb   (blend_src_rgb),
            .func_a     (blend_src_a),
            .src_color  (src_color[i]),
            .dst_color  (dst_color[i]),
            .cst_color  (blend_const),
            .factor_out (src_factor[i])
        );

        VX_rop_blend_func #(
        ) rop_blend_func_dst (
            .func_rgb   (blend_dst_rgb),
            .func_a     (blend_dst_a),
            .src_color  (src_color[i]),
            .dst_color  (dst_color[i]),
            .cst_color  (blend_const),
            .factor_out (dst_factor[i])
        );
    end

    wire                 valid_s1;
    wire [TAG_WIDTH-1:0] tag_s1;

    rgba_t [NUM_LANES-1:0] src_color_s1;
    rgba_t [NUM_LANES-1:0] dst_color_s1;
    rgba_t [NUM_LANES-1:0] src_factor_s1;
    rgba_t [NUM_LANES-1:0] dst_factor_s1;

    VX_pipe_register #(
        .DATAW  (1 + TAG_WIDTH + 32 * 4 * NUM_LANES),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_in, tag_in, src_color,    dst_color,    src_factor,    dst_factor}),
        .data_out ({valid_s1, tag_s1, src_color_s1, dst_color_s1, src_factor_s1, dst_factor_s1})
    );

    rgba_t [NUM_LANES-1:0] mult_add_color_out;
    rgba_t [NUM_LANES-1:0] min_color_out;
    rgba_t [NUM_LANES-1:0] max_color_out;
    rgba_t [NUM_LANES-1:0] logic_op_color_out;
    
    for (genvar i = 0; i < NUM_LANES; i++) begin : blend_op_inst
        VX_rop_blend_multadd #(
        ) rop_blend_multadd (
            .mode_rgb   (blend_mode_rgb),
            .mode_a     (blend_mode_a),
            .src_color  (src_color_s1[i]),
            .dst_color  (dst_color_s1[i]),
            .src_factor (src_factor_s1[i]),
            .dst_factor (dst_factor_s1[i]),
            .color_out  (mult_add_color_out[i])
        );

        VX_rop_blend_minmax #(
        ) rop_blend_minmax (
            .src_color (src_color_s1[i]),
            .dst_color (dst_color_s1[i]),
            .min_out   (min_color_out[i]),
            .max_out   (max_color_out[i])
        );

        VX_rop_logic_op #(
        ) rop_logic_op (
            .op        (logic_op),
            .src_color (src_color_s1[i]),
            .dst_color (dst_color_s1[i]),
            .color_out (logic_op_color_out[i])
        );
    end

    rgba_t [NUM_LANES-1:0] color_out_s1;

    for (genvar i = 0; i < NUM_LANES; i++) begin : blend_color_out_inst
        always @(*) begin
            // RGB Component
            case (blend_mode_rgb)
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
            case (blend_mode_a)
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

    VX_pipe_register #(
        .DATAW  (1 + TAG_WIDTH + 32 * NUM_LANES),
        .RESETW (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({valid_s1,  tag_s1,  color_out_s1}),
        .data_out ({valid_out, tag_out, color_out})
    );

endmodule
