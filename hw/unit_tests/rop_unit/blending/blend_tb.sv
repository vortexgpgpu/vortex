`timescale 1ns/1ns

`include "VX_rop_blend.sv"

`define check(x, y) if ((x == y) !== 1) if ((x == y) === 0) $error("x=%h, expected=%h", x, y); else $warning("x=%h, expected=%h", x, y)

module testbench();
    reg clk;
    reg reset;
    // reg valid_in;
    // reg ready_out;
    rop_dcrs_t dcrs;
    rgba_t src_color;
    rgba_t dst_color;
    rgba_t color_out;
    rgba_t src_factor;
    rgba_t dst_factor;
    
    VX_rop_blend #() dut (
        .clk        (clk),
        .reset      (reset),        
        .valid_in   (1), // only one needed
        .ready_in   (),
        .ready_out  (0), // only one needed
        .valid_out  (),
        .dcrs       (dcrs),
        .src_color  (src_color),
        .dst_color  (dst_color),
        .color_out  (color_out)
    );

    always begin
        #1 clk = !clk;
    end

    initial begin
        $monitor ("%d: clk=%b rst=%b mode_rgb=%h mode_a=%h src_color=%p, dst_color=%p, src_factor=%p, dst_factor=%p, out_color=%p", 
                  $time, clk, reset, dcrs.blend_mode_rgb, dcrs.blend_mode_a, src_color, dst_color, src_factor, dst_factor, out_color);
        #0 clk=0; reset=1; dcrs.blend_mode_rgb=0; dcrs.blend_mode_a=0; src_color=0; dst_color=0; src_factor=0; dst_factor=0; drcs.logic_op=0;
        
        #2 reset=0; dcrs.blend_src_rgb=`ROP_BLEND_FUNC_ONE; dcrs.blend_src_a=`ROP_BLEND_FUNC_ONE;
           dcrs.blend_dst_rgb=`ROP_BLEND_FUNC_ZERO; dcrs.blend_dst_a=`ROP_BLEND_FUNC_ZERO;
           dcrs.blend_mode_rgb=`ROP_BLEND_MODE_ADD; dcrs.blend_mode_a=`ROP_BLEND_MODE_SUB; 
           drcs.blend_const=32'h0;
           src_color='{8'hb4, 8'hef, 8'h4b, 8'h7b}; dst_color='{8'hc2, 8'hc4, 8'h26, 8'hf5}; 
        #2 `check(color_out, '{8'hb4, 8'hef, 8'h4b, 8'h7b});

        #2 dcrs.blend_src_rgb=`ROP_BLEND_FUNC_SRC_RGB; dcrs.blend_src_a=`ROP_BLEND_FUNC_SRC_A;
           dcrs.blend_dst_rgb=`ROP_BLEND_FUNC_SRC_RGB; dcrs.blend_dst_a=`ROP_BLEND_FUNC_SRC_A; 
        #2 `check(color_out, '{8'h2b, 8'hff, 8'hc7, 8'hb0});

        #2 dcrs.blend_mode_rgb=`ROP_BLEND_MODE_LOGICOP; dcrs.blend_mode_a=`ROP_BLEND_MODE_LOGICOP;
           dcrs.logic_op=`ROP_LOGIC_OP_AND_INVERTED;
        #2 `check(color_out, '{8'h42, 8'h00, 8'h24, 8'h84});
        #1 $finish; 
    end

endmodule