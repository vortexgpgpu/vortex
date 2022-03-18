`ifndef VX_ROP_TYPES
`define VX_ROP_TYPES

`include "VX_define.vh"

package rop_types;

typedef struct packed {
    logic [7:0] a;
    logic [7:0] r;
    logic [7:0] g;
    logic [7:0] b;
} rgba_t;

typedef struct packed {
    logic [31:0]                        cbuf_addr;
    logic [31:0]                        cbuf_pitch;
    logic [31:0]                        cbuf_mask;

    logic [31:0]                        zbuf_addr;
    logic [31:0]                        zbuf_pitch;

    logic [`ROP_DEPTH_FUNC_BITS-1:0]    depth_func;
    logic                               depth_writemask;

    logic [`ROP_DEPTH_FUNC_BITS-1:0]    stencil_front_func;    
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_front_zpass;
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_front_zfail;
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_front_fail;
    logic [`ROP_STENCIL_BITS-1:0]       stencil_front_ref;
    logic [`ROP_STENCIL_BITS-1:0]       stencil_front_mask;    
    logic [`ROP_DEPTH_FUNC_BITS-1:0]    stencil_back_func;    
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_back_zpass;
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_back_zfail;
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_back_fail;
    logic [`ROP_STENCIL_BITS-1:0]       stencil_back_ref;
    logic [`ROP_STENCIL_BITS-1:0]       stencil_back_mask;
    logic [`ROP_STENCIL_BITS-1:0]       stencil_writemask;
    
    logic [`ROP_BLEND_MODE_BITS-1:0]    blend_mode_rgb;
    logic [`ROP_BLEND_MODE_BITS-1:0]    blend_mode_a;
    logic [`ROP_BLEND_FUNC_BITS-1:0]    blend_src_rgb;
    logic [`ROP_BLEND_FUNC_BITS-1:0]    blend_src_a;
    logic [`ROP_BLEND_FUNC_BITS-1:0]    blend_dst_rgb;
    logic [`ROP_BLEND_FUNC_BITS-1:0]    blend_dst_a;
    rgba_t                              blend_const;
    
    logic [`ROP_LOGIC_OP_BITS-1:0]      logic_op;
} rop_dcrs_t;

typedef struct packed {
    logic [1:0]     rt_idx;
    logic [15:0]    pos_x;
    logic [15:0]    pos_y;    
    logic [23:0]    depth;
    logic [2:0]     sample_idx;
    logic [7:0]     sample_mask;
} rop_csrs_t;

endpackage

`endif