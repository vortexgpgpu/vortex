`ifndef VX_ROP_TYPES
`define VX_ROP_TYPES

`include "VX_define.vh"

package rop_types;

typedef struct packed {
    logic [31:0]                       zbuf_addr;
    logic [31:0]                       zbuf_pitch;
    logic [31:0]                       cbuf_addr;
    logic [31:0]                       cbuf_pitch;    
    logic [`ROP_DEPTH_FUNC_BITS-1:0]   zfunc;
    logic [`ROP_DEPTH_FUNC_BITS-1:0]   sfunc;    
    logic [`ROP_STENCIL_OP_BITS-1:0]   zfail;
    logic [`ROP_STENCIL_OP_BITS-1:0]   zpass;
    logic [`ROP_STENCIL_OP_BITS-1:0]   sfail;
    logic [`ROP_BLEND_FUNC_BITS-1:0]   blend_func_src_rgb;
    logic [`ROP_BLEND_FUNC_BITS-1:0]   blend_func_dst_rgb;
    logic [`ROP_BLEND_FUNC_BITS-1:0]   blend_func_src_a;
    logic [`ROP_BLEND_FUNC_BITS-1:0]   blend_func_dst_a;
    logic [`ROP_BLEND_MODE_BITS-1:0]   blend_mode_rgb;
    logic [`ROP_BLEND_MODE_BITS-1:0]   blend_mode_a;
    logic [31:0] blend_const;
    logic [`ROP_LOGIC_OP_BITS-1:0] logic_op;
} rop_csrs_t;

endpackage

`endif