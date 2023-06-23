`ifndef VX_ROP_TYPES_VH
`define VX_ROP_TYPES_VH

`include "VX_define.vh"

`ifdef XLEN_64
`define ROP_ADDR_BITS 32
`else
`define ROP_ADDR_BITS 25
`endif

package VX_rop_types;

typedef struct packed {
    logic [7:0] a;
    logic [7:0] r;
    logic [7:0] g;
    logic [7:0] b;
} rgba_t;

typedef struct packed {
    logic [`ROP_ADDR_BITS-1:0]          cbuf_addr;
    logic [`ROP_PITCH_BITS-1:0]         cbuf_pitch;
    logic [3:0]                         cbuf_writemask;

    logic [`ROP_ADDR_BITS-1:0]          zbuf_addr;
    logic [`ROP_PITCH_BITS-1:0]         zbuf_pitch;

    logic                               depth_enable;
    logic [`ROP_DEPTH_FUNC_BITS-1:0]    depth_func;
    logic                               depth_writemask;
    
    logic [1:0]                         stencil_enable;
    logic [1:0][`ROP_DEPTH_FUNC_BITS-1:0] stencil_func;    
    logic [1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_zpass;
    logic [1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_zfail;
    logic [1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_fail;
    logic [1:0][`ROP_STENCIL_BITS-1:0]  stencil_ref;
    logic [1:0][`ROP_STENCIL_BITS-1:0]  stencil_mask;    
    logic [1:0][`ROP_STENCIL_BITS-1:0]  stencil_writemask;
    
    logic                               blend_enable;
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
    logic [1:0]                 rt_idx;
    logic [`ROP_DIM_BITS-1:0]   pos_x;
    logic [`ROP_DIM_BITS-1:0]   pos_y;    
    logic [23:0]                depth;
    logic [2:0]                 sample_idx;
    logic [7:0]                 sample_mask;
} rop_csrs_t;

endpackage

`endif // VX_ROP_TYPES_VH
