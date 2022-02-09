`ifndef VX_ROP_TYPES
`define VX_ROP_TYPES

`include "VX_define.vh"

package rop_types;

typedef struct packed {
    logic [`ROP_BLEND_FACTOR_BITS-1:0] blend_src_rgb;
    logic [`ROP_BLEND_FACTOR_BITS-1:0] blend_dst_rgb;
    logic [`ROP_BLEND_FACTOR_BITS-1:0] blend_src_a;
    logic [`ROP_BLEND_FACTOR_BITS-1:0] blend_dst_a;
    logic [31:0] blend_const;
    logic [`ROP_LOGIC_OP_BITS-1:0] logic_op;
} rop_csrs_t;

endpackage

`endif