`ifndef VX_TEX_TYPES_VH
`define VX_TEX_TYPES_VH

`include "VX_define.vh"

`define TEX_FXD_INT         (`TEX_FXD_BITS - `TEX_FXD_FRAC)
`define TEX_FXD_ONE         (2 ** `TEX_FXD_FRAC)
`define TEX_FXD_HALF        (`TEX_FXD_ONE >> 1)
`define TEX_FXD_MASK        (`TEX_FXD_ONE - 1)

`define TEX_ADDR_BITS       32
`define TEX_FORMAT_BITS     3
`define TEX_WRAP_BITS       2
`define TEX_FILTER_BITS     1
`define TEX_MIPOFF_BITS     (2*`TEX_DIM_BITS+1)

`define TEX_LGSTRIDE_MAX    2
`define TEX_LGSTRIDE_BITS   2

`define TEX_BLEND_FRAC      `TEX_SUBPIXEL_BITS
`define TEX_BLEND_ONE       (2 ** `TEX_BLEND_FRAC)

package VX_tex_types;

typedef struct packed {
    logic [(`TEX_LOD_MAX+1)-1:0][`TEX_MIPOFF_BITS-1:0] mipoff;
    logic [1:0][`TEX_LOD_BITS-1:0]  logdims;
    logic [1:0][`TEX_WRAP_BITS-1:0] wraps;
    logic [`TEX_ADDR_BITS-1:0]      baseaddr;
    logic [`TEX_FORMAT_BITS-1:0]    format;
    logic [`TEX_FILTER_BITS-1:0]    filter;
} tex_dcrs_t;

typedef struct packed {
    logic [`TEX_STAGE_BITS-1:0] stage;
} tex_csrs_t;

endpackage

`endif // VX_TEX_TYPES_VH
