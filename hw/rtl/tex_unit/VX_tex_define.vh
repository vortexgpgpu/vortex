`ifndef VX_TEX_DEFINE
`define VX_TEX_DEFINE

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

`define TEX_WRAP_CLAMP      0
`define TEX_WRAP_REPEAT     1
`define TEX_WRAP_MIRROR     2

`define TEX_BLEND_FRAC      8
`define TEX_BLEND_ONE       (2 ** `TEX_BLEND_FRAC)

`define TEX_FORMAT_R8G8B8A8 `TEX_FORMAT_BITS'(0)
`define TEX_FORMAT_R5G6B5   `TEX_FORMAT_BITS'(1)
`define TEX_FORMAT_R4G4B4A4 `TEX_FORMAT_BITS'(2)
`define TEX_FORMAT_L8A8     `TEX_FORMAT_BITS'(3)
`define TEX_FORMAT_L8       `TEX_FORMAT_BITS'(4)
`define TEX_FORMAT_A8       `TEX_FORMAT_BITS'(5)

`endif