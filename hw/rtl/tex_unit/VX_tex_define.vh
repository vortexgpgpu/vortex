`ifndef VX_TEX_DEFINE
`define VX_TEX_DEFINE

`include "VX_define.vh"

`define FIXED_BITS          32
`define FIXED_FRAC          20
`define FIXED_INT           (`FIXED_BITS - `FIXED_FRAC)
`define FIXED_ONE           (2 ** `FIXED_FRAC)
`define FIXED_HALF          (`FIXED_ONE >> 1)
`define FIXED_MASK          (`FIXED_ONE - 1)

`define TEX_ADDR_BITS       32
`define TEX_FORMAT_BITS     3
`define TEX_WRAP_BITS       2
`define TEX_DIM_BITS        4
`define TEX_FILTER_BITS     1

`define TEX_MIPOFF_BITS     (2*12+1)
`define TEX_STRIDE_BITS     2

`define TEX_LOD_BITS        4
`define TEX_MIP_BITS        (`NTEX_BITS + `TEX_LOD_BITS)

`define TEX_WRAP_CLAMP      0
`define TEX_WRAP_REPEAT     1
`define TEX_WRAP_MIRROR     2

`define BLEND_FRAC          8
`define BLEND_ONE           (2 ** `BLEND_FRAC)

`define TEX_FORMAT_R8G8B8A8 `TEX_FORMAT_BITS'(0)
`define TEX_FORMAT_R5G6B5   `TEX_FORMAT_BITS'(1)
`define TEX_FORMAT_R4G4B4A4 `TEX_FORMAT_BITS'(2)
`define TEX_FORMAT_L8A8     `TEX_FORMAT_BITS'(3)
`define TEX_FORMAT_L8       `TEX_FORMAT_BITS'(4)
`define TEX_FORMAT_A8       `TEX_FORMAT_BITS'(5)

`endif