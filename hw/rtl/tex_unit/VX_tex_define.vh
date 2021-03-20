`ifndef VX_TEX_DEFINE
`define VX_TEX_DEFINE

`include "VX_define.vh"

`define FIXED_FRAC      20
`define FIXED_INT       (32 - `FIXED_FRAC)
`define FIXED_ONE       (1 << `FIXED_FRAC)
`define FIXED_MASK      (`FIXED_ONE - 1)

`define CLAMP(x,lo,hi)  ((x < lo) ? lo : ((x > hi) ? hi : x))

`define TEX_ADDR_BITS    32
`define TEX_FORMAT_BITS  3
`define TEX_WRAP_BITS    2
`define TEX_WIDTH_BITS   12
`define TEX_HEIGHT_BITS  12
`define TEX_STRIDE_BITS  2
`define TEX_FILTER_BITS  1

`define TEX_WRAP_REPEAT  0
`define TEX_WRAP_CLAMP   1
`define TEX_WRAP_MIRROR  2

`endif