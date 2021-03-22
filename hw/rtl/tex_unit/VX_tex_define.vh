`ifndef VX_TEX_DEFINE
`define VX_TEX_DEFINE

`include "VX_define.vh"

`define FIXED_FRAC      20
`define FIXED_INT       (32 - `FIXED_FRAC)
`define FIXED_ONE       (1 << `FIXED_FRAC)
`define FIXED_HALF      (`FIXED_ONE >> 1)
`define FIXED_MASK      (`FIXED_ONE - 1)

`define CLAMP(x,lo,hi)  ((x < lo) ? lo : ((x > hi) ? hi : x))

`define BLEND_FRAC_64    8

`define LERP_64(x1,x2,frac) ((x2 + (((x1 - x2) * frac) >> `BLEND_FRAC_64)) & 64'h00ff00ff00ff00ff)

`define TEX_ADDR_BITS    32
`define TEX_FORMAT_BITS  3
`define TEX_WRAP_BITS    2
`define TEX_WIDTH_BITS   4
`define TEX_HEIGHT_BITS  4
`define TEX_STRIDE_BITS  2
`define TEX_FILTER_BITS  1

`define TEX_WRAP_REPEAT  0
`define TEX_WRAP_CLAMP   1
`define TEX_WRAP_MIRROR  2

`define MAX_COLOR_WIDTH   8
`define NUM_COLOR_CHANNEL 4  

`define TEX_COLOR_BITS    8

`define TEX_FORMAT_R5G6B5       `TEX_FORMAT_BITS'(1)
`define TEX_FORMAT_R8G8B8       `TEX_FORMAT_BITS'(2)
`define TEX_FORMAT_R8G8B8A8     `TEX_FORMAT_BITS'(3)

`endif