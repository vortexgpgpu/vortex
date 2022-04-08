`ifndef VX_TEX_DEFINE
`define VX_TEX_DEFINE

`include "VX_define.vh"

`define TEX_STAGE_BITS      `LOG2UP(`TEX_STAGE_COUNT)

`define TCACHE_TAG_SEL_BITS  2
`define TCACHE_TAG_WIDTH    (`UUID_BITS + `TCACHE_TAG_SEL_BITS)
`define TCACHE_NUM_REQS     `NUM_THREADS
`define TCACHE_WORD_SIZE    4

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

`endif
