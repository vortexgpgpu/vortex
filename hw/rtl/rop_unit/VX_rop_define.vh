`ifndef VX_ROP_DEFINE
`define VX_ROP_DEFINE

`include "VX_define.vh"

`define ROP_BLEND_FACTOR_ZERO                   0 
`define ROP_BLEND_FACTOR_ONE                    1
`define ROP_BLEND_FACTOR_SRC_RGB                2
`define ROP_BLEND_FACTOR_ONE_MINUS_SRC_RGB      3
`define ROP_BLEND_FACTOR_DST_RGB                4
`define ROP_BLEND_FACTOR_ONE_MINUS_DST_RGB      5
`define ROP_BLEND_FACTOR_DST_A                  6
`define ROP_BLEND_FACTOR_ONE_MINUS_DST_A        7
`define ROP_BLEND_FACTOR_DST_A                  8
`define ROP_BLEND_FACTOR_ONE_MINUS_DST_A        9
`define ROP_BLEND_FACTOR_CONST_RGB              10
`define ROP_BLEND_FACTOR_ONE_MINUS_CONST_RGB    11
`define ROP_BLEND_FACTOR_CONST_A                12
`define ROP_BLEND_FACTOR_ONE_MINUS_CONST_A      13
`define ROP_BLEND_FACTOR_ALPHA_SAT              14
`define ROP_BLEND_FACTOR_BITS                   4

`define ROP_LOGIC_OP_CLEAR                      0
`define ROP_LOGIC_OP_AND                        1
`define ROP_LOGIC_OP_AND REVERSE                2
`define ROP_LOGIC_OP_COPY                       3
`define ROP_LOGIC_OP_AND INVERTED               4
`define ROP_LOGIC_OP_NOOP                       5
`define ROP_LOGIC_OP_XOR                        6
`define ROP_LOGIC_OP_OR                         7
`define ROP_LOGIC_OP_NOR                        8
`define ROP_LOGIC_OP_EQUIV                      9
`define ROP_LOGIC_OP_INVERT                     10
`define ROP_LOGIC_OP_OR REVERSE                 11
`define ROP_LOGIC_OP_COPY INVERTED              12
`define ROP_LOGIC_OP_OR INVERTED                13
`define ROP_LOGIC_OP_NAND                       14
`define ROP_LOGIC_OP_SET                        15
`define ROP_LOGIC_OP_BITS                       4

`include "VX_rop_types.vh"

`IGNORE_WARNINGS_BEGIN
import rop_types::*;
`IGNORE_WARNINGS_END

`endif