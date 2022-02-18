`ifndef VX_ROP_DEFINE
`define VX_ROP_DEFINE

`include "VX_define.vh"

`define ROP_DEPTH_FUNC_NEVER                    0
`define ROP_DEPTH_FUNC_ALWAYS                   1
`define ROP_DEPTH_FUNC_LESS                     2
`define ROP_DEPTH_FUNC_LEQUAL                   3
`define ROP_DEPTH_FUNC_EQUAL                    4
`define ROP_DEPTH_FUNC_GEQUAL                   5
`define ROP_DEPTH_FUNC_GREATER                  6
`define ROP_DEPTH_FUNC_NOTEQUAL                 7
`define ROP_DEPTH_FUNC_BITS                     3

`define ROP_STENCIL_OP_KEEP                     0 
`define ROP_STENCIL_OP_NEVER                    1
`define ROP_STENCIL_OP_REPLACE                  2
`define ROP_STENCIL_OP_INCR                     3
`define ROP_STENCIL_OP_DECR                     4
`define ROP_STENCIL_OP_INVERT                   5
`define ROP_STENCIL_OP_INCR_WRAP                6
`define ROP_STENCIL_OP_DECR_WRAP                7
`define ROP_STENCIL_OP_BITS                     3

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

`define ROP_BLEND_MODE_FUNC_ADD                 0
`define ROP_BLEND_MODE_FUNC_SUBTRACT            1
`define ROP_BLEND_MODE_FUNC_REVERSE_SUBTRACT    2
`define ROP_BLEND_MODE_MIN                      3
`define ROP_BLEND_MODE_MAX                      4
`define ROP_BLEND_MODE_LOGIC_OP                 5
`define ROP_BLEND_MODE_BITS                     3

`define ROP_LOGIC_OP_CLEAR                      0
`define ROP_LOGIC_OP_AND                        1
`define ROP_LOGIC_OP_AND_REVERSE                2
`define ROP_LOGIC_OP_COPY                       3
`define ROP_LOGIC_OP_AND_INVERTED               4
`define ROP_LOGIC_OP_NOOP                       5
`define ROP_LOGIC_OP_XOR                        6
`define ROP_LOGIC_OP_OR                         7
`define ROP_LOGIC_OP_NOR                        8
`define ROP_LOGIC_OP_EQUIV                      9
`define ROP_LOGIC_OP_INVERT                     10
`define ROP_LOGIC_OP_OR_REVERSE                 11
`define ROP_LOGIC_OP_COPY_INVERTED              12
`define ROP_LOGIC_OP_OR_INVERTED                13
`define ROP_LOGIC_OP_NAND                       14
`define ROP_LOGIC_OP_SET                        15
`define ROP_LOGIC_OP_BITS                       4

task trace_rop_state (
    input [`CSR_ADDR_BITS-1:0] state
);
    case (state)
        `CSR_ROP_ZBUF_ADDR:   dpi_trace("ZBUF_ADDR");     
        `CSR_ROP_ZBUF_PITCH:  dpi_trace("ZBUF_PITCH");
        `CSR_ROP_CBUF_ADDR:   dpi_trace("CBUF_ADDR");
        `CSR_ROP_CBUF_PITCH:  dpi_trace("CBUF_PITCH");
        `CSR_ROP_ZFUNC:       dpi_trace("ZFUNC");
        `CSR_ROP_SFUNC:       dpi_trace("SFUNC");
        `CSR_ROP_ZPASS:       dpi_trace("ZPASS");
        `CSR_ROP_ZPASS:       dpi_trace("ZPASS");
        `CSR_ROP_ZFAIL:       dpi_trace("ZFAIL");
        `CSR_ROP_SFAIL:       dpi_trace("SFAIL");
        `CSR_ROP_BLEND_MODE:  dpi_trace("BLEND_MODE");
        `CSR_ROP_BLEND_RGB:   dpi_trace("BLEND_RGB");
        `CSR_ROP_BLEND_APLHA: dpi_trace("BLEND_ALPHA");
        `CSR_ROP_BLEND_CONST: dpi_trace("BLEND_CONST");
        `CSR_ROP_LOGIC_OP:    dpi_trace("LOGIC_OP");
        default:              dpi_trace("??");
    endcase  
endtask

`include "VX_rop_types.vh"

`IGNORE_WARNINGS_BEGIN
import rop_types::*;
`IGNORE_WARNINGS_END

`endif