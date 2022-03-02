`ifndef VX_ROP_DEFINE
`define VX_ROP_DEFINE

`include "VX_define.vh"

task trace_rop_state (
    input [`DCR_ADDR_BITS-1:0] state
);
    case (state)
        `DCR_ROP_ZBUF_ADDR:   dpi_trace("ZBUF_ADDR");     
        `DCR_ROP_ZBUF_PITCH:  dpi_trace("ZBUF_PITCH");
        `DCR_ROP_CBUF_ADDR:   dpi_trace("CBUF_ADDR");
        `DCR_ROP_CBUF_PITCH:  dpi_trace("CBUF_PITCH");
        `DCR_ROP_ZFUNC:       dpi_trace("ZFUNC");
        `DCR_ROP_SFUNC:       dpi_trace("SFUNC");
        `DCR_ROP_ZPASS:       dpi_trace("ZPASS");
        `DCR_ROP_ZPASS:       dpi_trace("ZPASS");
        `DCR_ROP_ZFAIL:       dpi_trace("ZFAIL");
        `DCR_ROP_SFAIL:       dpi_trace("SFAIL");
        `DCR_ROP_BLEND_MODE:  dpi_trace("BLEND_MODE");
        `DCR_ROP_BLEND_SRC:   dpi_trace("BLEND_SRC");
        `DCR_ROP_BLEND_DST:   dpi_trace("BLEND_DST");
        `DCR_ROP_BLEND_CONST: dpi_trace("BLEND_CONST");
        `DCR_ROP_LOGIC_OP:    dpi_trace("LOGIC_OP");
        default:              dpi_trace("??");
    endcase  
endtask

`include "VX_rop_types.vh"

`IGNORE_WARNINGS_BEGIN
import rop_types::*;
`IGNORE_WARNINGS_END

`endif