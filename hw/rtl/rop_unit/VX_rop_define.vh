`ifndef VX_ROP_DEFINE
`define VX_ROP_DEFINE

`include "VX_define.vh"

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
        `CSR_ROP_BLEND_SRC:   dpi_trace("BLEND_SRC");
        `CSR_ROP_BLEND_DST:   dpi_trace("BLEND_DST");
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