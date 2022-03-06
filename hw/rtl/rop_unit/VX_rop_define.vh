`ifndef VX_ROP_DEFINE
`define VX_ROP_DEFINE

`include "VX_define.vh"

task trace_rop_state (
    input [`DCR_ADDR_BITS-1:0] state
);
    case (state)
        `DCR_ROP_CBUF_ADDR:     dpi_trace("CBUF_ADDR");
        `DCR_ROP_CBUF_PITCH:    dpi_trace("CBUF_PITCH");
        `DCR_ROP_CBUF_MASK:     dpi_trace("CBUF_MASK");
        `DCR_ROP_ZBUF_ADDR:     dpi_trace("ZBUF_ADDR");
        `DCR_ROP_ZBUF_PITCH:    dpi_trace("ZBUF_PITCH");
        `DCR_ROP_DEPTH_FUNC:    dpi_trace("DEPTH_FUNC");
        `DCR_ROP_DEPTH_MASK:    dpi_trace("DEPTH_MASK");
        `DCR_ROP_STENCIL_FUNC:  dpi_trace("STENCIL_FUNC");        
        `DCR_ROP_STENCIL_ZPASS: dpi_trace("STENCIL_ZPASS");
        `DCR_ROP_STENCIL_ZFAIL: dpi_trace("STENCIL_ZFAIL");
        `DCR_ROP_STENCIL_FAIL:  dpi_trace("STENCIL_FAIL");
        `DCR_ROP_STENCIL_MASK:  dpi_trace("STENCIL_MASK");
        `DCR_ROP_STENCIL_REF:   dpi_trace("STENCIL_REF");
        `DCR_ROP_BLEND_MODE:    dpi_trace("BLEND_MODE");
        `DCR_ROP_BLEND_SRC:     dpi_trace("BLEND_SRC");
        `DCR_ROP_BLEND_DST:     dpi_trace("BLEND_DST");
        `DCR_ROP_BLEND_CONST:   dpi_trace("BLEND_CONST");
        `DCR_ROP_LOGIC_OP:      dpi_trace("LOGIC_OP");        
        default:                dpi_trace("??");
    endcase  
endtask

`include "VX_rop_types.vh"

`IGNORE_WARNINGS_BEGIN
import rop_types::*;
`IGNORE_WARNINGS_END

`endif