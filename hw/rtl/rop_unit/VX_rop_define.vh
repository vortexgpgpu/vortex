`ifndef VX_ROP_DEFINE_VH
`define VX_ROP_DEFINE_VH

`include "VX_define.vh"
`include "VX_rop_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_rop_types::*;
`IGNORE_WARNINGS_END

task trace_rop_state (
    input int                  level,
    input [`DCR_ADDR_BITS-1:0] state
);
    case (state)
        `DCR_ROP_CBUF_ADDR:         dpi_trace(level, "CBUF_ADDR");
        `DCR_ROP_CBUF_PITCH:        dpi_trace(level, "CBUF_PITCH");
        `DCR_ROP_CBUF_WRITEMASK:    dpi_trace(level, "CBUF_WRITEMASK");
        `DCR_ROP_ZBUF_ADDR:         dpi_trace(level, "ZBUF_ADDR");
        `DCR_ROP_ZBUF_PITCH:        dpi_trace(level, "ZBUF_PITCH");
        `DCR_ROP_DEPTH_FUNC:        dpi_trace(level, "DEPTH_FUNC");
        `DCR_ROP_DEPTH_WRITEMASK:   dpi_trace(level, "DEPTH_WRITEMASK");
        `DCR_ROP_STENCIL_FUNC:      dpi_trace(level, "STENCIL_FUNC");        
        `DCR_ROP_STENCIL_ZPASS:     dpi_trace(level, "STENCIL_ZPASS");
        `DCR_ROP_STENCIL_ZFAIL:     dpi_trace(level, "STENCIL_ZFAIL");
        `DCR_ROP_STENCIL_FAIL:      dpi_trace(level, "STENCIL_FAIL");
        `DCR_ROP_STENCIL_REF:       dpi_trace(level, "STENCIL_REF");
        `DCR_ROP_STENCIL_MASK:      dpi_trace(level, "STENCIL_MASK");        
        `DCR_ROP_STENCIL_WRITEMASK: dpi_trace(level, "STENCIL_WRITEMASK");
        `DCR_ROP_BLEND_MODE:        dpi_trace(level, "BLEND_MODE");
        `DCR_ROP_BLEND_FUNC:        dpi_trace(level, "BLEND_FUNC");
        `DCR_ROP_BLEND_CONST:       dpi_trace(level, "BLEND_CONST");
        `DCR_ROP_LOGIC_OP:          dpi_trace(level, "LOGIC_OP");        
        default:                    dpi_trace(level, "?");
    endcase  
endtask

task trace_rop_csr (
    input int                  level,
    input [`CSR_ADDR_BITS-1:0] addr
);
    case (addr)
        `CSR_ROP_RT_IDX:        dpi_trace(level, "RT_IDX");
        `CSR_ROP_SAMPLE_IDX:    dpi_trace(level, "SAMPLE_IDX");
        default:                dpi_trace(level, "?");
    endcase  
endtask

`endif
