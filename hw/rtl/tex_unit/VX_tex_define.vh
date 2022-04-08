`ifndef VX_TEX_DEFINE_VH
`define VX_TEX_DEFINE_VH

`include "VX_define.vh"
`include "VX_tex_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_tex_types::*;
`IGNORE_WARNINGS_END

task trace_tex_dcr (
    input int                  level,
    input [`DCR_ADDR_BITS-1:0] addr
);
    case (addr)
        `DCR_TEX_ADDR:      dpi_trace(level, "ADDR");     
        `DCR_TEX_LOGDIM:    dpi_trace(level, "LOGDIM");
        `DCR_TEX_FORMAT:    dpi_trace(level, "FORMAT");
        `DCR_TEX_FILTER:    dpi_trace(level, "FILTER");
        `DCR_TEX_WRAP:      dpi_trace(level, "WRAP");
        //`DCR_TEX_MIPOFF
        default:            dpi_trace(level, "MIPOFF");
    endcase  
endtask

task trace_tex_csr (
    input int                  level,
    input [`CSR_ADDR_BITS-1:0] addr
);
    case (addr)
        `CSR_TEX_STAGE:  dpi_trace(level, "STAGE"); 
        default:         dpi_trace(level, "?");
    endcase  
endtask

`endif
