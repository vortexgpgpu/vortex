`include "VX_tex_define.vh"

package VX_tex_types;

typedef struct packed {
    logic [(`TEX_LOD_MAX+1)-1:0][`TEX_MIPOFF_BITS-1:0] mipoff;
    logic [1:0][`TEX_LOD_BITS-1:0]  logdims;
    logic [1:0][`TEX_WRAP_BITS-1:0] wraps;
    logic [`TEX_ADDR_BITS-1:0]      baddr;
    logic [`TEX_FORMAT_BITS-1:0]    format;
    logic [`TEX_FILTER_BITS-1:0]    filter;
} tex_dcrs_t;

typedef struct packed {
    logic [`TEX_STAGE_BITS-1:0] stage;
} tex_csrs_t;

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

endpackage