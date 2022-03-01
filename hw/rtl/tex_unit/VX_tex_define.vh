`ifndef VX_TEX_DEFINE
`define VX_TEX_DEFINE

`include "VX_define.vh"

`define TCACHE_TAG_ID_BITS  2
`define TCACHE_TAG_WIDTH    (`UUID_BITS + `TCACHE_TAG_ID_BITS)
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

`define TEX_BLEND_FRAC      8
`define TEX_BLEND_ONE       (2 ** `TEX_BLEND_FRAC)

task trace_tex_state (
    input [`CSR_ADDR_BITS-1:0] state
);
    case (state)
        `CSR_TEX_ADDR:      dpi_trace("ADDR");     
        `CSR_TEX_LOGDIM:    dpi_trace("LOGDIM");
        `CSR_TEX_FORMAT:    dpi_trace("FORMAT");
        `CSR_TEX_FILTER:    dpi_trace("FILTER");
        `CSR_TEX_WRAP:      dpi_trace("WRAP");
        //`CSR_TEX_MIPOFF:   
        default:            dpi_trace("MIPOFF");
    endcase  
endtask

`include "VX_tex_types.vh"

`IGNORE_WARNINGS_BEGIN
import tex_types::*;
`IGNORE_WARNINGS_END

`endif