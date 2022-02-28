`ifndef VX_RASTER_DEFINE
`define VX_RASTER_DEFINE

`include "VX_define.vh"

// TODO

task trace_raster_state (
    input [`CSR_ADDR_BITS-1:0] state
);
    case (state)
        `CSR_RASTER_TBUF_ADDR:   dpi_trace("TBUF_ADDR");     
        `CSR_RASTER_TILE_COUNT:  dpi_trace("TILE_COUNT");
        `CSR_RASTER_PBUF_ADDR:   dpi_trace("PBUF_ADDR");
        `CSR_RASTER_PBUF_STRIDE: dpi_trace("PBUF_STRIDE");
        `CSR_RASTER_TILE_LOGSIZE:dpi_trace("TILE_LOGSIZE);
        default:                 dpi_trace("??");
    endcase  
endtask

`include "VX_raster_types.vh"

`IGNORE_WARNINGS_BEGIN
import raster_types::*;
`IGNORE_WARNINGS_END

`endif