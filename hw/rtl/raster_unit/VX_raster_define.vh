`ifndef VX_RASTER_DEFINE
`define VX_RASTER_DEFINE

`include "VX_define.vh"

// TODO

task trace_raster_state (
    input [`CSR_ADDR_BITS-1:0] state
);
    case (state)
        `CSR_RASTER_PIDX_ADDR:   dpi_trace("PIDX_ADDR");     
        `CSR_RASTER_PIDX_SIZE:   dpi_trace("PIDX_SIZE");
        `CSR_RASTER_PBUF_ADDR:   dpi_trace("PBUF_ADDR");
        `CSR_RASTER_PBUF_STRIDE: dpi_trace("PBUF_STRIDE");
        `CSR_RASTER_TILE_XY:     dpi_trace("TILE_XY");
        `CSR_RASTER_TILE_WH:     dpi_trace("TILE_WH");
        default:                 dpi_trace("??");
    endcase  
endtask

`include "VX_raster_types.vh"

`IGNORE_WARNINGS_BEGIN
import raster_types::*;
`IGNORE_WARNINGS_END

`endif