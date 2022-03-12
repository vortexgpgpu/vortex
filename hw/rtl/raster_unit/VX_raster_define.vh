`ifndef VX_RASTER_DEFINE
`define VX_RASTER_DEFINE

`include "VX_define.vh"

`define RASTER_ADDR_BITS            32  
`define RASTER_DCR_DATA_BITS        32
`define RASTER_TILE_DATA_BITS       16
`define RASTER_PRIMITIVE_DATA_BITS  32

task trace_raster_state (
    input [`DCR_ADDR_BITS-1:0] state
);
    case (state)
        `DCR_RASTER_TBUF_ADDR:   dpi_trace("TBUF_ADDR");     
        `DCR_RASTER_TILE_COUNT:  dpi_trace("TILE_COUNT");
        `DCR_RASTER_PBUF_ADDR:   dpi_trace("PBUF_ADDR");
        `DCR_RASTER_PBUF_STRIDE: dpi_trace("PBUF_STRIDE");
        default:                 dpi_trace("??");
    endcase  
endtask

`include "VX_raster_types.vh"

`IGNORE_WARNINGS_BEGIN
import raster_types::*;
`IGNORE_WARNINGS_END

`endif