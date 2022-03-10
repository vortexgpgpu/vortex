`ifndef VX_RASTER_DEFINE
`define VX_RASTER_DEFINE

`include "VX_define.vh"

// TODO
`define RASTER_ADDR_BITS            32  
`define RASTER_CSR_DATA_BITS        32
`define RASTER_TILE_DATA_BITS       16
`define RASTER_PRIMITIVE_DATA_BITS  32
// using equal size to ease calculation and storage
//`define RASTER_EXTENT_DATA_BITS     16

`define RASTER_BLOCK_SIZE           8
`define RASTER_BLOCK_SIZE_BITS      $clog2(`RASTER_BLOCK_SIZE)
//`define RASTER_QUAD_NUM             4
`define RASTER_QUAD_NUM             `RASTER_BLOCK_SIZE/2

`define RASTER_TILE_SIZE            16
`define RASTER_TILE_SIZE_BITS       $clog2(`RASTER_TILE_SIZE)

//`define RASTER_LEVEL_DATA_BITS      (`RASTER_TILE_DATA_BITS - `RASTER_BLOCK_SIZE_BITS)
`define RASTER_LEVEL_DATA_BITS      $clog2(`RASTER_TILE_SIZE/`RASTER_BLOCK_SIZE) + 1
`define RASTER_FIFO_DATA_WIDTH      (`RASTER_LEVEL_DATA_BITS + 2*`RASTER_TILE_DATA_BITS + 3*`RASTER_PRIMITIVE_DATA_BITS)
`define RASTER_TILE_FIFO_DEPTH      8

`define RASTER_QUAD_OUTPUT_RATE     4
`define RASTER_QUAD_FIFO_DEPTH      64

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