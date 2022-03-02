`ifndef VX_RASTER_TYPES
`define VX_RASTER_TYPES

`include "VX_define.vh"

`define RASTER_DIM_BITS             15

package raster_types;

typedef struct packed {
    logic [31:0]    tbuf_addr;
    logic [31:0]    tile_count;
    logic [31:0]    pbuf_addr;
    logic [31:0]    pbuf_stride;
    logic [15:0]    tile_logsize;
} raster_dcrs_t;

endpackage

`endif