`ifndef VX_RASTER_TYPES
`define VX_RASTER_TYPES

`include "VX_define.vh"

package raster_types;

typedef struct packed {
    logic [`RASTER_DCR_DATA_BITS-1:0]    tbuf_addr;      // Tile buffer address
    logic [`RASTER_DCR_DATA_BITS-1:0]    tile_count;     // Number of tiles in the tile buffer
    logic [`RASTER_DCR_DATA_BITS-1:0]    pbuf_addr;      // Primitive (triangle) data buffer start address
    logic [`RASTER_DCR_DATA_BITS-1:0]    pbuf_stride;    // Primitive data stride to fetch vertices
} raster_dcrs_t;

endpackage

`endif