`ifndef VX_RASTER_TYPES
`define VX_RASTER_TYPES

`include "VX_define.vh"

package raster_types;

typedef struct packed {
    logic [`RASTER_DCR_DATA_BITS-1:0]    pidx_addr;      // Fetch index for pixels
    logic [`RASTER_DCR_DATA_BITS-1:0]    pidx_size;      // Fetch size for pixels
    logic [`RASTER_DCR_DATA_BITS-1:0]    pbuf_addr;      // Primitive (triangle) data buffer start address
    logic [`RASTER_DCR_DATA_BITS-1:0]    pbuf_stride;    // Primitive data stride to fetch vertices
    logic [`RASTER_TILE_DATA_BITS-1:0]   tile_left;      // left location of input tile
    logic [`RASTER_TILE_DATA_BITS-1:0]   tile_top;       // top location of input tile
    logic [`RASTER_TILE_DATA_BITS-1:0]   tile_width;     // width of input tile
    logic [`RASTER_TILE_DATA_BITS-1:0]   tile_height;    // heigth of input tile
} raster_dcrs_t;

endpackage

`endif