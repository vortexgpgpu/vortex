`ifndef VX_RASTER_TYPES_VH
`define VX_RASTER_TYPES_VH

`include "VX_define.vh"

`define RASTER_DCR_DATA_BITS    32

`define RASTER_DATA_BITS        32
`define RASTER_STRIDE_BITS      16
`define RASTER_TILE_BITS        16
`define RASTER_PID_BITS         16

package VX_raster_types;

typedef struct packed {
    logic [`RASTER_DCR_DATA_BITS-1:0] tbuf_addr;     // Tile buffer address
    logic [`RASTER_TILE_BITS-1:0]     tile_count;    // Number of tiles in the tile buffer
    logic [`RASTER_DCR_DATA_BITS-1:0] pbuf_addr;     // Primitive (triangle) data buffer start address
    logic [`RASTER_STRIDE_BITS-1:0]   pbuf_stride;   // Primitive data stride to fetch vertices
    logic [`RASTER_DIM_BITS-1:0]      dst_width;     // Destination window width
    logic [`RASTER_DIM_BITS-1:0]      dst_height;    // Destination window height
} raster_dcrs_t;

typedef struct packed {
    logic [`RASTER_DIM_BITS-2:0] pos_x;     // quad x position
    logic [`RASTER_DIM_BITS-2:0] pos_y;     // quad y position
    logic [3:0]                  mask;      // quad mask
    logic [2:0][3:0][31:0]       bcoords;   // barycentric coordinates
    logic [`RASTER_PID_BITS-1:0] pid;       // primitive index
} raster_stamp_t;

typedef struct packed {
    logic [31:0]           pos_mask;
    logic [2:0][3:0][31:0] bcoords;
} raster_csrs_t;

endpackage

`endif
