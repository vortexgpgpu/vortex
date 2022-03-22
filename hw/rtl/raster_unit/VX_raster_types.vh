`ifndef VX_RASTER_TYPES
`define VX_RASTER_TYPES

`include "VX_define.vh"

package raster_types;

typedef struct packed {
    logic [`RASTER_DCR_DATA_BITS-1:0] tbuf_addr;     // Tile buffer address
    logic [`RASTER_DCR_DATA_BITS-1:0] tile_count;    // Number of tiles in the tile buffer
    logic [`RASTER_DCR_DATA_BITS-1:0] pbuf_addr;     // Primitive (triangle) data buffer start address
    logic [`RASTER_DCR_DATA_BITS-1:0] pbuf_stride;   // Primitive data stride to fetch vertices
    logic [`RASTER_DIM_BITS-1:0]      dst_width;     // window width
    logic [`RASTER_DIM_BITS-1:0]      dst_height;    // window height
} raster_dcrs_t;

typedef struct packed {
    logic [`RASTER_DIM_BITS-2:0] pos_x;     // quad x position
    logic [`RASTER_DIM_BITS-2:0] pos_y;     // quad y position
    logic [3:0]                  mask;      // quad mask
    logic [3:0][31:0]            bcoord_x;  // barycentric coordinates x
    logic [3:0][31:0]            bcoord_y;  // barycentric coordinates y
    logic [3:0][31:0]            bcoord_z;  // barycentric coordinates z
    logic [`RASTER_PID_BITS-1:0] pid;       // primitive index
} raster_stamp_t;

typedef struct packed {
    logic [31:0]      pos_mask;
    logic [3:0][31:0] bcoord_x;
    logic [3:0][31:0] bcoord_y;
    logic [3:0][31:0] bcoord_z; 
} raster_csrs_t;

endpackage

`endif
