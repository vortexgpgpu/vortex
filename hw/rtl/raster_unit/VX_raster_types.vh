`include "VX_raster_define.vh"

package VX_raster_types;

typedef struct packed {
    logic [`RASTER_DCR_DATA_BITS-1:0] tbuf_addr;     // Tile buffer address
    logic [`RASTER_DCR_DATA_BITS-1:0] tile_count;    // Number of tiles in the tile buffer
    logic [`RASTER_DCR_DATA_BITS-1:0] pbuf_addr;     // Primitive (triangle) data buffer start address
    logic [`RASTER_DCR_DATA_BITS-1:0] pbuf_stride;   // Primitive data stride to fetch vertices
    logic [`RASTER_DIM_BITS-1:0]      dst_width;     // Destination window width
    logic [`RASTER_DIM_BITS-1:0]      dst_height;    // Destination window height
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

task trace_raster_state (
    input int                  level,
    input [`DCR_ADDR_BITS-1:0] state
);
    case (state)
        `DCR_RASTER_TBUF_ADDR:   dpi_trace(level, "TBUF_ADDR");     
        `DCR_RASTER_TILE_COUNT:  dpi_trace(level, "TILE_COUNT");
        `DCR_RASTER_PBUF_ADDR:   dpi_trace(level, "PBUF_ADDR");
        `DCR_RASTER_PBUF_STRIDE: dpi_trace(level, "PBUF_STRIDE");
        `DCR_RASTER_DST_SIZE:    dpi_trace(level, "DST_SIZE");
        default:                 dpi_trace(level, "?");
    endcase  
endtask

task trace_raster_csr (
    input int                  level,
    input [`CSR_ADDR_BITS-1:0] addr
);
    case (addr)
        `CSR_RASTER_POS_MASK:   dpi_trace(level, "POS_MASK");
        `CSR_RASTER_BCOORD_X0:  dpi_trace(level, "BCOORD_X0");
        `CSR_RASTER_BCOORD_X1:  dpi_trace(level, "BCOORD_X1");
        `CSR_RASTER_BCOORD_X2:  dpi_trace(level, "BCOORD_X2");
        `CSR_RASTER_BCOORD_X3:  dpi_trace(level, "BCOORD_X3");
        `CSR_RASTER_BCOORD_Y0:  dpi_trace(level, "BCOORD_Y0");
        `CSR_RASTER_BCOORD_Y1:  dpi_trace(level, "BCOORD_Y1");
        `CSR_RASTER_BCOORD_Y2:  dpi_trace(level, "BCOORD_Y2");
        `CSR_RASTER_BCOORD_Y3:  dpi_trace(level, "BCOORD_Y3");
        `CSR_RASTER_BCOORD_Z0:  dpi_trace(level, "BCOORD_Z0");
        `CSR_RASTER_BCOORD_Z1:  dpi_trace(level, "BCOORD_Z1");
        `CSR_RASTER_BCOORD_Z2:  dpi_trace(level, "BCOORD_Z2");
        `CSR_RASTER_BCOORD_Z3:  dpi_trace(level, "BCOORD_Z3");
        default:                dpi_trace(level, "?");
    endcase  
endtask

endpackage