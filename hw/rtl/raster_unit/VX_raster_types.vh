`ifndef VX_RASTER_TYPES
`define VX_RASTER_TYPES

`include "VX_define.vh"

package raster_types;

typedef struct packed {
    logic [(`TEX_LOD_MAX+1)-1:0][`TEX_MIPOFF_BITS-1:0] mipoff;
    logic [1:0][`TEX_LOD_BITS-1:0]  logdims;
    logic [1:0][`TEX_WRAP_BITS-1:0] wraps;
    logic [`TEX_ADDR_BITS-1:0]      baddr;
    logic [`TEX_FORMAT_BITS-1:0]    format;
    logic [`TEX_FILTER_BITS-1:0]    filter;
} raster_csrs_t;

endpackage

`endif