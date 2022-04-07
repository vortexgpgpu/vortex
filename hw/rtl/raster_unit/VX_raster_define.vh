`ifndef VX_RASTER_DEFINE
`define VX_RASTER_DEFINE

`include "VX_define.vh"
`include "VX_types.vh"

`define RASTER_ADDR_BITS            32  
`define RASTER_DCR_DATA_BITS        32
`define RASTER_PRIMITIVE_DATA_BITS  32

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

`include "VX_raster_types.vh"

`IGNORE_WARNINGS_BEGIN
import raster_types::*;
`IGNORE_WARNINGS_END

`endif
