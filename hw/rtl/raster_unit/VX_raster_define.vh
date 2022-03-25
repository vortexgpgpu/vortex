`ifndef VX_RASTER_DEFINE
`define VX_RASTER_DEFINE

`include "VX_define.vh"
`include "VX_types.vh"

`define RASTER_ADDR_BITS            32  
`define RASTER_DCR_DATA_BITS        32
`define RASTER_PRIMITIVE_DATA_BITS  32

task trace_raster_state (
    input [`DCR_ADDR_BITS-1:0] state
);
    case (state)
        `DCR_RASTER_TBUF_ADDR:   dpi_trace("TBUF_ADDR");     
        `DCR_RASTER_TILE_COUNT:  dpi_trace("TILE_COUNT");
        `DCR_RASTER_PBUF_ADDR:   dpi_trace("PBUF_ADDR");
        `DCR_RASTER_PBUF_STRIDE: dpi_trace("PBUF_STRIDE");
        default:                 dpi_trace("?");
    endcase  
endtask

task trace_raster_csr (
    input [`CSR_ADDR_BITS-1:0] addr
);
    case (addr)
        `CSR_RASTER_POS_MASK:   dpi_trace("POS_MASK");
        `CSR_RASTER_BCOORD_X0:  dpi_trace("BCOORD_X0");
        `CSR_RASTER_BCOORD_X1:  dpi_trace("BCOORD_X1");
        `CSR_RASTER_BCOORD_X2:  dpi_trace("BCOORD_X2");
        `CSR_RASTER_BCOORD_X3:  dpi_trace("BCOORD_X3");
        `CSR_RASTER_BCOORD_Y0:  dpi_trace("BCOORD_Y0");
        `CSR_RASTER_BCOORD_Y1:  dpi_trace("BCOORD_Y1");
        `CSR_RASTER_BCOORD_Y2:  dpi_trace("BCOORD_Y2");
        `CSR_RASTER_BCOORD_Y3:  dpi_trace("BCOORD_Y3");
        `CSR_RASTER_BCOORD_Z0:  dpi_trace("BCOORD_Z0");
        `CSR_RASTER_BCOORD_Z1:  dpi_trace("BCOORD_Z1");
        `CSR_RASTER_BCOORD_Z2:  dpi_trace("BCOORD_Z2");
        `CSR_RASTER_BCOORD_Z3:  dpi_trace("BCOORD_Z3");
        default:                dpi_trace("?");
    endcase  
endtask

`include "VX_raster_types.vh"

`IGNORE_WARNINGS_BEGIN
import raster_types::*;
`IGNORE_WARNINGS_END

`endif
