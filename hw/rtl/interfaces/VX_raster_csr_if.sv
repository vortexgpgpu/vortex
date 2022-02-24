`ifndef VX_RASTER_CSR_IF
`define VX_RASTER_CSR_IF

`include "VX_raster_define.vh"

interface VX_raster_csr_if ();

    raster_csrs_t data;

    modport master (
        output data
    );

    modport slave (
        input  data
    );

endinterface

`endif