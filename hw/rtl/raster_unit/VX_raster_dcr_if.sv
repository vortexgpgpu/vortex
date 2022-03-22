`ifndef VX_RASTER_DCR_IF
`define VX_RASTER_DCR_IF

`include "VX_raster_define.vh"

interface VX_raster_dcr_if ();

    wire [`VX_DCR_ADDR_WIDTH-1:0]   addr;
    wire                            valid;
    wire [`VX_DCR_DATA_WIDTH-1:0]   in_data;
    raster_dcrs_t data;

    modport master (
        input data,
        output addr,
        output valid,
        output in_data
    );

    modport slave (
        input addr,
        input valid,
        input in_data,
        output data
    );

endinterface

`endif