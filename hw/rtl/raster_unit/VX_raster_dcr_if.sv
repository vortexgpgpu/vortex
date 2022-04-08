`include "VX_raster_define.vh"

import VX_raster_types::*;

interface VX_raster_dcr_if ();

    raster_dcrs_t data;

    modport master (
        output data
    );

    modport slave (
        input data
    );

endinterface
