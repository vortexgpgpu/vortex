`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

interface VX_dcr_base_if ();

    base_dcrs_t data;

    modport master (
        output data
    );

    modport slave (
        input data
    );

endinterface
