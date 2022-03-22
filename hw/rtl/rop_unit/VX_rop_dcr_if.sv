`ifndef VX_ROP_DCR_IF
`define VX_ROP_DCR_IF

`include "VX_rop_define.vh"

interface VX_rop_dcr_if ();

    rop_dcrs_t data;

    modport master (
        input data
    );

    modport slave (
        output data
    );

endinterface

`endif
