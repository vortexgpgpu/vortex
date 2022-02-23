`ifndef VX_ROP_CSR_IF
`define VX_ROP_CSR_IF

`include "VX_rop_define.vh"

interface VX_rop_csr_if ();

    rop_csrs_t data;

    modport master (
        output data
    );

    modport slave (
        input  data
    );

endinterface

`endif