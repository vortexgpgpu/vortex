`ifndef VX_TEX_CSR_IF
`define VX_TEX_CSR_IF

`include "VX_tex_define.vh"

interface VX_tex_csr_if ();

    wire [`NTEX_BITS-1:0] stage;
    tex_csrs_t data;

    modport master (
        input  stage,
        output data
    );

    modport slave (
        output stage,
        input  data
    );

endinterface

`endif