`ifndef VX_FETCH_TO_CSR_IF
`define VX_FETCH_TO_CSR_IF

`include "VX_define.vh"

interface VX_fetch_to_csr_if ();

    wire [`PERF_CTR_BITS-1:0] cycles;

    modport master (
        output cycles
    );

    modport slave (
        input  cycles
    );

endinterface

`endif
