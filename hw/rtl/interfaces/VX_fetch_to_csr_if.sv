`ifndef VX_FETCH_TO_CSR_IF
`define VX_FETCH_TO_CSR_IF

`include "VX_define.vh"

interface VX_fetch_to_csr_if ();

    wire [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks;

    modport master (
        output thread_masks
    );

    modport slave (
        input  thread_masks
    );

endinterface

`endif