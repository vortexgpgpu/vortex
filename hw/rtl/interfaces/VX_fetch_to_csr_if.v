`ifndef VX_FETCH_TO_CSR_IF
`define VX_FETCH_TO_CSR_IF

`include "VX_define.vh"

interface VX_fetch_to_csr_if ();

    wire [`NUM_THREADS-1:0] thread_masks [`NUM_WARPS-1:0];

endinterface

`endif