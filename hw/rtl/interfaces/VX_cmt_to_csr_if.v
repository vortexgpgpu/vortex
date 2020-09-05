`ifndef VX_CMT_TO_CSR_IF
`define VX_CMT_TO_CSR_IF

`include "VX_define.vh"

interface VX_cmt_to_csr_if ();

    wire valid;

    wire [`NW_BITS-1:0] wid;

    wire [$clog2(`NUM_EXS+1)-1:0] num_commits;

    wire has_fflags;
    fflags_t fflags;

endinterface

`endif