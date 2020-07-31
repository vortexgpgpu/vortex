`ifndef VX_CMT_TO_CSR_IF
`define VX_CMT_TO_CSR_IF

`include "VX_define.vh"

interface VX_cmt_to_csr_if ();

    wire valid;

    wire [`NW_BITS-1:0] warp_num;

    wire [`NE_BITS:0] num_commits;

    wire upd_fflags;
    wire [`FFG_BITS-1:0] fflags;

endinterface

`endif