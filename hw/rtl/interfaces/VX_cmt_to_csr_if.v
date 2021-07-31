`ifndef VX_CMT_TO_CSR_IF
`define VX_CMT_TO_CSR_IF

`include "VX_define.vh"

interface VX_cmt_to_csr_if #(
    parameter SIZE
)();

    wire            valid;
    wire [SIZE-1:0] commit_size;

endinterface

`endif