`ifndef VX_EXU_TO_CMT_IF
`define VX_EXU_TO_CMT_IF

`include "VX_define.vh"

interface VX_exu_to_cmt_if ();

    wire                           valid;
    wire [`ISTAG_BITS-1:0]         issue_tag;     
    wire [`NUM_THREADS-1:0][31:0]  data;   

endinterface

`endif