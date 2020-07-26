`ifndef VX_COMMIT_IF
`define VX_COMMIT_IF

`include "VX_define.vh"

interface VX_commit_if ();

    wire                            valid;
    wire [`ISTAG_BITS-1:0]          issue_tag;     
    wire [`NUM_THREADS-1:0][31:0]	data;         
    wire                            ready;

endinterface

`endif