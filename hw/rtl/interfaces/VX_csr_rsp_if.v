`ifndef VX_CSR_RSP_IF
`define VX_CSR_RSP_IF

`include "VX_define.vh"

interface VX_csr_rsp_if ();

    wire                           valid;
    wire [`ISTAG_BITS-1:0]         issue_tag;     
    wire [`NUM_THREADS-1:0][31:0]  data; 
    wire                           ready;  

endinterface

`endif