`ifndef VX_CSR_REQ_IF
`define VX_CSR_REQ_IF

`include "VX_define.vh"

interface VX_csr_req_if ();

    wire                    valid;
    wire [`ISTAG_BITS-1:0]  issue_tag;
    wire [`NW_BITS-1:0]     wid;
`DEBUG_BEGIN
    wire [`NUM_THREADS-1:0] thread_mask;
`DEBUG_END
    wire [31:0]             curr_PC;  

    wire [`CSR_BITS-1:0]    op;

    wire [`CSR_ADDR_BITS-1:0] csr_addr;
    wire [31:0]             csr_mask;

    wire [`NR_BITS-1:0]     rd;
    wire                    wb;
    wire                    is_io;  
    
    wire                    ready;
    
endinterface

`endif
