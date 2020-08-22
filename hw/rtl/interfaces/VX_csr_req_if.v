`ifndef VX_CSR_REQ_IF
`define VX_CSR_REQ_IF

`include "VX_define.vh"

interface VX_csr_req_if ();

    wire                    valid;

    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] thread_mask;
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