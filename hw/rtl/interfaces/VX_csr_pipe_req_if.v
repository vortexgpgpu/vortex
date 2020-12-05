`ifndef VX_CSR_PIPE_REQ_IF
`define VX_CSR_PIPE_REQ_IF

`include "VX_define.vh"

interface VX_csr_pipe_req_if ();

    wire                    valid;

    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] tmask;
    wire [31:0]             PC;
    wire [`CSR_BITS-1:0]    op_type;
    wire [`CSR_ADDR_BITS-1:0] csr_addr;
    wire [31:0]             csr_mask;
    wire [`NR_BITS-1:0]     rd;
    wire                    wb;
    wire                    is_io;  
    
    wire                    ready;
    
endinterface

`endif