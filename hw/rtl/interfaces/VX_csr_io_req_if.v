`ifndef VX_CSR_IO_REQ_IF
`define VX_CSR_IO_REQ_IF

`include "VX_define.vh"

interface VX_csr_io_req_if ();

    wire                      valid;    
    wire [`CSR_ADDR_BITS-1:0] addr;
    wire                      rw;
    wire [31:0]               data;
    wire                      ready;
    
endinterface

`endif
