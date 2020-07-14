`ifndef VX_CSR_IO_REQ_IF
`define VX_CSR_IO_REQ_IF

`include "VX_define.vh"

interface VX_csr_io_req_if ();

    wire        valid;
    wire        rw;
    wire [11:0] addr;
    wire [31:0] data;
    wire        ready;
    
endinterface

`endif
