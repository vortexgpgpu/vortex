`ifndef VX_CSR_IO_RSP_IF
`define VX_CSR_IO_RSP_IF

`include "VX_define.vh"

interface VX_csr_io_rsp_if ();

    wire        valid;    
    wire [31:0] data;
    wire        ready;
    
endinterface

`endif
