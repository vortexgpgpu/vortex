`ifndef VX_TEX_CSR_IF
`define VX_TEX_CSR_IF

`include "VX_define.vh"

interface VX_tex_csr_if ();

    wire                      write_enable; 
    wire [`CSR_ADDR_BITS-1:0] write_addr;
    wire [31:0]               write_data;  

    modport master (
        output write_enable,
        output write_addr,
        output write_data
    );

    modport slave (
        input write_enable,
        input write_addr,
        input write_data
    );

endinterface

`endif