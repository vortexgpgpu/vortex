`ifndef VX_TEX_CSR_IF
`define VX_TEX_CSR_IF

`include "VX_define.vh"

interface VX_tex_csr_if ();

    // wire                      read_enable;
    // wire[`CSR_ADDR_BITS-1:0]  read_addr;
    // wire[`NW_BITS-1:0]        read_wid;
    // wire[31:0]               read_data;

    wire                      write_enable; 
    wire[`CSR_ADDR_BITS-1:0]  write_addr;
    // wire[`NW_BITS-1:0]        write_wid;
    wire[`CSR_WIDTH-1:0]      write_data;   

endinterface

`endif