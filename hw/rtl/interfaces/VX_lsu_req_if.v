
`ifndef VX_LSU_REQ_IF
`define VX_LSU_REQ_IF

`include "VX_define.vh"

interface VX_lsu_req_if ();

    wire [`NUM_THREADS-1:0]         valid;
    wire [31:0]                     curr_PC;
    wire [`NW_BITS-1:0]             warp_num;
    wire [`NUM_THREADS-1:0][31:0]   store_data;
    wire [`NUM_THREADS-1:0][31:0]   base_addr;  // A reg data
    wire [31:0]                     offset;     // itype_immed
    wire [`BYTE_EN_BITS-1:0]        mem_read; 
    wire [`BYTE_EN_BITS-1:0]        mem_write;
    wire [4:0]                      rd; // dest register
    wire [1:0]                      wb; //

endinterface

`endif