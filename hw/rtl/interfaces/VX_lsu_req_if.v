`ifndef VX_LSU_REQ_IF
`define VX_LSU_REQ_IF

`include "VX_define.vh"

interface VX_lsu_req_if ();

    wire [`NUM_THREADS-1:0]         valid;
    wire [31:0]                     curr_PC;
    wire [`NW_BITS-1:0]             warp_num;
    wire [`NUM_THREADS-1:0][31:0]   store_data;
    wire [`NUM_THREADS-1:0][31:0]   base_addr;
    wire [31:0]                     offset;   
    wire                            rw; 
    wire [`BYTEEN_BITS-1:0]         byteen;
    wire [`NR_BITS-1:0]             rd; 
    wire [`WB_BITS-1:0]             wb; 
    wire                            ready;

endinterface

`endif