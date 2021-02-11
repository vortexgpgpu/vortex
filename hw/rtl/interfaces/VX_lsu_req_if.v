`ifndef VX_LSU_REQ_IF
`define VX_LSU_REQ_IF

`include "VX_define.vh"

interface VX_lsu_req_if ();

    wire                            valid;
    wire [`NW_BITS-1:0]             wid;
    wire [`NUM_THREADS-1:0]         tmask;    
    wire [31:0]                     PC;
    wire [`LSU_BITS-1:0]            op_type;
    wire [`NUM_THREADS-1:0][31:0]   store_data;
    wire [`NUM_THREADS-1:0][31:0]   base_addr;    
    wire [31:0]                     offset;
    wire [`NR_BITS-1:0]             rd;
    wire                            wb;
    wire                            ready;

endinterface

`endif