`ifndef VX_MEM_REQ_IF
`define VX_MEM_REQ_IF

`include "../cache/VX_cache_define.vh"

interface VX_mem_req_if #(
    parameter LINE_WIDTH = 1,
    parameter ADDR_WIDTH = 1,
    parameter TAG_WIDTH  = 1,
    parameter LINE_SIZE  = LINE_WIDTH / 8
) ();

    wire                    valid;    
    wire                    rw;    
    wire [LINE_SIZE-1:0]    byteen;
    wire [ADDR_WIDTH-1:0]   addr;
    wire [LINE_WIDTH-1:0]   data;  
    wire [TAG_WIDTH-1:0]    tag;  
    wire                    ready;

endinterface

`endif